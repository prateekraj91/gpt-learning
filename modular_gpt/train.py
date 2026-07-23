import time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
import wandb

from config import BASE_CONFIG
from model import GPT

# -----------------------------
# Device
# -----------------------------

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# -----------------------------
# Dataset
# -----------------------------

dataset = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1"
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_function(example):
    return tokenizer(example["text"])

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# -----------------------------
# Build token streams
# -----------------------------

train_ids = []
for ex in tokenized_dataset["train"]:
    train_ids.extend(ex["input_ids"])

val_ids = []
for ex in tokenized_dataset["validation"]:
    val_ids.extend(ex["input_ids"])

train_ids = torch.tensor(train_ids)
val_ids = torch.tensor(val_ids)

# -----------------------------
# Batch loader
# -----------------------------

def get_batch(data, batch_size, seq_len):

    ix = torch.randint(
        0,
        len(data) - seq_len - 1,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i+seq_len]
        for i in ix
    ])

    y = torch.stack([
        data[i+1:i+seq_len+1]
        for i in ix
    ])

    return x.to(device), y.to(device)

# -----------------------------
# Validation
# -----------------------------

@torch.no_grad()
def evaluate(model, val_ids, seq_len):

    model.eval()

    losses = []

    for _ in range(20):

        x, y = get_batch(
            val_ids,
            batch_size=1,
            seq_len=seq_len
        )

        logits = model(x)

        B, T, C = logits.shape

        loss = F.cross_entropy(
            logits.reshape(B * T, C),
            y.reshape(B * T)
        )

        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)

    ppl = torch.exp(torch.tensor(avg_loss))

    model.train()

    return avg_loss, ppl.item()

# -----------------------------
# Inference benchmark
# -----------------------------

@torch.no_grad()
def benchmark_inference(model, seq_len):

    model.eval()

    x = torch.randint(
        0,
        50257,
        (1, seq_len)
    ).to(device)

    # Warmup
    for _ in range(10):
        _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()

    start = time.time()

    runs = 20

    for _ in range(runs):
        _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()

    elapsed = time.time() - start

    tokens_per_sec = (runs * seq_len) / elapsed

    model.train()

    return tokens_per_sec

# -----------------------------
# Run experiments
# -----------------------------

context_length = 1024
if True:

    POS_ENCODING = "learned"

    config = BASE_CONFIG.copy()

    config["seq_len"] = context_length
    config["pos_encoding"] = POS_ENCODING
    config["use_conv"] = False
    config["architecture"] = "interleaved"
    config["attention_type"] = "standard"

    run = wandb.init(
        project="coreML-SAIDL",
        group="positional_encodings",
        name=f"{config['architecture']}_conv{config['use_conv']}_{config['attention_type']}_{POS_ENCODING}_train512",
        config=config,
        reinit=True
    )

    model = GPT(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["train_steps"]
    )

    model.train()

    print(f"\n===== TRAINING CONTEXT {context_length} =====")

    training_losses = []

    epoch_start = time.time()

    # -----------------------------
    # Training loop
    # -----------------------------

    for step in range(config["train_steps"]):

        start_step = time.time()

        x, y = get_batch(
            train_ids,
            config["batch_size"],
            context_length
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        logits = model(x)

        B, T, C = logits.shape

        loss = F.cross_entropy(
            logits.reshape(B * T, C),
            y.reshape(B * T)
        )

        optimizer.zero_grad()

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        scheduler.step()

        ppl = torch.exp(loss).item()

        training_losses.append(loss.item())

        # -----------------------------
        # GPU memory
        # -----------------------------

        gpu_mem = (
            torch.cuda.max_memory_allocated() / 1e6
            if torch.cuda.is_available()
            else 0
        )

        # -----------------------------
        # Training throughput
        # -----------------------------

        train_tokens_per_sec = (
            config["batch_size"] *
            context_length
        ) / (time.time() - start_step)

        # -----------------------------
        # Training stability
        # -----------------------------

        stability = (
            torch.tensor(training_losses[-20:]).std().item()
            if len(training_losses) > 20
            else 0
        )

        # -----------------------------
        # Evaluation every 20 steps
        # -----------------------------

        if step % 20 == 0:

            inference_speed = benchmark_inference(
                model,
                context_length
            )

            val_loss, val_ppl = evaluate(
                model,
                val_ids,
                context_length
            )

            print(
                f"PE={POS_ENCODING} | CTX={context_length} | STEP={step} | "
                f"LOSS={loss.item():.4f} | PPL={ppl:.2f} | "
                f"VAL_LOSS={val_loss:.4f} | VAL_PPL={val_ppl:.2f} | "
                f"GPU={gpu_mem:.1f}MB"
            )

            wandb.log({

                # -----------------------------
                # Training metrics
                # -----------------------------

                "train_loss": loss.item(),
                "train_perplexity": ppl,
                "train_tokens_per_sec": train_tokens_per_sec,

                # -----------------------------
                # Validation metrics
                # -----------------------------

                "validation_loss": val_loss,
                "validation_perplexity": val_ppl,

                # -----------------------------
                # Memory / speed
                # -----------------------------

                "peak_gpu_memory_mb": gpu_mem,
                "inference_tokens_per_sec": inference_speed,

                # -----------------------------
                # Stability
                # -----------------------------

                "gradient_norm": grad_norm.item(),
                "loss_std_last_20": stability,

                # -----------------------------
                # Metadata
                # -----------------------------

                "context_length": context_length,
                "learning_rate": scheduler.get_last_lr()[0],

                "architecture": config["architecture"],
                "attention_type": config["attention_type"],
                "pos_encoding": config["pos_encoding"],
                "use_conv": config["use_conv"],

            }, step=step)

    # -----------------------------
    # Epoch timing
    # -----------------------------

    epoch_time = time.time() - epoch_start

    print(f"\n===== EXTRAPOLATION TEST PE={POS_ENCODING} =====")

    extrap_table = wandb.Table(columns=[
        "pos_encoding",
        "train_ctx",
        "eval_ctx",
        "val_loss",
        "val_ppl"
    ])

    for eval_ctx in [512, 1024, 2048]:
        e_loss, e_ppl = evaluate(model, val_ids, eval_ctx)
        print(f"EXTRAP | eval_ctx={eval_ctx} | VAL_LOSS={e_loss:.4f} | VAL_PPL={e_ppl:.2f}")

        # log as separate metrics so you can compare across runs in wandb
        wandb.log({
            f"extrap_ctx{eval_ctx}_val_loss": e_loss,
            f"extrap_ctx{eval_ctx}_val_ppl":  e_ppl,
        })

        extrap_table.add_data(
            POS_ENCODING,
            512,
            eval_ctx,
            e_loss,
            e_ppl
        )

    # -----------------------------
    # Comparative summary table
    # -----------------------------

    summary_table = wandb.Table(
        columns=[
        "pos_encoding",
        "train_ctx",
        "val_loss_512",
        "val_ppl_512",
        "val_loss_1024",
        "val_ppl_1024",
        "val_loss_2048",
        "val_ppl_2048",
        "peak_gpu_memory_mb",
        "train_tokens_per_sec",
        "inference_tokens_per_sec",
        "epoch_time_sec"
        ]
    )

    r512  = evaluate(model, val_ids, 512)
    r1024 = evaluate(model, val_ids, 1024)
    r2048 = evaluate(model, val_ids, 2048)

    summary_table.add_data(
    POS_ENCODING,
    512,
    r512[0],  r512[1],
    r1024[0], r1024[1],
    r2048[0], r2048[1],
    gpu_mem,
    train_tokens_per_sec,
    inference_speed,
    epoch_time
)

    wandb.log({
        "extrapolation_table": extrap_table,
        "comparison_table": summary_table,
        "epoch_time_sec": epoch_time
    })

    print(
        f"\nFinished Context Length {context_length} | "
        f"Epoch Time: {epoch_time:.2f} sec"
    )

    wandb.finish()