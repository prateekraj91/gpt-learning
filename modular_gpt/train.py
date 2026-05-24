import time
import torch
from torch.nn import functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from config import config
from model import GPT

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Dataset and Tokenizer
# -----------------------------
dataset = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1"
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=config["seq_len"]
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Convert to token stream
train_ids = []
for example in tokenized_dataset["train"]:
    train_ids.extend(example["input_ids"])

train_ids = torch.tensor(train_ids)

# -----------------------------
# Batch settings
# -----------------------------
batch_size = config["batch_size"]
seq_len = config["seq_len"]

# -----------------------------
# Batch function
# -----------------------------
def get_batch():
    ix = torch.randint(
        0,
        len(train_ids) - seq_len - 1,
        (batch_size,)
    )

    x = torch.stack([
        train_ids[i:i+seq_len]
        for i in ix
    ])

    y = torch.stack([
        train_ids[i+1:i+seq_len+1]
        for i in ix
    ])

    return x, y

# -----------------------------
# Model Initialization
# -----------------------------
model = GPT(config).to(device)

print("Model initialized with configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")
print("Total parameters:", sum(p.numel() for p in model.parameters()))

# -----------------------------
# Optimizer + Scheduler
# -----------------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"]
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=200
)

start_time = time.time()

# -----------------------------
# Training loop
# -----------------------------
model.train()

for step in range(200):
    x, y = get_batch()
    x = x.to(device)
    y = y.to(device)

    logits = model(x)
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    y = y.view(B * T)

    loss = F.cross_entropy(logits, y)
    ppl = torch.exp(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 10 == 0:
        elapsed = time.time() - start_time
        tokens_per_sec = (batch_size * seq_len * (step + 1)) / elapsed
        print(
            f"Step {step} | "
            f"Loss: {loss.item():.4f} | "
            f"PPL: {ppl.item():.4f} | "
            f"Tokens/sec: {tokens_per_sec:.2f}"
        )

# -----------------------------
# Validation
# -----------------------------
val_ids = []
for example in tokenized_dataset["validation"]:
    val_ids.extend(example["input_ids"])

val_ids = torch.tensor(val_ids)

def get_val_batch():
    ix = torch.randint(
        0,
        len(val_ids) - seq_len - 1,
        (batch_size,)
    )

    x = torch.stack([
        val_ids[i:i+seq_len]
        for i in ix
    ])

    y = torch.stack([
        val_ids[i+1:i+seq_len+1]
        for i in ix
    ])

    return x, y

model.eval()

with torch.no_grad():
    x, y = get_val_batch()
    x = x.to(device)
    y = y.to(device)

    logits = model(x)
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    y = y.view(B * T)

    val_loss = F.cross_entropy(logits, y)
    val_ppl = torch.exp(val_loss)

print("Validation Loss:", val_loss.item())
print("Validation Perplexity:", val_ppl.item())
