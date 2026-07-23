# =========================================================
# BASE CONFIG (single experiment settings)
# =========================================================

BASE_CONFIG = {
    # -----------------------------
    # Model size
    # -----------------------------
    "vocab_size": 50257,
    "d_model": 128,
    "num_heads": 4,
    "n_layers": 4,

    # -----------------------------
    # Sequence / batching
    # -----------------------------
    "seq_len": 1024,
    "batch_size": 2,

    # -----------------------------
    # Attention type
    # standard | sliding_window | linear | mqa
    # -----------------------------
    "attention_type": "standard",

    # -----------------------------
    # Positional encoding
    # rope | alibi | learned | relative
    # -----------------------------
    "pos_encoding": "learned",

    # -----------------------------
    # Sliding window size (only for sliding attention)
    # -----------------------------
    "window_size": 128,

    # -----------------------------
    # Optimizer
    # -----------------------------
    "learning_rate": 3e-4,

    # -----------------------------
    # Training
    # -----------------------------
    "train_steps": 200,

    # -----------------------------
    # Conv hybrid (optional ablation)
    # -----------------------------
    "use_conv": False,
    "architecture": "interleaved",
}

# =========================================================
# CONTEXT LENGTH EXPERIMENTS
# =========================================================

TRAIN_CTX = 512
CONTEXT_LENGTHS = [512, 1024, 2048]