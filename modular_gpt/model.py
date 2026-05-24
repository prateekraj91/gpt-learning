import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class Head(nn.Module):

    def __init__(self, d_model, head_size):

        super().__init__()

        self.key = nn.Linear(d_model, head_size, bias=False)

        self.query = nn.Linear(d_model, head_size, bias=False)

        self.value = nn.Linear(d_model, head_size, bias=False)

        self.register_buffer(
            'mask',
            torch.tril(torch.ones(1024, 1024))
        )

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)

        q = self.query(x)

        v = self.value(x)

        scores = q @ k.transpose(-2, -1) / (C ** 0.5)

        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float('-inf')
        )

        weights = F.softmax(scores, dim=-1)

        out = weights @ v

        return out

# -----------------------------
# Multi Head Attention
# -----------------------------

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):

        super().__init__()

        self.attention_type = config["attention_type"]

        head_size = d_model // num_heads

        self.heads = nn.ModuleList([
            Head(d_model, head_size)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):

        if self.attention_type == "sliding_window":
            out = torch.cat(
                [h(x) for h in self.heads],
                dim=-1
            )

        out = self.proj(out)

        return out

# -----------------------------
# FeedForward
# -----------------------------

class FeedForward(nn.Module):

    def __init__(self, d_model):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):

        return self.net(x)

# -----------------------------
# Transformer Block
# -----------------------------

class Block(nn.Module):

    def __init__(self, d_model, num_heads):

        super().__init__()

        self.attn = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.ffwd = FeedForward(d_model)

        self.ln1 = nn.LayerNorm(d_model)

        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):

        x = x + self.attn(self.ln1(x))

        x = x + self.ffwd(self.ln2(x))

        return x

# -----------------------------
# GPT Model
# -----------------------------

class GPT(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.token_embedding = nn.Embedding(
            config["vocab_size"],
            config["d_model"]
        )

        self.pos_embedding = nn.Embedding(
            config["seq_len"],
            config["d_model"]
        )

        self.blocks = nn.Sequential(
            *[
                Block(config["d_model"], config["num_heads"])
                for _ in range(config["n_layers"])
            ]
        )

        self.lm_head = nn.Linear(
            config["d_model"],
            config["vocab_size"]
        )

    def forward(self, x):

        B, T = x.shape

        token_emb = self.token_embedding(x)

        pos_emb = self.pos_embedding(
            torch.arange(T, device=x.device)
        )

        x = token_emb + pos_emb

        x = self.blocks(x)

        logits = self.lm_head(x)

        return logits
