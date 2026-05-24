import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Attention Head
# -----------------------------

class Head(nn.Module):

    def __init__(self, d_model, head_size, config):

        super().__init__()

        self.window_size = config["window_size"]

        self.attention_type = config["attention_type"]

        self.key = nn.Linear(
            d_model,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            d_model,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            d_model,
            head_size,
            bias=False
        )

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    config["seq_len"],
                    config["seq_len"]
                )
            )
        )

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)

        q = self.query(x)

        v = self.value(x)

        scores = q @ k.transpose(-2, -1)

        scores = scores / (C ** 0.5)

        # -----------------------------
        # Standard Attention
        # -----------------------------

        if self.attention_type == "standard":

            scores = scores.masked_fill(
                self.mask[:T, :T] == 0,
                float("-inf")
            )

        # -----------------------------
        # Sliding Window Attention
        # -----------------------------

        elif self.attention_type == "sliding_window":

            local_mask = torch.tril(
                torch.ones(
                    T,
                    T,
                    device=x.device
                )
            )

            local_mask = torch.triu(
                local_mask,
                diagonal=-self.window_size + 1
            )

            scores = scores.masked_fill(
                local_mask == 0,
                float("-inf")
            )

        weights = F.softmax(
            scores,
            dim=-1
        )

        out = weights @ v

        return out


# -----------------------------
# Multi Head Attention
# -----------------------------

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, config):

        super().__init__()

        head_size = d_model // num_heads

        self.heads = nn.ModuleList([
            Head(
                d_model,
                head_size,
                config
            )
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(
            d_model,
            d_model
        )

    def forward(self, x):

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

            nn.Linear(
                d_model,
                4 * d_model
            ),

            nn.ReLU(),

            nn.Linear(
                4 * d_model,
                d_model
            )
        )

    def forward(self, x):

        return self.net(x)


# -----------------------------
# Transformer Block
# -----------------------------

class Block(nn.Module):

    def __init__(self, d_model, num_heads, config):

        super().__init__()

        self.attn = MultiHeadAttention(
            d_model,
            num_heads,
            config
        )

        self.ffwd = FeedForward(
            d_model
        )

        self.ln1 = nn.LayerNorm(
            d_model
        )

        self.ln2 = nn.LayerNorm(
            d_model
        )

    def forward(self, x):

        x = x + self.attn(
            self.ln1(x)
        )

        x = x + self.ffwd(
            self.ln2(x)
        )

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

        self.pos_encoding_type = config["pos_encoding"]

        # -----------------------------
        # Positional Encoding
        # -----------------------------

        if self.pos_encoding_type == "learned":

            self.pos_embedding = nn.Embedding(
                config["seq_len"],
                config["d_model"]
            )

        # -----------------------------
        # Transformer Blocks
        # -----------------------------

        self.blocks = nn.Sequential(

            *[
                Block(
                    config["d_model"],
                    config["num_heads"],
                    config
                )

                for _ in range(
                    config["n_layers"]
                )
            ]
        )

        # -----------------------------
        # LM Head
        # -----------------------------

        self.lm_head = nn.Linear(
            config["d_model"],
            config["vocab_size"]
        )

    def forward(self, x):

        B, T = x.shape

        token_emb = self.token_embedding(x)

        if self.pos_encoding_type == "learned":

            pos_emb = self.pos_embedding(
                torch.arange(
                    T,
                    device=x.device
                )
            )

            x = token_emb + pos_emb

        else:

            x = token_emb

        x = self.blocks(x)

        logits = self.lm_head(x)

        return logits