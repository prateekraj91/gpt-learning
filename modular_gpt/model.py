import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# RoPE helpers
# -----------------------------


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, sin, cos):
    return (x * cos) + (rotate_half(x) * sin)

# =========================================================
# HEAD
# =========================================================

class Head(nn.Module):

    def __init__(self, d_model, head_size, config):
        super().__init__()

        self.head_size = head_size
        self.window_size = config["window_size"]
        self.attention_type = config["attention_type"]
        self.pos_encoding = config["pos_encoding"]
        self.max_len = config["seq_len"]

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config["seq_len"], config["seq_len"]))
        )

        if self.pos_encoding == "relative":
            self.rel_bias = nn.Embedding(2 * config["seq_len"] - 1, 1)


    def phi(self, x):
        return F.elu(x) + 1

    def forward(self, q, k, v):

        B, T, C = q.shape
        head_dim = C

        # -----------------------------
        # RoPE
        # -----------------------------
        if self.pos_encoding == "rope":
            freqs = torch.arange(0, head_dim, 2, device=q.device).float()
            inv_freq = 1.0 / (10000 ** (freqs / head_dim))
            pos = torch.arange(T, device=q.device).float()

            angles = pos[:, None] * inv_freq[None, :]
            sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1).unsqueeze(0)
            cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1).unsqueeze(0)

            q = apply_rotary_pos_emb(q, sin, cos)
            k = apply_rotary_pos_emb(k, sin, cos)

        scores = q @ k.transpose(-2, -1) / (head_dim ** 0.5)

        # -----------------------------
        # ALiBi
        # -----------------------------
        if self.pos_encoding == "alibi":
            head_index = getattr(self, "head_index", 0)
            num_heads = getattr(self, "num_heads", 1)

            slope = 2 ** (-8 * head_index / num_heads)
            pos = torch.arange(T, device=q.device)

            dist = (pos[None, :] - pos[:, None]).abs().float()
            scores = scores - slope * dist

        if self.pos_encoding == "relative":
            positions = torch.arange(T, device=q.device)
            rel_pos   = positions.unsqueeze(0) - positions.unsqueeze(1) + (self.max_len - 1)
            rel_pos   = rel_pos.clamp(0, 2 * self.max_len - 2)
            scores    = scores + self.rel_bias(rel_pos).squeeze(-1)

                # Linear attention — early return, no mask needed
        if self.attention_type == "linear":
            q  = self.phi(q)
            k  = self.phi(k)
            kv = k.transpose(-2, -1) @ v
            k_sum = k.sum(dim=1, keepdim=True)
            z  = 1 / ((q * k_sum).sum(dim=-1, keepdim=True) + 1e-6)
            return (q @ kv) * z

        # Sliding window mask
        if self.attention_type == "sliding_window":
            mask   = torch.tril(torch.ones(T, T, device=q.device))
            mask   = torch.triu(mask, diagonal=-self.window_size + 1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Standard / MQA / ReLU — causal mask
        else:
            causal_mask = torch.tril(
                torch.ones(T, T, device=q.device)
                )
                
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        # ReLU softmax-free
        if self.attention_type == "relu_attention":
            scores  = torch.where(torch.isinf(scores), torch.zeros_like(scores), scores)
            weights = F.relu(scores)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Standard / MQA / sliding window
        else:
            weights = F.softmax(scores, dim=-1)

        return weights @ v



# =========================================================
# MULTI-HEAD ATTENTION
# =========================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, config):
        super().__init__()

        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        self.attention_type = config["attention_type"]

        self.query = nn.Linear(d_model, d_model, bias=False)

        # standard K/V
        if self.attention_type != "mqa":
            self.key = nn.Linear(d_model, d_model, bias=False)
            self.value = nn.Linear(d_model, d_model, bias=False)

        # MQA
        else:
            self.shared_key = nn.Linear(d_model, self.head_size, bias=False)
            self.shared_value = nn.Linear(d_model, self.head_size, bias=False)

        self.heads = nn.ModuleList([
            Head(d_model, self.head_size, config)
            for _ in range(num_heads)
        ])

        for i, h in enumerate(self.heads):
            h.head_index = i
            h.num_heads = num_heads

        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):

        B, T, C = x.shape

        q = self.query(x)
        q = q.view(B, T, self.num_heads, self.head_size)

        outputs = []

        # -----------------------------
        # Standard / Sliding / Linear
        # -----------------------------
        if self.attention_type != "mqa":

            k = self.key(x)
            v = self.value(x)

            k = k.view(B, T, self.num_heads, self.head_size)
            v = v.view(B, T, self.num_heads, self.head_size)

            for i, head in enumerate(self.heads):
                outputs.append(
                    head(q[:, :, i, :], k[:, :, i, :], v[:, :, i, :])
                )

        # -----------------------------
        # MQA
        # -----------------------------
        else:

            k = self.shared_key(x)
            v = self.shared_value(x)

            for i, head in enumerate(self.heads):
                outputs.append(
                    head(q[:, :, i, :], k, v)
                )

        out = torch.cat(outputs, dim=-1)
        return self.proj(out)

# =========================================================
# BLOCK
# =========================================================

class Block(nn.Module):

    def __init__(self, d_model, num_heads, config):
        super().__init__()

        self.use_conv = config["use_conv"]

        if self.use_conv:
            self.conv = nn.Conv1d(d_model, d_model, 3, padding=1)
            self.conv_norm = nn.LayerNorm(d_model)

        self.attn = MultiHeadAttention(d_model, num_heads, config)
        self.ffwd = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):

        if self.use_conv:
            y = self.conv(x.transpose(1, 2)).transpose(1, 2)
            x = self.conv_norm(x + y)

        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class InterleavedBlock(nn.Module):

    def __init__(self, d_model, num_heads, config, use_conv=True):
        super().__init__()

        self.use_conv = use_conv

        if use_conv:
            self.conv    = nn.Conv1d(d_model, d_model, 3, padding=1)
            self.ln_conv = nn.LayerNorm(d_model)
        else:
            self.attn    = MultiHeadAttention(d_model, num_heads, config)
            self.ln_attn = nn.LayerNorm(d_model)

        self.ffwd  = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln_ff = nn.LayerNorm(d_model)

    def forward(self, x):
        if self.use_conv:
            y = self.conv(x.transpose(1, 2)).transpose(1, 2)
            x = x + self.ln_conv(y)
        else:
            x = x + self.attn(self.ln_attn(x))
        x = x + self.ffwd(self.ln_ff(x))
        return x

        

# =========================================================
# GPT
# =========================================================

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.token_embedding = nn.Embedding(
            config["vocab_size"],
            config["d_model"]
        )

        self.pos_encoding = config["pos_encoding"]

        if self.pos_encoding == "learned":
            self.pos_embedding = nn.Embedding(
                config["seq_len"],
                config["d_model"]
            )
        
        if config.get("architecture") == "interleaved":
            self.blocks = nn.Sequential(*[
                InterleavedBlock(
                    config["d_model"],
                    config["num_heads"],
                    config,
                    use_conv=(i % 2 == 0)   # even layers = conv, odd = attention
                )
                for i in range(config["n_layers"])
            ])
        else:
            self.blocks = nn.Sequential(*[
                Block(config["d_model"], config["num_heads"], config)
                for _ in range(config["n_layers"])
            ])

        self.ln_final = nn.LayerNorm(config["d_model"])
        self.lm_head  = nn.Linear(config["d_model"], config["vocab_size"])


    def forward(self, x):

        B, T = x.shape
        x = self.token_embedding(x)

        if self.pos_encoding == "learned":
            T_clamped = min(T, self.pos_embedding.num_embeddings)
            pos       = self.pos_embedding(torch.arange(T_clamped, device=x.device))
            if T > T_clamped:
                pad = torch.zeros(T - T_clamped, x.shape[-1], device=x.device)
                pos = torch.cat([pos, pad], dim=0)
            x = x + pos

        x = self.blocks(x)
        x = self.ln_final(x)
        return self.lm_head(x)