import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(256, 256)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        scores = q @ k.transpose(-2, -1) / (C ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.head_size = d_model // num_heads
        self.heads = nn.ModuleList([
            Head(d_model, self.head_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
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
    
class Block(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, num_heads)

        self.ffwd = FeedForward(d_model)

        self.ln1 = nn.LayerNorm(d_model)

        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):

        x = x + self.attn(self.ln1(x))
        
        x = x + self.ffwd(self.ln2(x))


        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(256, d_model)
        self.blocks = nn.Sequential(
            *[Block(d_model, num_heads) for _ in range(num_layers)]
        )
        
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        
        B, T = x.shape
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(T, device=x.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        return logits
        
model = GPT(
    vocab_size=1000,
    d_model=32,
    num_heads=4,
    num_layers=3
)

x = torch.randint(0, 1000, (2,128))

out = model(x)

print(out.shape)

num_params = sum(p.numel() for p in model.parameters())

print(num_params)