"""
Mini GPT Implementation in PyTorch
Description: 
    This module implements a GPT-style Transformer for character- or token-level language modeling.
    It includes:
        - Token and positional embeddings
        - Multi-head self-attention with causal masking
        - Transformer blocks with residual connections
        - Feed-forward layers with GELU activation
        - LayerNorm
        - Dropout
        - Optional loss computation in forward
    
Gebru
Oct 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------------------------------
# Custom LayerNorm (can also use nn.LayerNorm)
# -------------------------------
class LayerNorm(nn.Module):
    """
    Layer normalization over the last dimension of input tensor.
    """
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift

# -------------------------------
# GELU Activation
# -------------------------------
class GELU(nn.Module):
    """
    Gaussian Error Linear Unit activation function.
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

# -------------------------------
# Feed-forward network
# -------------------------------
class FeedForward(nn.Module):
    """
    Two-layer feed-forward network with GELU activation and dropout.
    """
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["dropout_rate"])
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------
# Multi-head self-attention
# -------------------------------
class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.
    """
    def __init__(self, cfg):
        super().__init__()
        d_in = cfg["emb_dim"]
        d_out = cfg["emb_dim"]
        self.num_heads = cfg["n_heads"]
        self.head_dim = d_out // self.num_heads

        assert d_out % self.num_heads == 0, "emb_dim must be divisible by n_heads"

        self.W_q = nn.Linear(d_in, d_out, bias=cfg["qkv_bias"])
        self.W_k = nn.Linear(d_in, d_out, bias=cfg["qkv_bias"])
        self.W_v = nn.Linear(d_in, d_out, bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(cfg["dropout_rate"])

        # causal mask for autoregressive attention
        self.register_buffer("mask", torch.triu(torch.ones(cfg["context"], cfg["context"]), diagonal=1))

    def forward(self, x):
        B, T, C = x.shape

        # linear projections
        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, T, T)

        # apply causal mask
        mask_bool = self.mask.bool()[:T, :T]
        attn_scores = attn_scores.masked_fill(mask_bool, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # attention output
        context = attn_weights @ V  # (B, heads, T, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, T, C)  # concat heads
        context = self.out_proj(context)
        return context

# -------------------------------
# Transformer block
# -------------------------------
class TransformerBlock(nn.Module):
    """
    A single transformer block with pre-LayerNorm, multi-head attention, feed-forward, residual connections.
    """
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])

    def forward(self, x):
        # attention + residual
        x = x + self.dropout(self.attn(self.norm1(x)))
        # feed-forward + residual
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

# -------------------------------
# GPT Model
# -------------------------------
class GPTModel(nn.Module):
    """
    Mini GPT language model.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["dropout_rate"])

        # stack transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.lm_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        idx: (batch_size, seq_len) input token indices
        targets: optional (batch_size, seq_len) target indices for loss computation
        returns: logits (and loss if targets provided)
        """
        B, T = idx.shape
        tok_embeds = self.tok_emb(idx)
        pos_embeds = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok_embeds + pos_embeds)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
            return logits, loss
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Autoregressive generation.
        idx: initial context (B, T)
        max_new_tokens: number of tokens to generate
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["context"]:]  # crop to context length
            logits = self(idx_cond)  # forward pass
            logits = logits[:, -1, :]  # last token
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
