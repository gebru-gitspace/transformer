"""
BERT Implementation in PyTorch
Description:
    This module implements a BERT-style Transformer for masked language modeling.
    It includes:
        - Token and positional embeddings
        - Segment embeddings for sentence A/B distinction
        - Multi-head self-attention
        - Transformer blocks with residual connections
        - Feed-forward networks with GELU activation
        - LayerNorm and dropout
        - Masked language modeling (MLM) head

Gebru
Oct 2025

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------------------------------
# Layer Normalization
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
            nn.Linear(cfg["emb_dim"], cfg["intermediate_dim"]),
            GELU(),
            nn.Linear(cfg["intermediate_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["dropout_rate"])
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------
# Multi-head self-attention
# -------------------------------
class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention layer.
    """
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg["n_heads"]
        self.head_dim = cfg["emb_dim"] // cfg["n_heads"]
        assert cfg["emb_dim"] % cfg["n_heads"] == 0, "emb_dim must be divisible by n_heads"

        self.W_q = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.W_k = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.W_v = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])

    def forward(self, x, mask=None):
        """
        x: input tensor (batch_size, seq_len, emb_dim)
        mask: attention mask (batch_size, seq_len, seq_len)
        """
        B, T, C = x.shape

        # Linear projections
        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, heads, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, :, :] == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ V  # (B, heads, T, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        context = self.out_proj(context)
        return context

# -------------------------------
# Transformer block
# -------------------------------
class TransformerBlock(nn.Module):
    """
    Transformer block with pre-LayerNorm, multi-head attention, feed-forward, and residual connections.
    """
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

# -------------------------------
# BERT model
# -------------------------------
class BERTModel(nn.Module):
    """
    BERT model for masked language modeling (MLM).
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["max_len"], cfg["emb_dim"])
        self.seg_emb = nn.Embedding(2, cfg["emb_dim"])  # segment embeddings for sentence A/B
        self.dropout = nn.Dropout(cfg["dropout_rate"])
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.norm = LayerNorm(cfg["emb_dim"])
        self.mlm_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Forward pass.
        input_ids: (batch_size, seq_len)
        token_type_ids: (batch_size, seq_len) optional, default 0
        attention_mask: (batch_size, seq_len) optional, 1=keep, 0=mask
        labels: (batch_size, seq_len) optional, for MLM loss
        """
        B, T = input_ids.shape
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        tok_emb = self.tok_emb(input_ids)
        pos_emb = self.pos_emb(torch.arange(T, device=input_ids.device))
        seg_emb = self.seg_emb(token_type_ids)
        x = tok_emb + pos_emb + seg_emb
        x = self.dropout(x)

        x = self.blocks(x, mask=attention_mask)
        x = self.norm(x)
        logits = self.mlm_head(x)

        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return logits, loss
        return logits
