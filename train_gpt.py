# -*- coding: utf-8 -*-
"""
Small Language Model (SLM) from Scratch using HuggingFace Tokenizer
Minimalized & formalized
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from tqdm.auto import tqdm
from contextlib import nullcontext

# === Step 2: Load Dataset ===
ds = load_dataset("roneneldan/TinyStories")

# === Step 3: Initialize HuggingFace Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure we have a pad token

# === Step 4: Tokenize Dataset ===
def tokenize(example):
    enc = tokenizer(example['text'], return_tensors='np', padding=False, truncation=False)
    ids = enc['input_ids'].squeeze()
    return {'ids': ids, 'len': len(ids)}

tokenized = ds.map(tokenize, remove_columns=['text'], num_proc=4)

# === Step 5: Save token IDs to disk for memmap ===
def save_bin(dset, filename):
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
    idx = 0
    for sample in tqdm(dset):
        arr[idx:idx+len(sample['ids'])] = sample['ids']
        idx += len(sample['ids'])
    arr.flush()

if not os.path.exists("train.bin"):
    save_bin(tokenized['train'], 'train.bin')
    save_bin(tokenized['validation'], 'validation.bin')

# === Step 6: Data Batching Function ===
block_size = 128
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu'
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=torch.float16)

def get_batch(split):
    fname = 'train.bin' if split == 'train' else 'validation.bin'
    data = np.memmap(fname, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# === Step 7: Minimal GPT Architecture ===
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head, self.n_embd = config['n_head'], config['n_embd']
        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'])
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])
        self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size'])).view(1, 1, config['block_size'], config['block_size']))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) * (1.0 / np.sqrt(C//self.n_head))
        att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1,2).contiguous().view(B,T,C)
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config['n_embd'], 4*config['n_embd'])
        self.proj = nn.Linear(4*config['n_embd'], config['n_embd'])
        self.dropout = nn.Dropout(config['dropout'])
    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config['n_embd'])
        self.ln2 = LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.wpe = nn.Embedding(config['block_size'], config['n_embd'])
        self.drop = nn.Dropout(config['dropout'])
        self.h = nn.ModuleList([Block(config) for _ in range(config['n_layer'])])
        self.ln_f = LayerNorm(config['n_embd'])
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx, targets=None):
        b,t = idx.size()
        pos = torch.arange(0,t,device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.h: x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] if idx.size(1) > block_size else idx
            logits, _ = self(idx_cond)
            logits = logits[:,-1,:]/temperature
            if top_k is not None:
                v,_ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits<v[:,-1]]= -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# === Step 8: Instantiate Model & Optimizer ===
config = {
    'vocab_size': tokenizer.vocab_size,
    'block_size': block_size,
    'n_layer': 6,
    'n_head': 6,
    'n_embd': 384,
    'dropout': 0.1
}
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# === Step 9: Minimal Training Loop ===
max_iters = 20
gradient_accumulation_steps = 16

for step in tqdm(range(max_iters)):
    X, y = get_batch('train')
    with ctx:
        logits, loss = model(X, y)
        loss = loss/gradient_accumulation_steps
        loss.backward()
    if (step+1) % gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step(); optimizer.zero_grad()

# === Step 10: Inference ===
context_sentence = "Once upon a time there was a pumpkin."
context_ids = torch.tensor(tokenizer.encode(context_sentence)).unsqueeze(0).to(device)
generated = model.generate(context_ids, max_new_tokens=200)
print(tokenizer.decode(generated.squeeze().tolist()))
