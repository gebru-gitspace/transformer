# Transformer models (GPT & BERT) from scratch
  
**Date:** October 2025  

This repository implements **GPT-style and BERT-style Transformers from scratch** in PyTorch, designed to work with **Tigrinya text** and **HuggingFace datasets**. It supports training from **Hugging Face datasets** or **local `.txt` files**, with **manual and HF tokenizers**, **dynamic batching**, **masking for BERT**, **gradient accumulation**, **evaluation**, **model saving**, and **text generation**.  

---

## Table of Contents
1. [Overview](#overview)  
2. [Model Design](#model-design)  
3. [Tokenization](#tokenization)  
4. [Dataset Handling](#dataset-handling)  
5. [Training Procedure](#training-procedure)  
6. [Evaluation](#evaluation)  
7. [Saving & Loading Models](#saving--loading-models)  
8. [Text Generation](#text-generation)  
9. [Usage](#usage)  
10. [Requirements](#requirements)  

---

## Overview

✅ Supports **GPT** for auto-regressive generation.  
✅ Supports **BERT** for Masked Language Modeling (MLM).  
✅ Works with **local Tigrinya datasets** (`.txt` files).  
✅ Supports **Hugging Face datasets**.  
✅ Implements **dynamic batching** & **gradient accumulation**.  
✅ Computes **loss** and **perplexity** metrics.  
✅ Saves checkpoints in `.pth` format.  

---

## Model Design

### GPT Model
- Token and positional embeddings  
- Stack of Transformer blocks:
  - Multi-head self-attention with causal masking  
  - Feed-forward layers with GELU activation  
  - Residual connections + LayerNorm  
- Final linear layer for token prediction  

### BERT Model
- Token, positional, and segment embeddings (sentence A/B)  
- Stack of Transformer blocks:
  - Multi-head self-attention with MLM masking  
  - Feed-forward layers with GELU activation  
  - Residual connections + LayerNorm  
- MLM head for predicting masked tokens  

---

## Tokenization

Supports two modes:

1. **Manual Tigrinya tokenizer**  
   - Character- or word-level tokenization  
   - Maps tokens to IDs and back  

2. **Hugging Face tokenizer**  
   - `AutoTokenizer` or custom-trained tokenizer  
   - Handles tokenization, padding, truncation, and special tokens  

---

## Dataset Handling

- Accepts multiple `.txt` files in a folder  
- Implements **chunking/token batching** to avoid memory overload  
- Dynamically converts long text sequences into **fixed-length token sequences**  
- **BERT:** Random masking of tokens (MLM)  
- **GPT:** Auto-regressive sequences with causal masking  

---

## Training Procedure

- Supports **batch training** with **gradient accumulation**  
- Optional **learning rate scheduling**  
- Uses `torch.nn.CrossEntropyLoss` for token prediction  
- Training loop prints:
  - Epoch progress  
  - Running loss  
  - Perplexity every N steps  
- Saves checkpoints per epoch in `.pth` format  

```python
train_model(
    model=model,
    dataloader=train_loader,
    optimizer=optimizer,
    num_epochs=5,
    gradient_accumulation_steps=4,
    save_dir
)
