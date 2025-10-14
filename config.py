# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------
GPT_CONFIG = {
    "vocab_size": 16000,   # vocabulary size
    "context": 1024,       # context length (number of tokens)
    "emb_dim": 768,        # embedding dimension
    "n_heads": 12,         # number of attention heads
    "n_layers": 12,        # number of transformer layers
    "dropout_rate": 0.1,   # dropout probability
    "qkv_bias": False       # whether to include bias in QKV projections
}

BERT_CONFIG = {
    "vocab_size": 30522,   # vocabulary size (e.g., BERT-base)
    "max_len": 512,        # maximum sequence length
    "emb_dim": 768,        # embedding dimension
    "n_heads": 12,         # number of attention heads
    "n_layers": 12,        # number of transformer layers
    "dropout_rate": 0.1,   # dropout probability
    "intermediate_dim": 3072,  # feed-forward hidden layer size
    "qkv_bias": True       # whether to include bias in QKV projections
}
