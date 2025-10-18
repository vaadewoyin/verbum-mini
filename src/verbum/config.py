""""
Config for model
"""
from dataclasses import dataclass

@dataclass
class VerbumConfig:
    vocab_size = 10000       # custom KJV tokenizer vocab
    context_len = 256            # max sequence (context) length
    n_layers = 4             # transformer blocks
    n_heads = 4              # attention heads
    emb_dim = 256              # feedforward hidden dim
    dropout  = 0.1           # dropout rate
