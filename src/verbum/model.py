"""
Contains code for MultiHead Attention, Transformer block and Model architecture
"""

# Imports
import torch
import torch.nn as nn


# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    """Implements multi-head self-attention with causal masking."""

    def __init__(self, dim_in, dim_out, context_len, dropout, num_heads):
        super().__init__()
        if dim_out % num_heads != 0:
            raise ValueError(f"dim_out ({dim_out}) is not divisble by num_heads ({num_heads})")
        else:
          self.dim_in = dim_in
          self.dim_out = dim_out
          self.context_len = context_len
          self.num_heads = num_heads
          self.head_dim = dim_out//num_heads
          self.Wq = nn.Linear(dim_in, dim_out, bias = False)
          self.Wk = nn.Linear(dim_in, dim_out, bias = False)
          self.Wv = nn.Linear(dim_in, dim_out, bias = False)
          self.out_projection = nn.Linear(dim_out, dim_out)
          self.register_buffer('mask', torch.triu(torch.ones((context_len, context_len)), diagonal=1))
          self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_tokens, dim_in = x.shape
        queries = self.Wq(x)
        keys = self.Wk(x)
        values = self.Wv(x)

        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim) # shape: [batch_size,  num_tokens, num_heads,  head_dim]
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(1,2) # shape: [batch_size, num_heads, num_tokens, head_dim]
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attention_weights = torch.softmax(attention_scores/ (keys.shape[-1])**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1,2)
        context_vector= context_vector.contiguous().view(batch_size, num_tokens, self.dim_out)
        context_vector= self.out_projection(context_vector)
        return context_vector

# Transformer block
class TransformerBlock(nn.Module):
    """Single Transformer block with self-attention and feed-forward layers."""
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            dim_in = config.emb_dim,
            dim_out = config.emb_dim,
            context_len = config.context_len,
            dropout = config.dropout,
            num_heads = config.n_heads)
        self.layer_norm1 = nn.LayerNorm(config.emb_dim)
        self.layer_norm2 = nn.LayerNorm(config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.ffn = nn.Sequential(
            nn.Linear(config.emb_dim, (4 * config.emb_dim)),
            nn.GELU(),
            nn.Linear((4 * config.emb_dim), config.emb_dim))

    def forward(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut
        return x

# Verbum-mini Architecture (based on gpt-2)
class Verbum_mini(nn.Module):
    """GPT-style Transformer language model (Verbum Mini)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embeds = nn.Embedding(config.vocab_size, config.emb_dim)
        self.position_embeds = nn.Embedding(config.context_len, config.emb_dim)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_layer_norm = nn.LayerNorm(config.emb_dim)
        self.final_output = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, x):
        token_embeddings = self.token_embeds(x)
        position_embeddings = self.position_embeds(torch.arange(x.size(1), device=x.device))
        input_embeddings =  token_embeddings +  position_embeddings
        transformer_output = self.transformer_blocks(input_embeddings)
        output = self.final_layer_norm(transformer_output)
        logits = self.final_output(output)
        return logits