import torch
from torch import nn
from torch.nn import functional as F

import tiktoken
import blobfile
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_tensor_type(torch.BFloat16Tensor)


### CONSTANTS ###
### PULLED FROM LLAMA3 TECHNICAL REPORT ###
MODEL_DIM = 4096
FEED_FORWARD_DIMENSION = 14336
N_TRANSFORMER_BLOCKS = 32
N_ATTENTION_HEADS = 32
RMS_NORM_EPS = 1e-05
ROPE_EMBEDDING_THETA = 500000
HEAD_DIM = MODEL_DIM // N_ATTENTION_HEADS
VOCABULARY_SIZE = 128256

### LIMITS FOR WORKING ON CPU ###
MAX_BATCH_SIZE = 4
MAX_SEQ_LEN = 128

### FOR GQA ###
N_KV_HEADS = 8
N_KV_HEAD_REPEAT = N_ATTENTION_HEADS // N_KV_HEADS


### RMS Norm ###

class RMSNorm(nn.Module):
    def __init__(self, dim, norm_eps):
        super().__init__()
        self.dim = dim
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, a):
        return a / torch.rsqrt(torch.mean(a ** 2, dim=-1, keepdim=True) + self.norm_eps)
    
    def forward(self, a):
        out = self._norm(a.float()).type_as(a)
        return out * self.weight.view(1, -1)


### RoPE Embedding Utilities ###

def precompute_freq_per_pd(dim, max_seq_length, theta = 10000):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    p = torch.arange(max_seq_length, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(p, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    n_dim = x.ndim
    assert n_dim > 1
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    
    # we need to take x's shape for broadcasting at the sequence length dim and embedding dim (which we've already checked must match with those of freqs_cis)
    # for every other dim, we will want to add a singleton such that when we use out_shape to broadcast freqs_cis, there are singletons at every batch dim
    out_shape = [d if i == 1 or i == n_dim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*out_shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    # Embedding matrix for a single sequence x has shape (max_seq_len, embed_dim)
    # Correspondingly, embedding matrix for a batch of sequences has shape (bsz, max_seq_len, embed_dim)
    # W_q should have shape (d_q, embed_dim) to project a single sequence embedding matrix of shape (max_seq_len, embed_dim) - transposed to (embed_dim, max_seq_len) - to be (d_q, max_seq_len), transposed back to (max_seq_len, d_q)
    # Similarly, W_k should project a single sequence to (d_k, max_seq_len) where d_k == d_q, and transposed back to (max_seq_len, d_k).
    # If we have applied W_q and W_k accordingly, but to a batch of size bsz, xq given to us as an input should be (bsz, max_seq_len, d_q) and xk should be (bsz, max_seq_len, d_k) 
    # freqs_cis should be shape (max_seq_len, rotary_dim)
    xq_complex, xk_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)), torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


### FFN Block ###

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.V = nn.Linear(MODEL_DIM, FEED_FORWARD_DIMENSION, bias=False)
        self.W = nn.Linear(MODEL_DIM, FEED_FORWARD_DIMENSION, bias=False)
        self.DIM_REDUCER = nn.Linear(FEED_FORWARD_DIMENSION, MODEL_DIM, bias=False)
        
    def forward(self, x):
        return self.DIM_REDUCER(F.silu(self.W(x)) * self.V(x))
        
        
### Grouped Query Attention ###
class GroupedQueryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(MODEL_DIM, N_ATTENTION_HEADS * HEAD_DIM)
        self.wk = nn.Linear(MODEL_DIM, N_KV_HEADS * HEAD_DIM)
        self.wv = nn.Linear(MODEL_DIM, N_KV_HEADS * HEAD_DIM)
        self.wo = nn.Linear(N_ATTENTION_HEADS * HEAD_DIM, MODEL_DIM)
        
        self.cache_k = torch.zeros((MAX_BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, HEAD_DIM))
        self.cache_v = torch.zeros((MAX_BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, HEAD_DIM))
        
    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seq_len, _ = x.shape
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x) # q = (bsz, seq_len, N_ATTENTION_HEADS * HEAD_DIM), k = (bsz, seq_len, N_KV_HEADS * HEAD_DIM), v = (bsz, seq_len, N_KV_HEADS * HEAD_DIM)
        queries, keys, values = queries.view(bsz, seq_len, N_ATTENTION_HEADS, HEAD_DIM), keys.view(bsz, seq_len, N_KV_HEADS, HEAD_DIM), values.view(bsz, seq_len, N_KV_HEADS, HEAD_DIM) # split the projection's last dimension into the two dimensions we refer to when talking about "groups", "KV heads", "query heads"
        
        self.cache_k = self.cache_k.to(queries.device)
        self.cache_v = self.cache_v.to(queries.device)
        
        # Apply RoPE to queries and keys
        queries, keys = apply_rotary_emb(queries, keys, freqs_cis)
        
        # updated KV cache
        self.cache_k[:bsz, start_pos : start_pos + seq_len, :, :] = keys
        self.cache_k[:bsz, start_pos : start_pos + seq_len, :, :] = values
        keys, values = self.cache_k[:bsz, : start_pos + seq_len, :, :], self.cache_v[:bsz, : start_pos + seq_len, :, :]
        
        # adjust raw / split shapes of KV projections to make sure each query isn't lonely / has its own key and value head for scaled dot product attention computation
        keys = torch.repeat_interleave(
            keys, dim=2, repeats=N_KV_HEAD_REPEAT
        )
        values = torch.repeat_interleave(
            values, dim=2, repeats=N_KV_HEAD_REPEAT
        )
        
        queries, keys, values = queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=mask
        )
        
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1) 
        return self.wo(out)
    

### Transformer Block ###

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = GroupedQueryAttention()
        self.feed_forward = FeedForward()
        self.initial_norm = RMSNorm(MODEL_DIM, RMS_NORM_EPS)
        self.final_norm = RMSNorm(MODEL_DIM, RMS_NORM_EPS)
    
    
    def forward(self, x, start_pos, freqs_cis, mask):
        h = x + self.attention(self.initial_norm(x), start_pos, freqs_cis, mask)
        return h + self.feed_forward(self.final_norm(h))      
    
    
### Llama3 Transformer ###

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings = nn.Embedding(
            VOCABULARY_SIZE,
            MODEL_DIM
        )
        
        self.layers = nn.ModuleList()
        for _ in range(N_TRANSFORMER_BLOCKS):
            self.layers.append(TransformerBlock())
        
        self.pre_linear_norm = RMSNorm(MODEL_DIM, RMS_NORM_EPS)
        self.linear_proj = nn.Linear(MODEL_DIM, VOCABULARY_SIZE, bias=False)
        self.freqs_cis = precompute_freq_per_pd(
            HEAD_DIM,
            MAX_SEQ_LEN * 2,
            ROPE_EMBEDDING_THETA
        )
        
    
    def forward(self, tokens, start_pos):
        _bsz, seq_len = tokens.shape
        x = self.token_embeddings(tokens)
        freqs_cis = self.freqs_cis.to(tokens.device)
        
        
        mask = None
        if seq_len > 1: 
            mask = torch.full((seq_len, seq_len), float('-inf'), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).to(tokens.device)
        
        for layer in self.layers:
            x = layer(x, start_pos, freqs_cis, mask)
        
        logits = self.linear_proj(self.pre_linear_norm(x)).float()