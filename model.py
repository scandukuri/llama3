import torch
from torch import nn

import tiktoken
import blobfile
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_tensor_type(torch.BFloat16Tensor)


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
    freqs = 1 / theta.pow(torch.arange(0, dim, 2).float() / dim) # Original implementation cuts this off at [:dim // 2]. This should not be necessary.
    p = torch.arange(max_seq_length)
    pass




### Transformer Block ###
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()