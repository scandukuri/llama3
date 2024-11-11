from torch import nn
import torch.nn.functional as F
import torch
from typing import Optional
from utils import compute_frequencies


class ModelArgs:
    def __init__(
        self,
        embed_dim,
        num_kv_heads,
        num_q_heads,
        
    ) -> None:
        pass


class RMSNorm(nn.Module):
    def __init__(
        self, 
        dim: int, 
        epsilon: Optional[float] = 1e-8
    ) -> None:
        """
        RMS Normalization along the last dimension of a tensor
        """
        super.__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.w = nn.Parameter(data=torch.ones(size=self.dim))
        
        
    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        normalizing_constant = torch.sqrt(torch.mean(x ** 2, dim=-1)) + self.epsilon
        return self.w * x / normalizing_constant
    
        

class RoPE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        freq_constant: Optional[int] = 10000
    ) -> torch.Tensor:
        super.__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.freq_constant = freq_constant
        self.token_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        self.frequencies = compute_frequencies(dimension=self.embed_dim, freq_constant=self.freq_constant) # should be the frequenceies as a torch.Tensor with shape (self.embed_dim,)
    
    
    def forward(
        self,
        ids: torch.Tensor, 
    ) -> torch.Tensor:
        # Expects a tensor ids of shape (bsz, seq_len) where bsz can be 1. This means you may have to unsqueeze if you are only calling for a single sequence.
        tok_embeddings = self.token_embedding(ids)  # tok_embeddings is now (bsz, seq_len, self.embed_dim)
        batch_size, seq_len, embed_dim = tok_embeddings.size()
    
        positions = torch.arange(seq_len, device=ids.device).unsqueeze(0)  # (1, seq_len)
        theta = positions.unsqueeze(-1) * self.frequencies  # (1, seq_len, embed_dim)


        # REVIEW: I think this is the correct way to do it, but I am not 100% sure.
        rotated_embeddings = tok_embeddings * torch.cos(theta) + torch.cat(
            [tok_embeddings[..., 1:], tok_embeddings[..., :1]], dim=-1
        ) * torch.sin(theta)
        
        return rotated_embeddings
        

class FeedForward(nn.Module):
    pass


class TransformerBlock(nn.Module):
    pass


class GroupQueryAttention(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ) -> None:
        super().__init__()
        self.args = args
        self.n_kv_heads = args.kv_heads
        self.n_q_heads = args.q_heads
        self.q = nn.ModuleList(
            [
                nn.Linear(in_features=args.embed_dim, out_features=args.q_dim) 
                for _ in range(self.q_heads)
            ]
        )
        self.k = nn.ModuleList(
            [
                nn.Linear(in_features=args.embed_dim, out_features=args.k_dim)
                for _ in range(self.kv_heads)
            ]
        )
        self.v = nn.ModuleList(
            [
                nn.Linear(in_features=args.embed_dim, out_features=args.v_dim)
                for _ in range(self.kv_heads)
            ]
        )   


class Llama3(nn.Module):
    pass
