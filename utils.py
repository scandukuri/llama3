import torch

def compute_frequencies(
    dimension: int,
    freq_constant: int
):
    if dimension % 2 != 0:
        print("Embedding dimension must be even for a clean RoPE setup. Sorry! 🦙")
        return None
    half_dim = dimension // 2
    power = -2 * torch.arange(0, half_dim) / dimension
    frequencies = freq_constant ** power
    return frequencies # should be the frequenceies as a torch.Tensor with shape (dimension // 2,)
