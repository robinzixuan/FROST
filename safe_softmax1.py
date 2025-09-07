import torch
import torch.nn as nn

def safe_softmax_1(input: torch.Tensor, dim=-1, eps=1e-8) -> torch.Tensor:
    """
    Numerically stable version of softmax_1:
    $\text(softmax)_1(x_i) = exp(x_i) / (1 + \sum_j exp(x_j))$
    
    Args:
        input: Input tensor
        dim: Dimension to apply softmax over
        eps: Small epsilon for numerical stability
    """
    # Clamp input to prevent overflow
    input_clamped = torch.clamp(input, min=-50, max=50)
    
    # Compute max for numerical stability
    input_max = input_clamped.max(dim=dim, keepdim=True).values
    
    # Shift inputs to prevent overflow
    shifted_inputs = input_clamped - input_max
    
    # Compute numerator
    numerator = torch.exp(shifted_inputs)
    
    # Compute denominator with numerical stability
    denominator = 1.0 + numerator.sum(dim=dim, keepdim=True) + eps
    
    # Compute result
    result = numerator / denominator
    
    # Check for NaN/Inf and replace with uniform distribution
    if torch.isnan(result).any() or torch.isinf(result).any():
        print("Warning: NaN/Inf detected in safe_softmax_1, using uniform distribution")
        uniform_size = list(result.shape)
        uniform_size[dim] = 1
        uniform_val = 1.0 / result.shape[dim]
        result = torch.full_like(result, uniform_val)
    
    return result

class SafeSoftmax1(nn.Module):
    """Safe version of Softmax1 with numerical stability"""
    
    def __init__(self, dim=-1, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, input):
        return safe_softmax_1(input, dim=self.dim, eps=self.eps)
    
    def extra_repr(self):
        return f"dim={self.dim}, eps={self.eps}"

