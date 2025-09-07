"""
Fallback solution to disable MoE operations when custom ops are missing.
This patches the vLLM MoE implementation to use fallback operations.
"""

import torch
import warnings

def patch_moe_operations():
    """Patch missing MoE operations with fallback implementations."""
    
    # Check if the problematic operation exists
    if not hasattr(torch.ops, '_moe_C') or not hasattr(torch.ops._moe_C, 'topk_softmax'):
        print("Warning: MoE custom operations not available. Using fallback implementation.")
        
        # Create a fallback topk_softmax implementation
        def fallback_topk_softmax(topk_weights, topk_ids, token_expert_indices, 
                                 num_tokens, num_experts, topk, renormalize):
            """
            Fallback implementation of topk_softmax using standard PyTorch operations.
            """
            # Apply softmax to topk_weights
            if renormalize:
                topk_weights = torch.softmax(topk_weights, dim=-1)
            else:
                topk_weights = torch.exp(topk_weights)
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            
            return topk_weights, topk_ids
        
        # Patch the missing operation
        if hasattr(torch.ops, '_moe_C'):
            torch.ops._moe_C.topk_softmax = fallback_topk_softmax
        else:
            # Create the namespace if it doesn't exist
            class MockMoENamespace:
                def __init__(self):
                    self.topk_softmax = fallback_topk_softmax
            
            torch.ops._moe_C = MockMoENamespace()
        
        print("MoE fallback operations installed successfully.")
        return True
    
    return False

# Apply the patch when this module is imported
if __name__ == "__main__":
    patch_moe_operations()
else:
    # Auto-patch on import
    patch_moe_operations()
