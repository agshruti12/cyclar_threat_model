import torch

def classify_danger(
    object_feature_sequences: torch.Tensor,
) -> torch.Tensor:
    """
    object_feature_sequences: (batch_size, seq_len, feature_dim)
      - each sequence corresponds to a single tracked object

    Returns:
      logits: (batch_size, 3) for [mild, dangerous, very_dangerous]
    """
    raise NotImplementedError("To be implemented later.")
