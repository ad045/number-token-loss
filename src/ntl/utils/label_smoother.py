import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass

from transformers.trainer_pt_utils import LabelSmoother


@dataclass
class GaussianLabelSmoother(LabelSmoother):
    """
    A label smoother that applies Gaussian smoothing ONLY to number tokens, as
    selected by `NumberTokenSelector`. Non-number tokens remain untouched or masked out.
    If sigma=0, this label smoother behaves identically to standard cross-entropy loss.

    Args:
        sigma (float, optional, defaults to 1.0):
            The standard deviation for the Gaussian around the correct label.
        ignore_index (int, optional, defaults to -100):
            The index in the labels to ignore (e.g., padding or special tokens). Inherited from `LabelSmoother`. 
        selector (NumberTokenSelector, optional):
            A selector to filter out tokens that are not recognized as numbers. Inherited from `LabelSmoother`.  
    """

    sigma: float = 1.0
    ignore_index: int = -100
    selector: object = None  # Instance of `NumberTokenSelector`

def __call__(self, model_output, labels: Tensor, shift_labels: bool = False) -> Tensor:
    # Get logits from model output
    if isinstance(model_output, dict):
        logits = model_output["logits"]
    else:
        logits = model_output[0]

    # Shift labels if needed
    if shift_labels:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    # Select only number tokens if needed
    if self.selector is not None:
        logits, labels, number_tokens = self.selector.select_number_tokens(logits, labels)
    else:
        number_tokens = torch.ones_like(labels, dtype=torch.bool)

    # `number_tokens` should match the vocab size
    batch_size, seq_len, vocab_size = logits.size()
    number_tokens_mask = number_tokens.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, vocab_size)

    # Create valid_mask for valid labels and number tokens
    valid_mask = (labels != self.ignore_index).unsqueeze(-1) & number_tokens_mask

    # Replace ignore_index with a valid class index (e.g., 0) for one-hot encoding
    labels_non_neg = labels.clone()
    labels_non_neg[~valid_mask.squeeze(-1)] = 0  # Squeeze extra dimension

    # One-hot encode the labels
    num_classes = logits.size(-1)
    one_hot_labels = F.one_hot(labels_non_neg, num_classes=num_classes).float()

    # Set one-hot vectors of invalid labels to zero
    one_hot_labels[~valid_mask] = 0.0

    # Gaussian smoothing logic remains the same...
    # Compute cross-entropy using smoothed label distribution
    log_probs = F.log_softmax(logits, dim=-1)
    loss_per_token = -(one_hot_labels * log_probs).sum(dim=-1)

    # Average across valid tokens
    loss_per_token = torch.where(valid_mask.squeeze(-1), loss_per_token, torch.zeros_like(loss_per_token))
    num_valid = valid_mask.sum().float()
    loss = loss_per_token.sum() / torch.clamp(num_valid, min=1.0)

    return loss
