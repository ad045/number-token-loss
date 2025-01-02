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
        """
        Compute the Gaussian-smoothed cross-entropy loss.
        """
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
            # If no selector is given, assume all are number tokens!
            number_tokens = torch.ones_like(labels, dtype=torch.bool)

        # Mask for valid number labels and non-padding tokens
        valid_mask = (labels != self.ignore_index) & (number_tokens)
            # Example: 
            # tensor([[   1,    2, -100],           
            #         [   0, -100,    4]])
            # tensor([[ True,  True, False],
            #         [ True, False,  True]])
            
        # Replace ignore_index with a valid class index (e.g., 0) for one-hot encoding
        labels_non_neg = labels.clone()
        labels_non_neg[~valid_mask] = 0  # Set invalid labels to 0

        # One-hot encode the labels >> labels and logits then have the shape [batch_size, seq_len, num_number_classes]
        num_classes = logits.size(-1)
        one_hot_labels = F.one_hot(labels_non_neg, num_classes=num_classes).float()

        # Set one-hot vectors of invalid labels to zero
        one_hot_labels[~valid_mask] = 0.0
        
        # Gaussian smoothing. Computation is vectorized, which is why reshaping is done.
        # Gaussian distribution around each label index:
        #    Over [0..num_classes-1] for each label l_i: 
        #       dist_j = exp(-((j - l_i)^2 / (2*sigma^2)))
        
        # Flatten for vectorized computation >> shape [B*S, ...]
        labels_flat = labels[valid_mask].view(-1)  # only the valid positions
        if labels_flat.numel() == 0: # If there are no valid number tokens, return 0.0 as loss
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        classes_arange = torch.arange(num_classes, device=logits.device).unsqueeze(0)  # shape [1, num_classes]
        labels_flat_expanded = labels_flat.unsqueeze(1) # shape [V, 1], where V is number of valid tokens

        # Compute Gaussian
        print(classes_arange)
            # tensor([[0, 1, 2, 3, 4]]) 
        print(labels_flat_expanded)
            # tensor([[1],
                    # [2],
                    # [0],
                    # [4]])
        dist_sq = (classes_arange - labels_flat_expanded) ** 2
        print(dist_sq)
            # tensor([[ 1,  0,  1,  4,  9],
            #         [ 4,  1,  0,  1,  4],
            #         [ 0,  1,  4,  9, 16],
            #         [16,  9,  4,  1,  0]])
        gauss = torch.exp(-dist_sq / (2 * (self.sigma ** 2)))
        # Normalize
        print(gauss.sum(dim=-1))
            # tensor([2.3595, 2.4837, 1.7533, 1.7533])
        gauss = gauss / gauss.sum(dim=-1, keepdim=True)  # shape [V, num_classes]

        # Reshape >> [batch_size, seq_len, num_classes]
        gaussian_labels = torch.zeros_like(one_hot_labels)  
        gaussian_labels[valid_mask] = gauss # Computed distribution is assigned only to valid positions

        # Compute cross-entropy using smoothed label distribution
        log_probs = F.log_softmax(logits, dim=-1)  # shape [B, S, num_classes]
        loss_per_token = -(gaussian_labels * log_probs).sum(dim=-1) # distribution = - sum_{j} (smoothed_label_j * log_probs_j)

        # Average across the valid tokens
        loss_per_token = torch.where(valid_mask, loss_per_token, torch.tensor(0.0, device=loss_per_token.device))
        num_valid = valid_mask.sum().item()
        loss = loss_per_token.sum() / max(num_valid, 1)

        return loss