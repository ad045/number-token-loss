import torch
import torch.nn.functional as F
from torch._tensor import Tensor

from ntl.tokenizer.abstract_tokenizer import NumberEncodingTokenizer
from ntl.utils.number_token_selector import NumberTokenSelector


# class NumberTokenLoss:
#     def __init__(self, tokenizer: NumberEncodingTokenizer, vocab_size: int, device, loss_function=F.mse_loss, weight=0.5):
#         self.loss_function = loss_function
#         self.weight = weight
        
#         # create a tensor of shape (vocab_size,) with the number tokens replaced by their corresponding number
#         self.nvocab = torch.full((vocab_size,), float("nan"), device=device)

#         self.selector = NumberTokenSelector(tokenizer, self.nvocab)



#     def forward(self, logits: Tensor, labels: Tensor):
#         if logits.numel() == 0:
#             raise ValueError("Logits passed to the NumberTokenLoss are empty!")
#         if labels.numel() == 0:
#             raise ValueError("Labels passed to the NumberTokenLoss are empty!")

#         logits, labels, number_tokens = self.selector.select_number_tokens(logits, labels)

#         # Compute the weighted average of number tokens (yhat)
#         softmaxed = F.softmax(logits, dim=-1)
#         yhat = torch.sum(softmaxed * self.nvocab[number_tokens], dim=-1)
#         y = self.nvocab[labels]

#         loss = self.loss_function(yhat[~torch.isnan(y)], y[~torch.isnan(y)])
#         return loss

class NumberTokenLoss:
    def __init__(self, tokenizer: NumberEncodingTokenizer, vocab_size: int, device, loss_function=F.mse_loss, weight=0.5):
        self.loss_function = loss_function
        self.weight = weight

        # Create a tensor of shape (vocab_size,) with number tokens replaced by their corresponding number
        self.nvocab = torch.full((vocab_size,), float("nan"), device=device)

        self.selector = NumberTokenSelector(tokenizer, self.nvocab)

    def forward(self, logits: Tensor, labels: Tensor):
        if logits.numel() == 0:
            raise ValueError("Logits passed to the NumberTokenLoss are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NumberTokenLoss are empty!")

        # Select only the number tokens
        logits, labels, number_tokens = self.selector.select_number_tokens(logits, labels)

        # Mask `self.nvocab` with the vocabulary indices to align with logits
        valid_nvocab = self.nvocab[~torch.isnan(self.nvocab)]  # Shape: [num_number_tokens]

        # Filter logits to include only valid number tokens
        softmaxed = F.softmax(logits, dim=-1)  # Shape: [batch_size, seq_len, vocab_size]
        logits_filtered = softmaxed[..., ~torch.isnan(self.nvocab)]  # Shape: [batch_size, seq_len, num_number_tokens]

        # Compute the weighted average of logits
        yhat = torch.sum(logits_filtered * valid_nvocab.unsqueeze(0).unsqueeze(0), dim=-1)

        # Map labels to their corresponding numeric values
        y = self.nvocab[labels]

        # Compute loss only for valid labels
        valid_indices = ~torch.isnan(y)
        loss = self.loss_function(yhat[valid_indices], y[valid_indices])

        return loss

