# import torch
# import torch.nn.functional as F
# from torch._tensor import Tensor

# from ntl.tokenizer.abstract_tokenizer import NumberEncodingTokenizer

# class NumberTokenSelector:
#     '''
#     Select number tokens 
#     '''
    
#     def __init__(self, tokenizer: NumberEncodingTokenizer, nvocab):
#         self.tokenizer = tokenizer
#         self.nvocab = nvocab
#         hashed_num_tokens = set(self.tokenizer.get_num_tokens())
#         # hashed_num_tokens = set(token for token in tokenizer.get_vocab().keys() if token.isdigit())

        
#         for token, id in self.tokenizer.get_vocab().items():
#             if token in hashed_num_tokens:
#                 self.nvocab[id] = self.tokenizer.decode_number_token(token, ignore_order=True)


#     def select_number_tokens(self, logits: Tensor, labels: Tensor):
        
#         # Create a mask to filter out non-digit tokens and labels
#         number_tokens = ~torch.isnan(self.nvocab)
#         logits = logits[:, :, number_tokens] 
#         labels = labels.masked_fill(labels == -100, 0)

#         return logits, labels, number_tokens


import torch
from torch import Tensor

class NumberTokenSelector:
    """
    Select number tokens.
    """

    def __init__(self, tokenizer, nvocab):
        self.tokenizer = tokenizer
        self.nvocab = nvocab

        # Identify number tokens in the vocabulary
        hashed_num_tokens = set(self.tokenizer.get_num_tokens())

        # Populate `nvocab` with decoded number tokens
        for token, id in self.tokenizer.get_vocab().items():
            if token in hashed_num_tokens:
                self.nvocab[id] = self.tokenizer.decode_number_token(token, ignore_order=True)

    def select_number_tokens(self, logits: Tensor, labels: Tensor):
        """
        Select only the number tokens from logits and labels.
        """

        # `self.nvocab` is 1D and represents the vocabulary size
        number_tokens = ~torch.isnan(self.nvocab)  # Shape: [vocab_size]

        # Move `number_tokens` to the same device as `logits`
        number_tokens = number_tokens.to(logits.device)

        # Expand `number_tokens` to match `logits` shape: [batch_size, seq_len, vocab_size]
        number_tokens = number_tokens.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, vocab_size]
        number_tokens = number_tokens.expand(logits.size(0), logits.size(1), -1)

        # Mask logits and labels to filter out non-number tokens
        logits = logits.masked_fill(~number_tokens, 0.0)  # Mask logits at non-number positions
        labels = labels.masked_fill(labels == -100, 0)  # Replace ignore_index with 0

        return logits, labels, number_tokens
