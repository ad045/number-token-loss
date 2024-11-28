import torch
import torch.nn.functional as F
from torch._tensor import Tensor

from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer


class GCENumberTokenLoss:
    def __init__(self, tokenizer: NumberEncodingTokenizer, vocab_size: int, device, loss_function=F.mse_loss, weight=0.5, sigma=0.5):  
        self.tokenizer = tokenizer
        self.loss_function = loss_function
        self.weight = weight
        
        self.sigma = sigma
        hashed_num_tokens = set(self.tokenizer.get_num_tokens())

        # create a tensor of shape (vocab_size,) with the number tokens replaced by their corresponding number
        self.nvocab = torch.full((vocab_size,), float("nan"), device=device)

        for token, id in self.tokenizer.get_vocab().items():
            if token in hashed_num_tokens:
                self.nvocab[id] = self.tokenizer.decode_number_token(token, ignore_order=True)



    def calculate_batched_gce(self, X, Y):
        """
        Compute Gaussian Cross-Entropy (GCE) loss for batched distributions.
        Args:
            X (Tensor): Predicted distributions (B x N).
            Y (Tensor): Ground truth distributions (B x N).
            sigma (float): Standard deviation for the Gaussian.
        Returns:
            Tensor: Loss value (scalar).
        """
        # Check dimensions
        if X.shape != Y.shape:
            raise ValueError("Expecting equal shapes for X and Y!")
        
        if X.ndim > 2 or X.ndim == 0: 
            raise ValueError("Expecting 2D inputs for X and Y (B x N)!")

        if X.ndim == 1: # i.e., at least one Tensor has only 1 dimension
            X = X.unsqueeze(0) 
            Y = Y.unsqueeze(0) 
            
        # Normalize to ensure they sum to 1
        X = X / X.sum(dim=1, keepdim=True)
        Y = Y / Y.sum(dim=1, keepdim=True)
        # X = X / X.sum(dim=0, keepdim=True) >> solution, if we look at non-batched things solely. 
        # Y = Y / Y.sum(dim=0, keepdim=True)
        
        # Calculate the GCE Loss
        maximum_loss = 1 / (torch.sqrt(torch.tensor(2 * torch.pi)) * self.sigma)
        gce_loss = maximum_loss - maximum_loss * torch.exp(-((X - Y) ** 2) / (2 * self.sigma ** 2))
        print(gce_loss)
        return gce_loss.mean()  # Average over the batch



    def forward(self, logits: Tensor, labels: Tensor):
        if logits.numel() == 0:
            raise ValueError("Logits passed to the NumberTokenLoss are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NumberTokenLoss are empty!")

        labels = labels.masked_fill(labels == -100, 0)

        # Create a mask to filter out non-digit tokens
        number_tokens = ~torch.isnan(self.nvocab)
        logits = logits[:, :, number_tokens]

        # Compute the weighted average of number tokens (yhat)
        softmaxed = F.softmax(logits, dim=-1)
        yhat = torch.sum(softmaxed * self.nvocab[number_tokens], dim=-1)
        y = self.nvocab[labels]

        gce_loss = self.calculate_batched_gce(yhat[~torch.isnan(y)], y[~torch.isnan(y)])
        
        return gce_loss
