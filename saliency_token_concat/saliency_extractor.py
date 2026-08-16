import torch
import torch.nn as nn


class VisSalFormerLatentExtractor(nn.Module):
    """
    Frozen wrapper around a pretrained SalFormer that returns the
    [B, 49, 768] latent saliency tokens.
    """

    def __init__(self, vissalformer: nn.Module):
        super().__init__()
        self.vissalformer = vissalformer
        self.vissalformer.eval()
        for p in self.vissalformer.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        return super().train(False) # always stay in eval mode

    @torch.no_grad()
    def forward(self, img: torch.Tensor, q_inputs) -> torch.Tensor:
        """
        Args:
            img: [B, 3, H, W] pixel tensor, same as SalFormer.forward expects.
            q_inputs: tokenized BERT inputs dict (input_ids/attention_mask/...),
                      same as SalFormer.forward expects.
        Returns:
            latent_tokens: [B, 49, 768] float tensor, detached, no grad graph.
        """
        latent = self.vissalformer(img, q_inputs, return_latent_features=True)
        return latent.detach()