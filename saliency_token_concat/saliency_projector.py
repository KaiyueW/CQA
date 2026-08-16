import torch
import torch.nn as nn


class SaliencyProjector(nn.Module):
    """
    Linear (optionally 2-layer MLP) projector: 768 -> llm_hidden_size.
    """

    def __init__(
        self,
        saliency_dim: int = 768,
        llm_hidden_size: int = 4096,
        hidden_mlp: bool = False,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.use_layernorm = use_layernorm
        if use_layernorm:
            # scale all the saliency dims to have mean of 0 and variance of 1.
            self.pre_norm = nn.LayerNorm(saliency_dim)

        if hidden_mlp: #2-step pipeline
            self.proj = nn.Sequential(
                nn.Linear(saliency_dim, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
        else:
            self.proj = nn.Linear(saliency_dim, llm_hidden_size) #build a single linear layer 768-> 4096

        self._init_weights()

    def _init_weights(self):
        # Small init so saliency tokens start close to "no-op" perturbation
        # of the sequence and LoRA can learn how much to lean on them,
        # rather than the model being hit with a large out-of-distribution
        # embedding at step 0.
        for m in self.proj.modules() if isinstance(self.proj, nn.Sequential) else [self.proj]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_tokens: [B, 49, 768]
        Returns:
            [B, 49, llm_hidden_size], dtype matching the projector's params
        """
        x = latent_tokens
        if self.use_layernorm:
            x = self.pre_norm(x)
        return self.proj(x)