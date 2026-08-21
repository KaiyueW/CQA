"""
Subclasses Qwen3VLForConditionalGeneration to inject the projected saliency
tokens into inputs_embeds at forward time, right at the placeholder
positions the collator already reserved.
"""

from typing import List, Optional, Dict

import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration

from saliency_projector import SaliencyProjector


class Qwen3VLWithSaliencyBottleneck(Qwen3VLForConditionalGeneration):
    """
    Drop-in replacement for Qwen3VLForConditionalGeneration that also carries
    a frozen VisSalFormer + trainable projector, and knows how to splice
    projected saliency tokens into inputs_embeds.

    """

    def attach_saliency_modules(
        self,
        saliency_token_id: int,
        llm_hidden_size: int,
        saliency_dim: int = 768,
        use_layernorm: bool = True,
    ):
        self.saliency_projector = SaliencyProjector(
            saliency_dim=saliency_dim,
            llm_hidden_size=llm_hidden_size,
            use_layernorm=use_layernorm,
        )
        self.saliency_token_id = saliency_token_id

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None, # 0 = text/saliency, 1 = image, 2 = video
        labels: Optional[torch.LongTensor] = None,
        saliency_latents: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):

        assert "pixel_values" not in kwargs, (
            "pixel_values must not be passed via **kwargs -- use the explicit "
            "pixel_values= argument so this method can control vision scatter."
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        # get real vectors corresponding to the input ids, already get text embeddings in this step, no need to scatter.

            # Prompt:       "Describe this picture: <IMAGE_PLACEHOLDER>"
            #                     │
            #                     ▼
            # inputs_embeds: [ "Describe", "this", "picture:",  [EMPTY_TOKEN] ]
            # image_mask:    [   False,     False,   False,         True      ]
            # image_embeds:  [             <Vision Feature Vectors>           ]
            #                     │
            #                     ▼ (masked_scatter)
            # Merged Embeds: [ "Describe", "this", "picture:", <Vision Vectors>]

            # --- 2. scatter （enter) the SALIENCY tokens (new) ---
            if saliency_latents is not None and hasattr(self, "saliency_projector"):
                latent_tokens = saliency_latents
                target_dtype = next(self.saliency_projector.parameters()).dtype
                latent_tokens = latent_tokens.to(device=inputs_embeds.device, dtype=target_dtype)
                # PyTorch requires tensor inputs and layer weights to have the exact same dtype.
                saliency_embeds = self.saliency_projector(latent_tokens)  # [B,49,H]

                sal_mask = (input_ids == self.saliency_token_id).unsqueeze(-1)
                sal_mask = sal_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

                inputs_embeds = inputs_embeds.masked_scatter(
                    sal_mask, saliency_embeds.reshape(-1, saliency_embeds.size(-1)).to(inputs_embeds.dtype)
                ) # reshape to [B*49, H] so that masked_scatter can work with the flattened mask.


        # --- 3. hand off to the normal Qwen3-VL forward, letting it compute
        #         position_ids (get_rope_index of (Time, Height, Width)) and run the LLM stack as usual ---
        
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            labels=labels,
            image_grid_thw=image_grid_thw,  # still needed by get_rope_index for the vision block
            mm_token_type_ids=mm_token_type_ids,  # drives get_rope_index's text/vision grouping
            **kwargs,
        )