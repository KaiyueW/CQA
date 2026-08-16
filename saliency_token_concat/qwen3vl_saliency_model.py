"""
Subclasses Qwen3VLForConditionalGeneration to inject the projected saliency
tokens into inputs_embeds at forward time, right at the placeholder
positions the collator already reserved.
"""

from typing import List, Optional, Dict

import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration

from saliency_extractor import VisSalFormerLatentExtractor
from saliency_projector import SaliencyProjector


class Qwen3VLWithSaliencyBottleneck(Qwen3VLForConditionalGeneration):
    """
    Drop-in replacement for Qwen3VLForConditionalGeneration that also carries
    a frozen VisSalFormer + trainable projector, and knows how to splice
    projected saliency tokens into inputs_embeds.

    """

    def attach_saliency_modules(
        self,
        vissalformer: nn.Module,
        saliency_token_id: int,
        llm_hidden_size: int,
        saliency_dim: int = 768,
        use_layernorm: bool = True,
    ):
        self.saliency_backbone = VisSalFormerLatentExtractor(vissalformer)
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
        saliency_pixel_values: Optional[torch.Tensor] = None,
        saliency_q_inputs: Optional[Dict[str, torch.Tensor]] = None,
        # saliency_q_inputs = {
        #     "input_ids": tensor([[101, 2129, 2116, ...], [101, 2003, 2009, ...]]),      # [B, L]
        #     "attention_mask": tensor([[1, 1, 1, ...], [1, 1, 1, ...]]),                  # [B, L]
        #     "token_type_ids": tensor([[0, 0, 0, ...], [0, 0, 0, ...]]),                  # [B, L]
        # }
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

            # --- 1. scatter the ORIGINAL vision tokens (unchanged Qwen3-VL logic) --- 填坑
            if pixel_values is not None:
                image_outputs = self.get_image_features(pixel_values, image_grid_thw) # through ViT.
                image_embeds = image_outputs.pooler_output 
                '''
                Qwen3-VL的vision tower,不再是"只输出最后一层的图片特征,scatter进inputs_embeds一次就完事"这么简单。它额外从ViT的多个中间层抽取特征(deepstack_features),然后在LLM decoder的前几层(默认是第8/16/24层附近,具体看deepstack_visual_indexes这个config),在原图片token的位置上,再叠加注入一次这些多层次的视觉特征——这是Qwen3-VL相比Qwen2.5-VL的一个核心改进点,目的是让模型既有细粒度细节(浅层ViT特征)又有高层语义(深层ViT特征)。
而get_image_features(...)返回的这个BaseModelOutputWithDeepstackFeatures对象,就是把这两部分打包在一起:
.last_hidden_state:最后一层的图片特征(就是我们一直以为的、要scatter进inputs_embeds的那个tensor)
.deepstack_features:一个list,多个中间层的特征,专门用来注入decoder前几层的
我们现在这个自定义forward,是手动重新实现了"把图片特征scatter进inputs_embeds"这一步,然后直接调用super().forward(inputs_embeds=..., ...)——跳过了Qwen3-VL原生forward里"处理pixel_values→顺便把deepstack_features也传给decoder"这一整套机制。也就是说,即使我现在只是简单地把.to()报错修复成取.last_hidden_state,这个模型依然会丢失DeepStack这部分能力,跟官方原生的Qwen3-VL在"图片信息怎么融进LLM"这件事上,已经不是完全等价的架构了。

                '''
                image_embeds = torch.cat(image_embeds, dim=0) if isinstance(image_embeds, (list, tuple)) \
                    else image_embeds
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1) # return a boolean tensor of shape [B, L, 1] where True indicates the position of the image token in the input_ids.
                image_mask = image_mask.expand_as(inputs_embeds).to(inputs_embeds.device) # [B, L, Hidden_size]
                inputs_embeds = inputs_embeds.masked_scatter(
                    image_mask, image_embeds.to(inputs_embeds.dtype)
                )

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
            if saliency_pixel_values is not None and hasattr(self, "saliency_backbone"):
                latent_tokens = self.saliency_backbone(
                    saliency_pixel_values, saliency_q_inputs
                )  # [B,49,768]
                target_dtype = next(self.saliency_projector.parameters()).dtype
                print(f"Saliency projector params dtype: {target_dtype}, latent tokens dtype: {latent_tokens.dtype}")
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
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            image_grid_thw=image_grid_thw,  # still needed by get_rope_index for the vision block
            mm_token_type_ids=mm_token_type_ids,  # drives get_rope_index's text/vision grouping
            **kwargs,
        )