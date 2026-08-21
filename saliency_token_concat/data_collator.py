from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torchvision import transforms

SALIENCY_PLACEHOLDER_TOKEN = "<|saliency_pad|>"
NUM_SALIENCY_TOKENS = 49
TEXT_TOKEN_TYPE = 0

def add_saliency_token(tokenizer, model=None):
    """
    Call this once, right after loading the tokenizer (and before wrapping
    the model in LoRA). Resizes the embedding table by 1 row.

    Returns the new token's id.

    tokenizer vocablary eg:
    "<|endoftext|>": 0,
    "The": 464,
    "cat": 3758,
    """
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [SALIENCY_PLACEHOLDER_TOKEN]}
    ) # how many new special tokens were successfully added to the tokenizer's vocabulary. num_added is 0/1.
    if model is not None and num_added > 0:
        model.resize_token_embeddings(len(tokenizer)) # total size of the tokenizer's vocabulary
    saliency_token_id = tokenizer.convert_tokens_to_ids(SALIENCY_PLACEHOLDER_TOKEN)
    print(f"Added saliency placeholder token '{SALIENCY_PLACEHOLDER_TOKEN}' with id {saliency_token_id}")
    return saliency_token_id

@dataclass
class SaliencyCollator:
    """
    Wraps your existing Qwen3-VL / ms-swift-style collation and adds:
      - saliency placeholder tokens spliced into input_ids/labels/attention_mask
      - raw image + question fields needed to run VisSalFormer at train time
        (saliency_images, saliency_questions) so the model's forward() can
        compute latent tokens on the fly.

    Adapt the field names to whatever your dataset/processor already
    produces -- this is written to be a thin wrapper around your current
    Qwen3-VL processor output, not a replacement for it.
    """

    processor: Any  # Qwen3-VL AutoProcessor
    vision_end_token_id: int
    saliency_token_id: int
    num_saliency_tokens: int = NUM_SALIENCY_TOKENS
    pad_token_id: int = 0 # used for padding.

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        mm_token_type_ids_list = []
        saliency_latents = []

        for ex in batch:
            # ex["input_ids"] / ex["labels"]: already-tokenized Qwen3-VL chat
            # sequence (image tokens + vision_end_token_id + question + answer),
            # produced by your existing preprocessing exactly as before.
            ids = list(ex["input_ids"])
            labels = list(ex["labels"])
            mm_token_type_ids = list(ex["mm_token_type_ids"])

            if len(mm_token_type_ids) != len(ids):  
                raise ValueError(
                    f"mm_token_type_ids length ({len(mm_token_type_ids)}) doesn't match "
                    f"input_ids length ({len(ids)}) BEFORE splicing -- check "
                    f"that your processor produced these together and neither "
                )

            idx = ids.index(self.vision_end_token_id) + 1
            ids = ids[:idx] + [self.saliency_token_id] * self.num_saliency_tokens + ids[idx:]
            # labels: -100 for saliency placeholders, which means ignore this token, do not compute loss on it. Since we only want the model to learn from answering the question, not to predict the placeholder tokens.
            labels = labels[:idx] + [-100] * self.num_saliency_tokens + labels[idx:]
            mm_token_type_ids = mm_token_type_ids[:idx] + [TEXT_TOKEN_TYPE] * self.num_saliency_tokens + mm_token_type_ids[idx:]

            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            mm_token_type_ids_list.append(torch.tensor(mm_token_type_ids, dtype=torch.long))

            saliency_latents.append(ex["saliency_latent"])

        # pad to max length in batch
        max_len = max(x.size(0) for x in input_ids_list)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long) #set paddings to 0.
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long) # so padded tail tokens are ignored during training.
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long) # so padding is ignored by attention 
        mm_token_type_ids = torch.full((len(batch), max_len), TEXT_TOKEN_TYPE, dtype=torch.long) # doesnt matter what we set this to for the padding tokens, since attention_mask will ignore them anyway.

        for i, (ids, lab, mm_ids) in enumerate(zip(input_ids_list, labels_list, mm_token_type_ids_list)):
            # i is the batch index.
            L = ids.size(0)
            input_ids[i, :L] = ids
            labels[i, :L] = lab
            attention_mask[i, :L] = 1
            mm_token_type_ids[i, :L] = mm_ids


        out = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "saliency_latents": torch.stack(saliency_latents, dim=0),
        }

        # carry through whatever pixel_values / image_grid_thw your existing
        # processor already produces for the *original* Qwen3-VL image input
        # (these are unrelated to the saliency placeholder insertion above).

        if "pixel_values" in batch[0]: # processed image tensors
            pixel_vals = [ex["pixel_values"] for ex in batch]
            out["pixel_values"] = torch.cat(pixel_vals, dim=0) if torch.is_tensor(pixel_vals[0]) \
                else pixel_vals 
            # concatenate all tensors in the batch into one combined batch tensor.

        if "image_grid_thw" in batch[0]: # image grid shape (T, Height, Weight)
            grid_thws = [ex["image_grid_thw"] for ex in batch]
            out["image_grid_thw"] = torch.cat(grid_thws, dim=0) if torch.is_tensor(grid_thws[0]) \
                else torch.tensor(grid_thws, dtype=torch.long)

        return out

    # input id: [464, 3758, 151653, 151653, 151653, 151653, 151653]
    # labels 答案: [464, 3758, -100, -100, -100, -100, -100] 这个位置模型应该预测出哪个token (loss)
    # input_embeds: real vectors corresponding to the input ids.
    # attn_mask: 1表示"这个位置是真实token,正常参与attention计算";0表示"这个位置是padding,attention计算时直接忽略它。
