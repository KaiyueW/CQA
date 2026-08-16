import torch
from transformers import BertModel, SwinModel

import sys
import os
_VISSALFORMER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "visSalFormer"))
sys.path.insert(0, _VISSALFORMER_DIR)

from model_swin import SalFormer

CACHE_DIR = "/tmp/kwang67_cache"
BERT_CKPT = "bert-base-uncased"
SWIN_CKPT = "microsoft/swin-tiny-patch4-window7-224"

 
def remap_swin_state_dict(state_dict: dict) -> dict:
    new_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith("vit."):
            new_state_dict[key] = value
            continue
 
        if key.endswith("attention.self.relative_position_index"):
            # no longer a loadable parameter in the current SwinModel --
            # it's recomputed automatically, so just drop it.
            continue
 
        new_key = key
        new_key = new_key.replace("attention.self.query", "attention.q_proj")
        new_key = new_key.replace("attention.self.key", "attention.k_proj")
        new_key = new_key.replace("attention.self.value", "attention.v_proj")
        new_key = new_key.replace(
            "attention.self.relative_position_bias_table",
            "attention.relative_position_bias.relative_position_bias_table",
        )
        # must run BEFORE the bare "output.dense" replace below, since that
        # one would otherwise also match inside "attention.output.dense"
        new_key = new_key.replace("attention.output.dense", "attention.o_proj")
        new_key = new_key.replace("intermediate.dense", "mlp.fc1")
        new_key = new_key.replace("output.dense", "mlp.fc2")
 
        new_state_dict[new_key] = value
 
    return new_state_dict

def load_frozen_vissalformer(
    ckpt_path: str,
    device: str = "cuda",
) -> SalFormer:
    # text encoder -- identical to evaluation()
    bert = BertModel.from_pretrained(BERT_CKPT, cache_dir=CACHE_DIR)
    print("-------------BertModel loaded-------------")

    # img encoder -- identical to evaluation()
    vit = SwinModel.from_pretrained(SWIN_CKPT, cache_dir=CACHE_DIR)
    print("-------------SwinModel loaded-------------")

    model = SalFormer(vit, bert).to(device)
    checkpoint = torch.load(ckpt_path)
    remapped_state_dict = remap_swin_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(remapped_state_dict)  # load trained weights
    print(f"-------------Loaded checkpoint: {ckpt_path}-------------")

    model.eval()  # eval mode is more stable/repeatable than train mode (dropout/batchnorm off)

    # --- freeze everything: this model is a fixed feature extractor here,
    # it is NOT being fine-tuned as part of the Qwen3-VL LoRA run. ---
    for p in model.parameters():
        p.requires_grad_(False)

    return model