import os
os.environ["HF_HOME"] = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface/hub"
os.environ["XDG_CACHE_HOME"] = "/ubc/cs/research/nlp-raid/students/kwang67/.cache"

import time
import torch
from torch.utils.data import random_split
from transformers import AutoProcessor, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model

from qwen3vl_saliency_model import Qwen3VLWithSaliencyBottleneck
from data_collator import add_saliency_token, SaliencyCollator, NUM_SALIENCY_TOKENS

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
VISION_END_TOKEN_STR = "<|vision_end|>"

TRAIN_JSON_PATH = "../data/ChartQA_data/train/train_all_preprocessed.json"
TRAIN_IMG_DIR = "../data/ChartQA_data/train/png"
TRAIN_LATENT_DIR = "./saliency_latents/train"

SPLIT_DATASET_RATIO = 0.05
SPLIT_SEED = 42

def _split_dataset_ratio(full_dataset, ratio: float, seed: int):
    n_eval = max(1, int(len(full_dataset) * ratio))
    n_train = len(full_dataset) - n_eval
    generator = torch.Generator().manual_seed(seed)
    train_subset, eval_subset = random_split(full_dataset, [n_train, n_eval], generator=generator)
    return train_subset, eval_subset

class ProgressLoggingCallback(TrainerCallback):
 
    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.time()
 
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return  # skip eval-only / non-training log calls
        elapsed = time.time() - self._start_time
        step = state.global_step
        max_steps = state.max_steps
        speed = elapsed / step if step > 0 else 0.0
        remaining = speed * (max_steps - step) if max_steps > 0 else 0.0
        logs["global_step/max_steps"] = f"{step}/{max_steps}"
        logs["elapsed_time"] = _format_hms(elapsed)
        logs["remaining_time"] = _format_hms(remaining)
        logs["train_speed(s/it)"] = round(speed, 4)
        if torch.cuda.is_available():
            logs["memory(GiB)"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
 
 
def _format_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"
 

def resolve_vision_end_token_id(tokenizer) -> int:
    token_id = tokenizer.convert_tokens_to_ids(VISION_END_TOKEN_STR)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(
            f"Could not resolve '{VISION_END_TOKEN_STR}' to a real token id "
            f"in this tokenizer (got {token_id}, which matches unk_token_id). "
            f"Check the exact special-token spelling for your installed "
            f"Qwen3-VL processor/tokenizer version."
        )
    return token_id


def build_model_and_tokenizer(attn_implementation: str = "sdpa"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    vision_end_token_id = resolve_vision_end_token_id(tokenizer)

    model = Qwen3VLWithSaliencyBottleneck.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    ) # attn

   
    model.enable_input_require_grads()

    saliency_token_id = add_saliency_token(tokenizer, model=model)

    model.attach_saliency_modules(
        saliency_token_id=saliency_token_id,
        llm_hidden_size=model.config.text_config.hidden_size,  # 4096 for the 8B model
    )

    # --- LoRA on the LLM only -- 
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # keep the projector fully trainable alongside the LoRA adapters
        modules_to_save=["saliency_projector"],
    )
    model = get_peft_model(model, lora_config)

    # sanity check: confirm what's actually trainable before a long run
    model.print_trainable_parameters()

    return model, tokenizer, processor, saliency_token_id, vision_end_token_id


def main():
    model, tokenizer, processor, saliency_token_id, vision_end_token_id = \
        build_model_and_tokenizer()
 

    from chartqa_dataset import ChartQASaliencyDataset
    full_dataset = ChartQASaliencyDataset(
        json_path=TRAIN_JSON_PATH,
        img_dir=TRAIN_IMG_DIR,
        processor=processor,
        latent_dir=TRAIN_LATENT_DIR,
    )

    train_dataset, eval_dataset = _split_dataset_ratio(
        full_dataset, ratio=SPLIT_DATASET_RATIO, seed=SPLIT_SEED
    )
 
    collator = SaliencyCollator(
        processor=processor,
        vision_end_token_id=vision_end_token_id,
        saliency_token_id=saliency_token_id,
        num_saliency_tokens=NUM_SALIENCY_TOKENS,
        pad_token_id=tokenizer.pad_token_id or 0,
    )

    training_args = TrainingArguments(
        output_dir="finetune/checkpoints/saliency_bottleneck",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch size 16, same as before
        num_train_epochs=3,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        bf16=True,
        max_steps = -1,
        logging_steps=30,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[ProgressLoggingCallback()],
    )
    trainer.train()


if __name__ == "__main__":
    main()