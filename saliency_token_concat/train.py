import os

# CHANGE: added -- matches the cache setup in your other scripts
# (inference_finetune.py, check_mm_token_type_ids.py, etc.) so weights are
# read from/written to the same place instead of re-downloading to the
# default ~/.cache location.
os.environ["HF_HOME"] = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface/hub"
os.environ["XDG_CACHE_HOME"] = "/ubc/cs/research/nlp-raid/students/kwang67/.cache"

import torch
from transformers import AutoProcessor, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

from qwen3vl_saliency_model import Qwen3VLWithSaliencyBottleneck
from data_collator import add_saliency_token, SaliencyCollator, NUM_SALIENCY_TOKENS

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
VISION_END_TOKEN_STR = "<|vision_end|>"

# CHANGE: filled in with your actual paths (confirmed via mm.py smoke test).
VISSALFORMER_CKPT_PATH = "../visSalFormer/VisSalFormer_weights.tar"
BERT_CKPT = "bert-base-uncased"  # must match what SalFormer was trained with
BERT_CACHE_DIR = "/tmp/kwang67_cache"  # matches tokenizer_bert.py exactly
TRAIN_JSON_PATH = "../data/ChartQA_data/train/train_all_preprocessed1.json"
TRAIN_IMG_DIR = "../data/ChartQA_data/train/png"


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


def build_model_and_tokenizer(vissalformer_frozen, attn_implementation: str = "sdpa"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    vision_end_token_id = resolve_vision_end_token_id(tokenizer)

    model = Qwen3VLWithSaliencyBottleneck.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    ) # sdpa

    model.enable_input_require_grads()

    saliency_token_id = add_saliency_token(tokenizer, model=model)

    model.attach_saliency_modules(
        vissalformer=vissalformer_frozen,
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


def main(smoke_test: bool = False):
    # CHANGE: was a TODO -- now actually loads + freezes VisSalFormer,
    # mirroring your existing eval pipeline's loading logic.
    from vissalformer_loading import load_frozen_vissalformer
    vissalformer = load_frozen_vissalformer(
        ckpt_path=VISSALFORMER_CKPT_PATH,
        device="cuda",
    )

    model, tokenizer, processor, saliency_token_id, vision_end_token_id = \
        build_model_and_tokenizer(vissalformer)

    # CHANGE: was a TODO -- now actually constructs the dataset from your
    # ChartQA json + png directory. See chartqa_dataset.py (new file).
    from chartqa_dataset import ChartQASaliencyDataset
    train_dataset = ChartQASaliencyDataset(
        json_path=TRAIN_JSON_PATH,
        img_dir=TRAIN_IMG_DIR,
        processor=processor,
        # CHANGE: smoke_test caps the dataset to a handful of samples so a
        # "step" actually happens almost immediately, instead of Trainer
        # spending time iterating over your full training set.
        max_samples=8 if smoke_test else None,
    )

    # CHANGE: added -- SaliencyCollator now needs a BERT tokenizer to
    # pre-tokenize saliency_questions for VisSalFormer. Must be the SAME
    # checkpoint + cache_dir tokenizer_bert.py used (AutoTokenizer, not
    # BertTokenizer -- matching your existing eval pipeline exactly).
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_CKPT, cache_dir=BERT_CACHE_DIR)

    collator = SaliencyCollator(
        processor=processor,
        vision_end_token_id=vision_end_token_id,
        saliency_token_id=saliency_token_id,
        bert_tokenizer=bert_tokenizer,
        num_saliency_tokens=NUM_SALIENCY_TOKENS,
        pad_token_id=tokenizer.pad_token_id or 0,
    )

    training_args = TrainingArguments(
        # CHANGE: smoke_test writes to a throwaway dir so it never collides
        # with (or gets confused for) a real training run's checkpoints.
        output_dir="finetune/checkpoints/saliency_bottleneck_SMOKE_TEST" if smoke_test
                    else "finetune/checkpoints/saliency_bottleneck",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch size 16, same as before
        num_train_epochs=3,
        learning_rate=1e-4,
        bf16=True,
        # CHANGE: smoke_test forces exactly a few optimizer steps and logs
        # every step (instead of every 10), so you see loss immediately and
        # the run ends in well under a minute of actual training time (model
        # loading itself will still take a bit).
        max_steps=2 if smoke_test else -1,
        logging_steps=1 if smoke_test else 10,
        save_strategy="no" if smoke_test else "epoch",
        gradient_checkpointing=True,
        report_to=[] if smoke_test else ["tensorboard"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train()

    if smoke_test:
        print("\n" + "=" * 70)
        print("SMOKE TEST PASSED: forward + backward + optimizer step all ran")
        print("without error. See the loss values printed above (should be")
        print("finite numbers, not NaN). This did NOT save a real checkpoint")
        print("(save_strategy='no') -- rerun with smoke_test=False for a real")
        print("training run.")
        print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Run a minimal 2-step, 8-sample end-to-end check instead of real training.",
    )
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
