import torch
from transformers import AutoProcessor, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

from qwen3vl_saliency_model import Qwen3VLWithSaliencyBottleneck
from data_collator import add_saliency_token, SaliencyCollator, NUM_SALIENCY_TOKENS

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
VISION_END_TOKEN_STR = "<|vision_end|>"

VISSALFORMER_CKPT_PATH = "../visSalFormer/VisSalFormer_weights.tar"
BERT_CKPT = "bert-base-uncased"
BERT_CACHE_DIR = "/tmp/kwang67_cache"
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


def build_model_and_tokenizer(vissalformer_frozen):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    vision_end_token_id = resolve_vision_end_token_id(tokenizer)

    model = Qwen3VLWithSaliencyBottleneck.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

   
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


def main():
    from vissalformer_loading import load_frozen_vissalformer
    vissalformer = load_frozen_vissalformer(
        ckpt_path=VISSALFORMER_CKPT_PATH,
        device="cuda",
    )
 
    model, tokenizer, processor, saliency_token_id, vision_end_token_id = \
        build_model_and_tokenizer(vissalformer)
 

    from chartqa_dataset import ChartQASaliencyDataset
    train_dataset = ChartQASaliencyDataset(
        json_path=TRAIN_JSON_PATH,
        img_dir=TRAIN_IMG_DIR,
        processor=processor,
    )
 
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_CKPT, cache_dir=BERT_CACHE_DIR)

    # Built AFTER train_dataset so we can reuse its exact bert_tokenizer
    # instance/checkpoint -- the collator's tokenization of saliency
    # questions must match whatever VisSalFormer was trained against, same
    # as the dataset's own BERT tokenizer.
    collator = SaliencyCollator(
        processor=processor,
        vision_end_token_id=vision_end_token_id,
        saliency_token_id=saliency_token_id,
        bert_tokenizer=bert_tokenizer,
        num_saliency_tokens=NUM_SALIENCY_TOKENS,
        pad_token_id=tokenizer.pad_token_id or 0,
    )

    training_args = TrainingArguments(
        output_dir="finetune/checkpoints/saliency_bottleneck",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch size 16, same as before
        num_train_epochs=3,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()