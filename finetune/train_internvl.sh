#!/bin/bash
# LoRA fine-tuning for InternVL baseline: chart image + question -> answer.
#
# Requires: pip install ms-swift -U
# Docs: https://github.com/modelscope/ms-swift
#
# Usage:
#   bash train_internvl.sh

set -euo pipefail

export HF_HOME="/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface"
export HF_HUB_CACHE="/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface/hub"
export XDG_CACHE_HOME="/ubc/cs/research/nlp-raid/students/kwang67/.cache"
export MODELSCOPE_CACHE="/ubc/cs/research/nlp-raid/students/kwang67/.cache/modelscope"

# === CHANGE === must match whatever InternVL checkpoint you've been
# benchmarking in your zero-shot/few-shot pipeline
MODEL_ID="OpenGVLab/InternVL3-8B-hf"

DATA_FILE="./data/train_saliency.jsonl"
OUTPUT_DIR="/ubc/cs/research/nlp-raid/students/kwang67/VisSalFormer/finetune/vlm_checkpoints/internvl_saliency_lora"

echo "=== Training internvl_saliency ==="
echo "Model:  $MODEL_ID"
echo "Data:   $DATA_FILE"
echo "Output: $OUTPUT_DIR"

/ubc/cs/research/nlp-raid/students/kwang67/envs/qwen3vl/bin/swift sft \
    --model "$MODEL_ID" \
    --dataset "$DATA_FILE" \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --max_length 4096 \
    --output_dir "$OUTPUT_DIR" \
    --gradient_checkpointing true \
    --dataloader_num_workers 4 \
    --report_to none \
    --split_dataset_ratio 0.05

echo "=== Done. LoRA adapter saved under $OUTPUT_DIR ==="
echo "Next: run merge_lora.py to produce a standalone checkpoint for your existing eval pipeline."
