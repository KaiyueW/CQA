#!/bin/bash
# LoRA fine-tuning for Qwen3-VL saliency: chart image + question -> answer.
#
# Requires: pip install ms-swift -U
# Docs: https://github.com/modelscope/ms-swift
#
# Usage:
#   bash train_qwen3vl.sh

set -euo pipefail

export HF_HOME="/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface"
export HF_HUB_CACHE="/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface/hub"
export XDG_CACHE_HOME="/ubc/cs/research/nlp-raid/students/kwang67/.cache"
export MODELSCOPE_CACHE="/ubc/cs/research/nlp-raid/students/kwang67/.cache/modelscope"

# === CHANGE === must match whatever Qwen3-VL checkpoint you've been
# benchmarking in your zero-shot/few-shot pipeline
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"

DATA_FILE="./data/train_all_no_saliency.jsonl"
OUTPUT_DIR="/ubc/cs/research/nlp-raid/students/kwang67/VisSalFormer/finetune/checkpoints/vision/qwen3vl_no_saliency"
LOG_DIR="/ubc/cs/research/nlp-raid/students/kwang67/VisSalFormer/finetune/tb_logs/vision/qwen3vl_no_saliency"

echo "=== Training qwen3vl_no_saliency ==="
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
    --modules_to_save model.visual.merger model.visual.deepstack_merger_list.0 model.visual.deepstack_merger_list.1 model.visual.deepstack_merger_list.2 \
    --freeze_vit true \
    --freeze_aligner false \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 20 \
    --max_length 4096 \
    --output_dir "$OUTPUT_DIR" \
    --gradient_checkpointing true \
    --dataloader_num_workers 6 \
    --report_to tensorboard \
    --logging_dir "$LOG_DIR" \
    --metric_for_best_model eval_loss \
    --load_best_model_at_end true \
    --split_dataset_ratio 0.05
    

echo "=== Done. LoRA adapter saved under $OUTPUT_DIR ==="
echo "Next: run merge_lora.py to produce a standalone checkpoint for your existing eval pipeline."
