"""
bash merge_lora.sh \
    /ubc/cs/research/nlp-raid/students/kwang67/VisSalFormer/finetune/vlm_checkpoints/qwen3vl_baseline_lora/v1-20260729-162917/checkpoint-1320 \
    /ubc/cs/research/nlp-raid/students/kwang67/vissalformer_checkpoints/qwen3vl_baseline_merged
Run:
  
bash merge_lora.sh ./checkpoints/projector_llm/qwen3vl_saliency/v0-20260806-003530/checkpoint-3200 ./checkpoints/projector_llm/qwen3vl_saliency/qwen3vl_saliency_merged
"""
set -euo pipefail
 
ADAPTER_DIR=${1:?"pass the LoRA checkpoint dir to merge"}
OUTPUT_DIR=${2:?"pass where to write the merged standalone model"}
 
# same cache locations as training, so nothing re-downloads
export HF_HOME="/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface"
export HF_HUB_CACHE="/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface/hub"
export XDG_CACHE_HOME="/ubc/cs/research/nlp-raid/students/kwang67/.cache"
export MODELSCOPE_CACHE="/ubc/cs/research/nlp-raid/students/kwang67/.cache/modelscope"
 
echo "=== Merging LoRA adapter ==="
echo "Adapter: $ADAPTER_DIR"
echo "Output:  $OUTPUT_DIR"
 
/ubc/cs/research/nlp-raid/students/kwang67/envs/qwen3vl/bin/swift export \
    --adapters "$ADAPTER_DIR" \
    --merge_lora true \
    --output_dir "$OUTPUT_DIR"
 
echo "=== Done. Merged standalone checkpoint written to $OUTPUT_DIR ==="
echo "Point your existing Qwen3-VL model class's model_path at this directory."