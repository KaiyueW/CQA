from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

config = Qwen3VLConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
model = Qwen3VLForConditionalGeneration(config)

for name, module in model.named_modules():
    if any(k in name.lower() for k in ["project", "merger", "visual"]):
        print(name)