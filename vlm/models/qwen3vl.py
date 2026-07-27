import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from models.base import BaseVLM

class Qwen3VL(BaseVLM):
    MODEL_ID  = "Qwen/Qwen3-VL-8B-Instruct"
    CACHE_DIR = "/ubc/cs/research/nlp-raid/students/kwang67/.cache"

    def load(self):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=self.CACHE_DIR
        )
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            cache_dir=self.CACHE_DIR
        )
        print(f"Loaded: {self.MODEL_ID}")
        return self

    def generate(self, conversation: list, max_new_tokens: int) -> str:
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to("cuda")

        # apply_chat_template 可能会加 token_type_ids，Qwen3 不需要，去掉
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        # generated output = prompt tokens + newly generated tokens
        prompt_length = inputs["input_ids"].shape[1]

        generated_ids = output_ids[:, prompt_length:] # [[1]]

        response = self.processor.decode(
            generated_ids[0], #[1]
            skip_special_tokens=True
        )

        return response.strip()