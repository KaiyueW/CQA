import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from models.base import BaseVLM

class LLaVA15(BaseVLM): # LLaVA15 inherits from BaseVLM, must implement load and generate methods.
    MODEL_ID  = "llava-hf/llava-1.5-7b-hf"
    CACHE_DIR = "/ubc/cs/research/nlp-raid/students/kwang67/.cache"

    def load(self):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=self.CACHE_DIR
        )
        # process images and text into a tensor that model can use.
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            cache_dir=self.CACHE_DIR
        )
        print(f"Loaded: {self.MODEL_ID}")
        return self

    def generate(self, prompt: str, images: list) -> str:
        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )

        # Decode the newly generated tokens (skip the prompt)
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()