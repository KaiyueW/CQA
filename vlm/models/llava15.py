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
        self.processor.patch_size = self.model.config.vision_config.patch_size
        self.processor.vision_feature_select_strategy = self.model.config.vision_feature_select_strategy

        print(f"Loaded: {self.MODEL_ID}")
        return self


    def generate(self, prompt: list) -> str:
        # since just use conversation can't handle image input (no pixel_values), we need to manually process the prompt and images into model input format.
        # Llava separates the text and images pipelines.
        prompt_text = self.processor.apply_chat_template(
            prompt,
            add_generation_prompt=True
        ) # get the text part of the prompt.
        
        images = []

        for msg in prompt:
            for item in msg["content"]:
                if item["type"] == "image":
                    images.append(item["image"])

        inputs = self.processor(
            text=prompt_text,
            images=images,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )

        # Decode the newly generated tokens (skip the prompt)
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()