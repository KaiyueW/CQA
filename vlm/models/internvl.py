import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from models.base import BaseVLM

class InternVL3(BaseVLM):
    MODEL_ID  = "OpenGVLab/InternVL3-8B-hf"
    CACHE_DIR = "/ubc/cs/research/nlp-raid/students/kwang67/.cache"

    def load(self):
        self.model = AutoModelForImageTextToText.from_pretrained(
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

        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_length:]
        response = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=True
        )

        return response.strip()