from models.llava15 import LLaVA15
# from models.qwen3vl import Qwen3VL
 
MODELS = {
    "llava15":  LLaVA15,
    # "qwen3vl":  Qwen3VL,
}
 
def load_model(model_name: str):
    assert model_name in MODELS, f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}"
    return MODELS[model_name]().load()