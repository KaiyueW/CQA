from models.llava15 import LLaVA15
from models.qwen3vl import Qwen3VL
from models.bespoke import BespokeMinChart
from models.internvl import InternVL3, InternVLFinetunedBaseline, InternVLFinetunedSaliency
from models.chartr1 import ChartR1
from models.qwen3vl import Qwen3VLFinetunedBaseline, Qwen3VLFinetunedSaliency
 
MODELS = {
    "llava15":  LLaVA15,
    "qwen3vl":  Qwen3VL,
    "bespokeminchart": BespokeMinChart,
    "internvl": InternVL3,
    "chartr1": ChartR1,
    "qwen3vl_finetuned_baseline": Qwen3VLFinetunedBaseline,
    "qwen3vl_finetuned_saliency": Qwen3VLFinetunedSaliency,
    "internvl_finetuned_baseline": InternVLFinetunedBaseline,
    "internvl_finetuned_saliency": InternVLFinetunedSaliency
}
 
def load_model(model_name: str):
    assert model_name in MODELS, f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}"
    return MODELS[model_name]().load()