import torch
from torch.utils.data import DataLoader
from env import *
from collections import defaultdict

import argparse
from get_dataset import ChartQADataset
from transformers import SwinModel
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision.utils import save_image


def evaluation(ckpt: str, device: str, batch_size: int, img_dir: str, json_path: str, output_dir: str, max_samples: int):
    from model_swin import SalFormer
    from transformers import BertModel
    from tokenizer_bert import padding_fn_eval
    
    # text encoder
    llm = BertModel.from_pretrained("bert-base-uncased", cache_dir="/tmp/kwang67_cache")
    print('-------------BertModel loaded-------------')

    # img encoder
    vit = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224", cache_dir="/tmp/kwang67_cache")
    print('-------------SwinModel loaded-------------')

    model = SalFormer(vit, llm).to(device)
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict']) #load trained weights
    model.eval() #eval mode is more stable and repeatable，while train mode is more random.
    print(f"-------------Loaded checkpoint: {ckpt}-------------")

    # get dataset
    dataset = ChartQADataset(
        img_dir = img_dir,
        json_path = json_path,
        max_samples = max_samples
    )

    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=padding_fn_eval, num_workers=8)
    # concat several data entries into a batch after getting the dataset.
    # then feed them into the model to get batch_size saliency maps. Then feed another batch to the model to get the saliency maps.

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    query_counter = defaultdict(int)

    for batch, (img, query_ids, imgnames, labels) in enumerate(test_dataloader): #for each example in the test dataset
        img = img.to(device)
        query_ids = {k: v.to(device) for k, v in query_ids.items()}

        with torch.no_grad():
            preds = model(img, query_ids) # predicted saliency maps

        for i in range(preds.shape[0]):
            stem = os.path.splitext(imgnames[i])[0] #"chart001.png" → "chart001"
            q_idx = query_counter[stem]          # which query number of this img
            save_path = f"{output_dir}/{stem}_Q{q_idx}.png"
            save_image(preds[i], save_path)
            query_counter[stem] += 1
        
    print("-------------Results saved to folder.-------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--ckpt", type=str, default='./ckpt/model_bert_freeze_10kl_5cc_2nss.tar')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_dir", type=str, default='train/png')
    parser.add_argument("--json_path", type=str, default='train/train_human.json')
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default='./heatmaps')
    args = vars(parser.parse_args())

    evaluation(device = args['device'], 
               ckpt = args['ckpt'], 
               batch_size = args['batch_size'], 
               img_dir=args['img_dir'],
               json_path=args['json_path'],
               output_dir=args['output_dir'],
               max_samples=args['max_samples'])