import os
import torch
from torch.utils.data import DataLoader
from env import *
from collections import defaultdict

import argparse
from get_dataset import ChartQADataset
from transformers import SwinModel
from pathlib import Path


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
    model.load_state_dict(checkpoint['model_state_dict'])  # load trained weights
    model.eval()
    print(f"-------------Loaded checkpoint: {ckpt}-------------")

    # get dataset
    dataset = ChartQADataset(
        img_dir=img_dir,
        json_path=json_path,
        max_samples=max_samples
    )

    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=padding_fn_eval, num_workers=8)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_counter = defaultdict(int)

    for batch, (img, query_ids, imgnames, labels) in enumerate(test_dataloader):
        img = img.to(device)
        query_ids = {k: v.to(device) for k, v in query_ids.items()}

        with torch.no_grad():
            latents = model(img, query_ids, return_latent_features=True)  # [B, 49, 768]

        for i in range(latents.shape[0]):
            stem = os.path.splitext(imgnames[i])[0]  # "chart001.png" -> "chart001"
            q_idx = query_counter[stem]               # which query number of this img
            save_path = out_dir / f"{stem}_Q{q_idx}.pt"

            # save as plain fp32 CPU tensor, matching the precompute script's convention
            torch.save(latents[i].detach().cpu().float(), save_path)
            query_counter[stem] += 1

        print(f"batch {batch + 1}/{len(test_dataloader)} done")

    print("-------------Latents saved to folder.-------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--ckpt", type=str, default='./VisSalFormer_weights.tar')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_dir", type=str, default='../data/ChartQA_data/test/png')
    parser.add_argument("--json_path", type=str, default='../data/ChartQA_data/test/test_all_preprocessed.json')
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default='../saliency_token_concat/saliency_latents/test')
    args = vars(parser.parse_args())

    evaluation(device=args['device'],
               ckpt=args['ckpt'],
               batch_size=args['batch_size'],
               img_dir=args['img_dir'],
               json_path=args['json_path'],
               output_dir=args['output_dir'],
               max_samples=args['max_samples'])