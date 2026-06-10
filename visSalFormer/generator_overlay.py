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

def save_overlay(original_img_path, saliency_tensor, save_path):
    # load the original image
    original_img = Image.open(original_img_path).convert("RGB")
    origin_w, origin_h = original_img.size
    img = np.array(original_img) / 255.0 # [H, W, 3], normalize to 0-1

    # saliency map → colormap
    sal = saliency_tensor.squeeze().cpu().numpy()  # [Height, Weight]
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-10)  # normalize to 0-1
    heatmap = cm.jet(sal)[:, :, :3]  # [H, W, 3], low->blue, high->red

    # resize heatmap
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)) #resize tensor to uint8
    heatmap_pil = heatmap_pil.resize((origin_w, origin_h), Image.BILINEAR) #resize to origin img size
    heatmap = np.array(heatmap_pil) / 255.0 #resize tensor to float

    # overlay
    overlay = 0.3 * img + 0.7 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.imsave(save_path, overlay)

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

    for batch, (img, input_ids, imgnames, labels) in enumerate(test_dataloader): #for each example in the test dataset
        img = img.to(device)
        input_ids = {k: v.to(device) for k, v in input_ids.items()}

        with torch.no_grad():
            preds = model(img, input_ids) # predicted saliency maps

        for i in range(preds.shape[0]):
            stem = os.path.splitext(imgnames[i])[0] #"chart001.png" → "chart001"
            q_idx = query_counter[stem]          # which query number of this img
            save_path = f"{output_dir}/{stem}_Q{q_idx}.png"
            img_path = os.path.join(img_dir, imgnames[i])
            save_overlay(img_path, preds[i], save_path)
            query_counter[stem] += 1
        
    print("-------------Results saved to folder.-------------")

    # Metrics evaluation (KL, CC, NSS) can be implemented here if needed, but since the question only asks for generating saliency maps, we can skip it for now.

    # kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    # test_kl, test_cc, test_nss = 0,0,0 
    # for batch, (img, input_ids, fix, hm, name) in enumerate(test_dataloader):
    #     img = img.to(device)
    #     input_ids = input_ids.to(device)
    #     fix = fix.to(device)
    #     hm = hm.to(device)

    #     y = model(img, input_ids)
        
    #     y_sum = y.view(y.shape[0], -1).sum(1, keepdim=True)
    #     y_distribution = y / (y_sum[:, :, None, None] + eps)

    #     hm_sum = hm.view(y.shape[0], -1).sum(1, keepdim=True)
    #     hm_distribution = hm / (hm_sum[:, :, None, None] + eps)
    #     hm_distribution = hm_distribution + eps
    #     hm_distribution = hm_distribution / (1+eps)

    #     if fix.sum() != 0:
    #         normal_y = (y-y.mean())/y.std()
    #         nss = torch.sum(normal_y*fix)/fix.sum()
    #     else:
    #         nss = torch.Tensor([0.0]).to(device)
    #     kl = kl_loss(torch.log(y_distribution), torch.log(hm_distribution))

    #     vy = y - torch.mean(y)
    #     vhm = hm - torch.mean(hm)  

    #     if (torch.sqrt(torch.sum(vy ** 2)) * torch.sqrt(torch.sum(vhm ** 2))) != 0:
    #         cc = torch.sum(vy * vhm) / (torch.sqrt(torch.sum(vy ** 2)) * torch.sqrt(torch.sum(vhm ** 2)))
    #     else: 
    #         cc = torch.Tensor([0.0]).to(device)
        
    #     test_kl += kl.item()/len(test_dataloader)
    #     test_cc += cc.item()/len(test_dataloader)
    #     test_nss += nss.item()/len(test_dataloader)

    #     for i in range(0, y.shape[0]):
    #         save_image(y[i], f"./eval_results/{name[i]}")

    # print("kl:", test_kl, "cc", test_cc, "nss", test_nss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--ckpt", type=str, default='./ckpt/model_bert_freeze_10kl_5cc_2nss.tar')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_dir", type=str, default='train/png')
    parser.add_argument("--json_path", type=str, default='train/train_human.json')
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default='./saliency_haha_0.7_heatmap')
    args = vars(parser.parse_args())

    evaluation(device = args['device'], 
               ckpt = args['ckpt'], 
               batch_size = args['batch_size'], 
               img_dir=args['img_dir'],
               json_path=args['json_path'],
               output_dir=args['output_dir'],
               max_samples=args['max_samples'])
