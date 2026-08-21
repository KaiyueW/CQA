"""
Data-pipeline smoke test -- NOT part of the real training run.

Deliberately does NOT load the full Qwen3-VL 8B model (slow, needs a GPU
with real memory). Only exercises the parts that are cheap and easy to get
wrong:
  1. ChartQASaliencyDataset.__getitem__  -- prompt construction, tokenization,
                                              loading precomputed saliency latents
  2. SaliencyCollator.__call__            -- placeholder insertion, label
                                              masking, latent stacking

Run:
    python smoke_test_data_pipeline.py
"""

import os

os.environ["HF_HOME"] = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/ubc/cs/research/nlp-raid/students/kwang67/.cache/huggingface/hub"
os.environ["XDG_CACHE_HOME"] = "/ubc/cs/research/nlp-raid/students/kwang67/.cache"

import torch
from transformers import AutoProcessor, AutoTokenizer

from data_collator import add_saliency_token, SaliencyCollator, NUM_SALIENCY_TOKENS
from chartqa_dataset import ChartQASaliencyDataset

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
VISION_END_TOKEN_STR = "<|vision_end|>"

TRAIN_JSON_PATH = "../data/ChartQA_data/train/train_all_preprocessed1.json"
TRAIN_IMG_DIR = "../data/ChartQA_data/train/png"
# precomputed saliency latents live here now -- one <stem>.pt per chart,
# produced offline by precompute_saliency_latents.py. This replaces the
# old on-the-fly VisSalFormerLatentExtractor call in STEP 5.
LATENT_DIR = "./saliency_latents/train"

BATCH_SIZE = 2  # small, just enough to exercise padding logic


def main():
    print("=" * 70)
    print("STEP 1: load tokenizer/processor (NOT the full 8B model)")
    print("=" * 70)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    vision_end_token_id = tokenizer.convert_tokens_to_ids(VISION_END_TOKEN_STR)
    print(f"vision_end_token_id = {vision_end_token_id}")

    # model=None here -- we don't have a model loaded to resize embeddings on,
    # that only matters once you actually load Qwen3VLWithSaliencyBottleneck
    # in the real train.py. We only need the token id here.
    saliency_token_id = add_saliency_token(tokenizer, model=None)
    print(f"saliency_token_id   = {saliency_token_id}")

    print("\n" + "=" * 70)
    print("STEP 2: build dataset, pull a couple of raw samples")
    print("=" * 70)
    dataset = ChartQASaliencyDataset(
        json_path=TRAIN_JSON_PATH,
        img_dir=TRAIN_IMG_DIR,
        processor=processor,
        latent_dir=LATENT_DIR,
        max_samples=BATCH_SIZE,
    )
    print(f"dataset size (after max_samples cap) = {len(dataset)}")

    raw_samples = [dataset[i] for i in range(min(BATCH_SIZE, len(dataset)))]

    for i, ex in enumerate(raw_samples):
        print(f"\n--- raw sample {i} (BEFORE collator, no saliency placeholders yet) ---")
        print(f"input_ids length: {len(ex['input_ids'])}")
        print(f"labels length:    {len(ex['labels'])}")
        print(f"mm_token_type_ids length: {len(ex['mm_token_type_ids'])}")
        n_masked = sum(1 for l in ex["labels"] if l == -100)
        n_answer = len(ex["labels"]) - n_masked
        print(f"labels: {n_masked} masked (-100) tokens, {n_answer} real-answer tokens")
        # decode the whole thing so you can visually sanity-check the prompt
        decoded_full = processor.tokenizer.decode(ex["input_ids"], skip_special_tokens=False)
        print(f"decoded full sequence (prompt+answer):\n{decoded_full}")
        # decode just the answer tail, should match the ground-truth label
        answer_only_ids = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
        decoded_answer = processor.tokenizer.decode(answer_only_ids, skip_special_tokens=True)
        print(f"decoded ANSWER-only tokens (should match ground-truth label): '{decoded_answer}'")
        print(f"saliency_latent shape (loaded from disk): {tuple(ex['saliency_latent'].shape)}")

    print("\n" + "=" * 70)
    print("STEP 3: build collator, run it on this mini-batch")
    print("=" * 70)
    collator = SaliencyCollator(
        processor=processor,
        vision_end_token_id=vision_end_token_id,
        saliency_token_id=saliency_token_id,
        num_saliency_tokens=NUM_SALIENCY_TOKENS,
        pad_token_id=tokenizer.pad_token_id or 0,
    )
    batch = collator(raw_samples)

    print(f"batched input_ids shape:      {batch['input_ids'].shape}")
    print(f"batched labels shape:         {batch['labels'].shape}")
    print(f"batched attention_mask shape: {batch['attention_mask'].shape}")
    print(f"batched mm_token_type_ids shape: {batch['mm_token_type_ids'].shape}")
    print(f"batched saliency_latents shape: {batch['saliency_latents'].shape}")

    print("\n" + "=" * 70)
    print("STEP 4: verify saliency placeholder insertion, per sample")
    print("=" * 70)
    for i in range(batch["input_ids"].shape[0]):
        ids_row = batch["input_ids"][i].tolist()
        labels_row = batch["labels"][i].tolist()

        # 1. exact count check
        n_sal = sum(1 for t in ids_row if t == saliency_token_id)
        status = "OK" if n_sal == NUM_SALIENCY_TOKENS else "*** MISMATCH ***"
        print(f"\nsample {i}: found {n_sal} saliency placeholder tokens (expected {NUM_SALIENCY_TOKENS}) [{status}]")

        # 2. contiguity check -- they should all be consecutive, right after
        # vision_end_token_id
        sal_positions = [idx for idx, t in enumerate(ids_row) if t == saliency_token_id]
        if sal_positions:
            is_contiguous = sal_positions == list(range(sal_positions[0], sal_positions[-1] + 1))
            print(f"  positions: {sal_positions[0]}..{sal_positions[-1]} "
                  f"(contiguous: {is_contiguous})")
            vis_end_pos = ids_row.index(vision_end_token_id)
            print(f"  vision_end_token_id is at position {vis_end_pos} "
                  f"(saliency should start right after, at {vis_end_pos + 1}): "
                  f"{'OK' if sal_positions[0] == vis_end_pos + 1 else '*** MISMATCH ***'}")

        # 3. label masking check -- every saliency position must be -100
        sal_labels = [labels_row[p] for p in sal_positions]
        all_masked = all(l == -100 for l in sal_labels)
        print(f"  all saliency positions masked to -100 in labels: "
              f"{'OK' if all_masked else '*** MISMATCH, got: ' + str(sal_labels) + ' ***'}")

    print("\n" + "=" * 70)
    print("STEP 5: sanity-check the PRECOMPUTED saliency latents in the batch")
    print("=" * 70)
    print("(No VisSalFormer/BERT loaded here anymore -- latents were already")
    print(" computed offline by precompute_saliency_latents.py and are just")
    print(" being read off disk by the dataset + stacked by the collator.)")

    latent_tokens = batch["saliency_latents"]

    print(f"\nlatent_tokens shape: {tuple(latent_tokens.shape)} "
          f"(expected ({BATCH_SIZE}, 49, 768))")
    print(f"latent_tokens dtype: {latent_tokens.dtype}")

    has_nan = torch.isnan(latent_tokens).any().item()
    has_inf = torch.isinf(latent_tokens).any().item()
    print(f"contains NaN: {has_nan} {'*** MISMATCH ***' if has_nan else '(OK)'}")
    print(f"contains Inf: {has_inf} {'*** MISMATCH ***' if has_inf else '(OK)'}")

    print(f"\nvalue stats: mean={latent_tokens.mean().item():.4f}, "
          f"std={latent_tokens.std().item():.4f}, "
          f"min={latent_tokens.min().item():.4f}, "
          f"max={latent_tokens.max().item():.4f}")

    # relu1 was the last op in SalFormer's return_latent_features=True branch
    # when these were precomputed, so every value should still be >= 0 --
    # if you see negatives here, something in precompute_saliency_latents.py
    # (or the .pt files themselves) is off.
    all_non_negative = (latent_tokens >= 0).all().item()
    print(f"all values >= 0 (SalFormer's last op is ReLU): "
          f"{'OK' if all_non_negative else '*** MISMATCH, ReLU output should never be negative ***'}")

    # sanity check the samples in the batch don't produce identical latents
    # (would suggest every .pt file is loading the same cached tensor, e.g.
    # a stem-collision bug in precompute_saliency_latents.py)
    if BATCH_SIZE >= 2:
        identical = torch.allclose(latent_tokens[0], latent_tokens[1])
        print(f"sample 0 and sample 1 latents identical: "
              f"{'*** MISMATCH, expected different charts to differ ***' if identical else 'OK (different, as expected)'}")

    print("\nFull smoke test finished. Review all '*** MISMATCH ***' lines above.")


if __name__ == "__main__":
    main()