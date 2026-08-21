import torch

latent = torch.load(
    "../saliency_token_concat/saliency_latents/test/41699051005347_Q0.pt",
    map_location="cpu",
)

print(latent)
print("shape:", latent.shape)
print("dtype:", latent.dtype)