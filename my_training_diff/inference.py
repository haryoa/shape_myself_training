from my_training_diff.diffusiion_utils import extract_alpha_from_timesteps
import torch
from tqdm import tqdm

@torch.no_grad()
def p_sample(
    model,
    images,
    timesteps,
    t_index,
    betas,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
):
    betas_t = extract_alpha_from_timesteps(betas, timesteps, images.shape)
    sqrt_one_minus_alphas_cumprod_t = extract_alpha_from_timesteps(
        sqrt_one_minus_alphas_cumprod, timesteps, images.shape
    )
    sqrt_recip_alphas_t = extract_alpha_from_timesteps(sqrt_recip_alphas, timesteps, images.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        images - betas_t * model(images, timesteps) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract_alpha_from_timesteps(posterior_variance, timesteps, images.shape)
        noise = torch.randn_like(images)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape, timesteps):
    device = next(model.parameters()).device

    batch_size= shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(
        reversed(range(0, timesteps)), desc="sampling loop time step", total=timesteps
    ):
        img = p_sample(
            model, img, torch.full((batch_size,), i, device=device, dtype=torch.long), i
        )
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
