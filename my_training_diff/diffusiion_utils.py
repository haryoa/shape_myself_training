# Forward Diffusion
from typing import List
import torch
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class AlphaVars:
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_proba: torch.Tensor


def linear_beta_schedule(timesteps: int, beta_start: float = 0.001, beta_end=0.02):
    """
    Return linear schedule for beta in model diffusion

    Parameters
    ----------
    timesteps : int
        total number of timesteps
    beta_start : float, optional
        start of the schedule, by default 0.001
    beta_end : float, optional
        end of the schedule, by default 0.02
    Returns
    -------
    _type_
        _description_
    """
    beta_schedule = np.linspace(beta_start, beta_end, timesteps)
    return beta_schedule


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def get_alphas_properties(betas: List[float]) -> AlphaVars:
    """
    Get all betas used in model diffusion
    """
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # For q(x_t | x_{t-1})
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # For posterior q(x_{t-1} | x_t, x_0)
    posterior_proba = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return AlphaVars(
        alphas,
        alphas_cumprod,
        alphas_cumprod_prev,
        sqrt_recip_alphas,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
        posterior_proba,
    )


def extract_alpha_from_timesteps(
    alpha_diffusion, timesteps: torch.Tensor, x_shape: List[int]
):
    """
    Extract alpha from timesteps for diffusion

    Parameters
    ----------
    alpha_diffusion : _type_
        Alpha from beta schedule
    timesteps : torch.Tensor
        Timesteps for creating the noise for diffusion
    x_shape : List[int]
        Shape of the image

    Returns
    -------
    _type_
        _description_
    """
    batch_size = timesteps.shape[0]
    # TODO: can be replaced to Tensor only??
    out = alpha_diffusion.gather(-1, timesteps.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(timesteps.device)


def q_sample(
    x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None
):
    """
    Forward diffusion with nice properties

    Parameters
    ----------
    x_start : _type_
        _description_
    t : _type_
        _description_
    sqrt_alphas_cumprod : _type_
        _description_
    sqrt_one_minus_alphas_cumprod : _type_
        _description_
    noise : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract_alpha_from_timesteps(
        sqrt_alphas_cumprod, t, x_start.shape
    )
    sqrt_one_minus_alphas_cumprod_t = extract_alpha_from_timesteps(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
