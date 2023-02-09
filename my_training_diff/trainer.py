from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim import Adam

from my_training_diff.diffusiion_utils import (
    cosine_beta_schedule,
    get_alphas_properties,
    q_sample,
)
from my_training_diff.inference import sample
from my_training_diff.model_main import Unet


class LitModelDiffusionV1(pl.LightningModule):
    def __init__(
        self,
        init_dim: int = 64,
        out_dim: int = 64,
        num_img_channel: int = 3,
        resnet_blocks_num: int = 4,
        dim_mults: List[int] = [1, 2, 4],
        timesteps: int = 150,
        learning_rate: float = 1e-3,
        num_eval_samples: int = 4,
    ) -> None:
        """
        Unet with Resnet Blocks

        Parameters
        ----------
        init_dim : int
            Initial image dimension (image size: dim x dim)
        out_dim : int
            Output image dimension (image size: dim x dim)
        num_img_channel : int, optional
            Number of image channels (RGB), by default 3
        resnet_blocks_num : int, optional
            Resnet block quantity that will be formed, by default 4
        dim_mults : List[int], optional
            Dimension for downsampling and upsampling in UNET multiplier
            from the initial dimension, by default [1, 2, 4]
        timesteps: int
            Number of timesteps for diffusion
        learning_rate: float
            Learning rate for Adam optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        self.model_diff = Unet(
            init_dim, out_dim, num_img_channel, resnet_blocks_num, dim_mults
        )
        self.timesteps = timesteps

        # Diffusion Beta Schedule
        self.beta_schedule = cosine_beta_schedule(self.timesteps)
        self.alpha_vars = get_alphas_properties(self.beta_schedule)

    def forward(self, img_noise, timestep):
        return self.model_diff(img_noise, timestep)

    def training_step(self, batch, batch_idx):
        """
        Current training step:
        1. Create a random noise and sample timestep (uniform)
        2. Produce noise with scheduled beta noise using nice property
        3. Predict noise from model
        4. Calculate loss using huber loss

        Parameters
        ----------
        batch : torch.Tensor
            Batch of images, shape: (batch_size, num_img_channel, dim, dim)
        """
        batch_size = batch.shape[0]
        noise = torch.randn_like(batch, device=self.device)
        timesteps_batch_sample = torch.randint(
            0, self.timesteps, (batch_size,), device=self.device
        ).long()

        noise_from_beta = q_sample(
            batch,
            timesteps_batch_sample,
            self.alpha_vars.sqrt_alphas_cumprod,
            self.alpha_vars.sqrt_one_minus_alphas_cumprod,
            noise,
        )

        predicted_noise = self(noise_from_beta, timesteps_batch_sample)

        loss = F.smooth_l1_loss(noise_from_beta, predicted_noise)
        return loss

    def on_validation_epoch_end(self):
        """
        Log generated image
        """
        imgs = sample(
            self.model_diff,
            self.hparams.init_dim,
            num_eval_samples=self.hparams.num_eval_samples,
        )
        imgs = (imgs + 1) * 0.5
        grid = torchvision.utils.make_grid(imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

    def configure_optimizers(self):
        return Adam(self.model_diff.parameters(), lr=self.hparams.learning_rate)
