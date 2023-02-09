from torch import nn
from my_training_diff.model_helper import (
    ResnetBlock,
    SinusoidalPositionEmbeddings,
    LinearAttention,
    PreNorm,
    Residual,
    Downsample,
    Upsample,
)
import torch
from typing import List


class Unet(nn.Module):
    def __init__(
        self,
        init_dim: int,
        out_dim: int,
        num_img_channel: int = 3,
        resnet_blocks_num: int = 4,
        dim_mults: List[int] = [1, 2, 4],
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
        """
        super().__init__()
        self.num_img_channel = num_img_channel
        self.init_dim = init_dim
        self.out_dim = out_dim
        self.init_conv = nn.Conv2d(self.num_img_channel, init_dim, 1, padding=0)

        # e.g.: [64, 128, 256, 512], used for upsampling and downsampling
        dims = [init_dim * dim_mult for dim_mult in dim_mults]
        in_out = list(
            zip(dims[:-1], dims[1:])
        )  # e.g.: [(64, 128), (128, 256), (256, 512)]
        self.in_out = in_out
        # Blocks

        ## Time Embedding
        time_dim = init_dim * 4  # Idk why it's multiplied by 4...
        self.time_mlp_blocks = nn.Sequential(
            SinusoidalPositionEmbeddings(init_dim),
            nn.Linear(init_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        ## Downsampling Blocks
        ## Low -> High. e.g.: 64 -> 128 -> 256

        self.downsampling_module_list = nn.ModuleList([])

        num_resolutions = len(in_out)
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx == (num_resolutions - 1)

            self.downsampling_module_list.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_in,
                            dim_in,
                            time_emb_dim=time_dim,
                            groups=resnet_blocks_num,
                        ),
                        ResnetBlock(
                            dim_in,
                            dim_in,
                            time_emb_dim=time_dim,
                            groups=resnet_blocks_num,
                        ),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        ## Middle Block
        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_blocks_num
        )
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_blocks_num
        )

        ## Upsampling Blocks
        ## High -> Low. e.g.: 256 -> 128 -> 64
        self.upsampling_module_list = nn.ModuleList([])

        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx == (num_resolutions - 1)
            self.upsampling_module_list.append(
                nn.ModuleList(
                    [
                        # Skip connection
                        ResnetBlock(
                            dim_in + dim_out,
                            dim_out,
                            time_emb_dim=time_dim,
                            groups=resnet_blocks_num,
                        ),
                        ResnetBlock(
                            dim_in + dim_out,
                            dim_out,
                            time_emb_dim=time_dim,
                            groups=resnet_blocks_num,
                        ),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        # Final ResNet Block
        self.final_block = ResnetBlock(
            init_dim * 2, init_dim, time_emb_dim=time_dim, groups=resnet_blocks_num
        )
        self.final_conv = nn.Conv2d(init_dim, out_dim, 1, padding=0)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x : torch.Tensor
            Image with noise
        timestep : torch.Tensor
            What time step it is. e.g. [3]

        Returns
        -------
        torch.Tensor
            Return the predicted noise
        """
        x = self.init_conv(x)  # (batch_size, init_dim, img_size, img_size)
        skip_x = x.clone()  # Will be used for the last layer

        # get time out
        time_out = self.time_mlp_blocks(timestep)

        skip_down_x = []
        # DOWNSAMPLING TIME
        for block1, block2, attention, downsample in self.downsampling_module_list:
            x = block1(x, time_out)
            skip_down_x.append(x)

            x = block2(x, time_out)

            x = attention(x)
            skip_down_x.append(x)

            x = downsample(x)

        # Middle Block
        x = self.mid_block1(x, time_out)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_out)

        # UPSAMPLING TIME
        for block1, block2, attention, upsample in self.upsampling_module_list:
            x = torch.cat([x, skip_down_x.pop()], dim=1)
            x = block1(x, time_out)

            x = torch.cat([x, skip_down_x.pop()], dim=1)
            x = block2(x, time_out)
            x = attention(x)
            x = upsample(x)

        x = torch.cat([x, skip_x], dim=1)
        x = self.final_block(x, time_out)
        x = self.final_conv(x)

        return x
