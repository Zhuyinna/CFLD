import copy
from typing import Iterable, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import *
from diffusers.models.unet_2d_condition import *
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name






class MultiScaleReferenceImageEncoder(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        down_block_types: Tuple[str] = (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        # block_out_channels: Tuple[int] = (128, 256, 512, 512),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        num_layers: int = 1,
        norm_eps: float = 1e-5,
        use_time_emb: Optional[bool] = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)
        )
        self.down_blocks = nn.ModuleList([])

        # time
        if use_time_emb:
            timestep_input_dim = block_out_channels[0]
            time_embed_dim = block_out_channels[0] * 4
            self.time_proj = Timesteps(
                block_out_channels[0], flip_sin_to_cos, freq_shift
            )
            self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        else:
            time_embed_dim = None
            self.time_proj = None
            self.time_embedding = None

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            down_block = get_down_block(
                down_block_type,
                num_layers=num_layers,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=i > 0,
                resnet_eps=norm_eps,
                resnet_act_fn="silu",
                # attn_num_head_channels=1,
                attention_head_dim=1,
                resnet_groups=32,
                downsample_padding=1,
            )
            self.down_blocks.append(down_block)

    def forward(self, sample, timesteps=None):
        sample = self.conv_in(sample)
        samples = []
        if timesteps is not None and self.config.use_time_emb:
            # timesteps = timestep
            # timesteps has batchsize shape, so no need to expand
            timesteps = timesteps.expand(sample.shape[0])
            temb = self.time_proj(timesteps)
            temb = temb.to(dtype=self.dtype)
            temb = self.time_embedding(temb)
        else:
            temb = None
        for i, downsample_block in enumerate(self.down_blocks):
            sample, _ = downsample_block(hidden_states=sample, temb=temb)
            samples.append(sample)

        return samples
    

class EnhancedMultiScaleReferenceImageEncoder(MultiScaleReferenceImageEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, sample, timesteps=None):
        return super().forward(sample, timesteps=timesteps)
    


def build_rie(pretrained_path):
    rie = EnhancedMultiScaleReferenceImageEncoder.from_pretrained(pretrained_path) \
        if pretrained_path else EnhancedMultiScaleReferenceImageEncoder(
            use_time_emb=True,
            norm_eps=1e-8
        )
    return rie

def apply_mask(x, mask=None):
    # TODO: need fix
    if mask:
        embed_dim = x.shape[1]
        masked_embed = nn.Parameter(torch.zeros(1, embed_dim))

        if x.shape[-2:] != mask.shape[-2:]:
            htimes, wtimes = np.array(x.shape[-2:]) // np.array(mask.shape[-2:])
            mask = mask.repeat_interleave(htimes, -2).repeat_interleave(wtimes, -1)

        # mask embed
        x.permute(0, 2, 3, 1)[mask, :] = masked_embed.to(x.dtype)

    return x

