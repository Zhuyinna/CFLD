import copy
from typing import Iterable, Tuple, Union, Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import *
from diffusers.models.unet_2d_condition import *
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.util import log_txt_as_img, exists, instantiate_from_config

from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name




class ReferenceImageEncoder(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        down_block_types: Tuple[str] = (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        block_out_channels: Tuple[int] = (128, 256, 512),
        multi_scale: bool = False,
        use_cls_token: bool = False,
    ):
        super().__init__()
        self.multi_scale = multi_scale
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)
        )
        self.down_blocks = nn.ModuleList([])
        if self.multi_scale:
            self.channel_mappers = nn.ModuleList([])
            for i in range(len(block_out_channels) - 1):
                channel_mapper = nn.Conv2d(
                    block_out_channels[i], block_out_channels[-1], kernel_size=1
                )
                self.channel_mappers.append(channel_mapper)

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            down_block = get_down_block(
                down_block_type,
                num_layers=1,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,
                add_downsample=True,
                resnet_eps=1e-5,
                resnet_act_fn="silu",
                # attn_num_head_channels=8,
                attention_head_dim=8,
                resnet_groups=32,
                downsample_padding=1,
            )
            self.down_blocks.append(down_block)

        if use_cls_token:
            self.cls_token = nn.Parameter(th.zeros(1, 1, block_out_channels[-1]))
        else:
            self.cls_token = None

    @staticmethod
    def flatten(sample):
        batch, channel, height, width = sample.shape
        sample = sample.permute(0, 2, 3, 1).reshape(batch, height * width, channel)
        return sample

    def forward(self, sample):
        sample = self.conv_in(sample)
        print(f'sample shape: {sample.shape}')
        if self.multi_scale:
            res_samples = []
            for i, downsample_block in enumerate(self.down_blocks):
                sample, _ = downsample_block(hidden_states=sample, temb=None)
                print(f'sample[{i}] shape: {sample.shape}')
                res_sample = sample
                if i != len(self.down_blocks) - 1:
                    res_sample = self.channel_mappers[i](res_sample)
                    res_sample = self.flatten(res_sample)
                    print(f'res_sample[{i}] shape: {res_sample.shape}')
                else:
                    res_sample = self.flatten(res_sample)
                res_samples.append(res_sample)
            sample = th.cat(res_samples, dim=1)
        else:
            for downsample_block in self.down_blocks:
                sample, _ = downsample_block(hidden_states=sample, temb=None)
            sample = self.flatten(sample)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(sample.shape[0], -1, -1)
            sample = th.cat([cls_token, sample], dim=1)
        return sample


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




class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        # if context_dim is not None:
        #     assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
        #     from omegaconf.listconfig import ListConfig
        #     if type(context_dim) == ListConfig:
        #         context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        # x:noisy latents, hint:cond_img, context:clipped_cond_img
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        hint = self.upsample(hint)
        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs
    


# class EnhancedMultiScaleReferenceImageEncoder(MultiScaleReferenceImageEncoder):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#     def forward(self, sample, timesteps=None):
#         return super().forward(sample, timesteps=timesteps)


class AppearanceControlNet(ControlNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x, hint, timesteps, context, **kwargs):
        return super().forward(x, hint, timesteps, context, **kwargs)

class EnhancedControlNet(ConfigMixin,ModelMixin):

    # @register_to_config
    def __init__(
        self,
        # Include only the necessary arguments
        in_channels: int = 4,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        layers_per_block: Union[int, Tuple[int]] = 2,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        # new add
        use_spatial_transformer=False,  # custom transformer support
        context_dim=None,  # custom transformer support
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        class_embeddings_concat: bool = False,
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        mid_block_scale_factor: float = 1,
        dims=2,
        hint_channels=3,
        model_channels=320,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'
        
        
        # time
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        ) 

        
        self.down_blocks = nn.ModuleList([])
        self.mid_block = None

        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention

            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False


        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)
    
        if class_embeddings_concat:
            # The time embeddings are concatenated with the class embeddings. The dimension of the
            # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
            # regular time embeddings
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        # down
        
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,  # Define or adjust this based on your config
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                only_cross_attention=only_cross_attention[i],  # Adjust according to your config
            )
            self.down_blocks.append(down_block)

        # Mid block setup
        if mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn(
                transformer_layers_per_block=transformer_layers_per_block[-1],
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,  # Define or adjust this based on your config
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,  # Define or adjust this based on your config
            )
        elif mid_block_type == "UNetMidBlock2DSimpleCrossAttn":
            self.mid_block = UNetMidBlock2DSimpleCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,  # Define or adjust this based on your config
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,  # Define or adjust this based on your config
            )
        elif mid_block_type is None:
            self.mid_block = None
        else:
            raise ValueError(f"Unknown mid_block_type: {mid_block_type}")
        
        self.mid_block_out = self.make_zero_conv(block_out_channels[-1])

        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )
    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, 
                x, 
                hint, 
                timesteps, 
                context,
                timestep_cond: Optional[th.Tensor] = None, 
                **kwargs):
        # x:noisy latents, hint:cond_img, context:clipped_cond_img

        # 1. time
        if not th.is_tensor(timesteps):
            is_mps = x.device.type == "mps"
            if isinstance(th, float):
                dtype = th.float32 if is_mps else th.float64
            else:
                dtype = th.int32 if is_mps else th.int64
            timesteps = th.tensor([timesteps], dtype=dtype, device=x.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)
        timesteps = timesteps.expand(x.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. pre-process hint
        hint = self.upsample(hint)
        guided_hint = self.input_hint_block(hint, emb, context)

        # 3. down

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.down_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        # 4. mid
        h = self.mid_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


# def build_rie(pretrained_path):
#     # TODO: 初始化传入参数

#     rie = EnhancedControlNet.from_pretrained(pretrained_path) \
#         if pretrained_path else EnhancedControlNet(
#             in_channels: int = 4,
#             block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
#             down_block_types: Tuple[str] = (
#                 "CrossAttnDownBlock2D",
#                 "CrossAttnDownBlock2D",
#                 "CrossAttnDownBlock2D",
#                 "DownBlock2D",
#             ),
#             mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
#             layers_per_block: Union[int, Tuple[int]] = 2,
#             act_fn: str = "silu",
#             norm_num_groups: Optional[int] = 32,
#             norm_eps: float = 1e-5,
#             cross_attention_dim: Union[int, Tuple[int]] = 1280,
#             use_spatial_transformer=False,  # custom transformer support
#             context_dim=None,  # custom transformer support
#             time_embedding_type: str = "positional",
#             time_embedding_dim: Optional[int] = None,
#             flip_sin_to_cos: bool = True,
#             freq_shift: int = 0,
#             timestep_post_act: Optional[str] = None,
#             time_cond_proj_dim: Optional[int] = None,
#             class_embeddings_concat: bool = False,
#             only_cross_attention: Union[bool, Tuple[bool]] = False,
#             mid_block_only_cross_attention: Optional[bool] = None,
#             num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
#             attention_head_dim: Union[int, Tuple[int]] = 8,
#             transformer_layers_per_block: Union[int, Tuple[int]] = 1,
#             mid_block_scale_factor: float = 1,
#             dims=2,
#             hint_channels=3,
#             model_channels=320,
#         )
#     return rie


# def build_rie(block_out_channels, pretrained_path):
#     rie = ReferenceImageEncoder.from_pretrained(pretrained_path) \
#         if pretrained_path else ReferenceImageEncoder(
#             block_out_channels=block_out_channels,
#             multi_scale=True,
#         )
#     return rie

def apply_mask(x, mask=None):
    # TODO: need fix
    if mask:
        embed_dim = x.shape[1]
        masked_embed = nn.Parameter(th.zeros(1, embed_dim))

        if x.shape[-2:] != mask.shape[-2:]:
            htimes, wtimes = np.array(x.shape[-2:]) // np.array(mask.shape[-2:])
            mask = mask.repeat_interleave(htimes, -2).repeat_interleave(wtimes, -1)

        # mask embed
        x.permute(0, 2, 3, 1)[mask, :] = masked_embed.to(x.dtype)

    return x

