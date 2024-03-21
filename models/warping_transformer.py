import torch.nn as nn
from einops import rearrange

from ldm.modules.diffusionmodules.util import (
    zero_module,
)

from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.util import exists
from ldm.modules.attention import Normalize, CrossAttentionWithMask, MemoryEfficientCrossAttentionWithMask, XFORMERS_IS_AVAILBLE, FeedForward

class CustomBasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttentionWithMask,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttentionWithMask
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False,use_loss=True):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.use_loss = use_loss

    def forward(
            self, 
            x, 
            context=None, 
            mask=None, 
            mask1=None, 
            mask2=None, 
            use_attention_mask=False,
            use_attention_tv_loss=False,
            tv_loss_type=None,
        ):
        if not (use_attention_tv_loss or use_attention_mask):
            x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask) + x
            x = self.attn2(self.norm2(x), context=context, mask=mask) + x
            x = self.ff(self.norm3(x)) + x
            return x
        elif use_attention_mask:
            x1 = self.attn1(
                self.norm1(x), 
                context=context if self.disable_self_attn else None, 
                mask=mask, 
                mask1=mask1, 
                mask2=mask2, 
                use_attention_tv_loss=False,
            )
            x = x1 + x
            x2 = self.attn2(  # cross attention
                self.norm2(x), 
                context=context,
                mask=mask,
                mask1=mask1, 
                mask2=mask2, 
                use_attention_tv_loss=False,
            )
            x = x2 + x
            x = self.ff(self.norm3(x)) + x
            return x
        else:
            x1, loss1 = self.attn1(
                self.norm1(x), 
                context=context if self.disable_self_attn else None, 
                mask=mask, 
                mask1=mask1, 
                mask2=mask2, 
                use_attention_tv_loss=use_attention_tv_loss,
                tv_loss_type=tv_loss_type,
            )
            x = x1 + x
            x2, loss2 = self.attn2(
                self.norm2(x), 
                context=context,
                mask=mask,
                mask1=mask1, 
                mask2=mask2, 
                use_attention_tv_loss=use_attention_tv_loss,
                use_loss=self.use_loss,
                tv_loss_type=tv_loss_type,
            )
            x = x2 + x
            x = self.ff(self.norm3(x)) + x
            loss = loss1 + loss2
            return x, loss
        
class CustomSpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True,use_loss=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                CustomBasicTransformerBlock(
                inner_dim, 
                n_heads, 
                d_head, 
                dropout=dropout, 
                context_dim=context_dim[d],
                disable_self_attn=disable_self_attn, 
                checkpoint=use_checkpoint, use_loss=use_loss) for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear
        self.use_loss = use_loss
    def forward(
            self, 
            x, 
            context=None, 
            mask=None, 
            mask1=None, 
            mask2=None, 
            use_attention_mask=False,
            use_attention_tv_loss=False,
            tv_loss_type=None,
    ):

        # note: if no context is given, cross-attention defaults to self-attention
        loss = 0
        if not isinstance(context, list):
            context = [context]  # 如果depth>1，可能输入多个context
        # new add

        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        # depth个self.transformer_blocks
        for i, block in enumerate(self.transformer_blocks):
            if not (use_attention_tv_loss or use_attention_mask):
                x = block(x, context=context[i], mask=mask)
            elif use_attention_mask:
                x = block(
                    x,
                    context=context[i],
                    mask=mask, 
                    mask1=mask1, 
                    mask2=mask2, 
                    use_attention_mask=True,
                    use_attention_tv_loss=False,
                    use_center_loss=False,
                )
            else:
                x, attn_loss = block(
                    x,
                    context=context[i],
                    mask=mask, 
                    mask1=mask1, 
                    mask2=mask2, 
                    use_attention_mask=use_attention_mask,
                    use_attention_tv_loss=use_attention_tv_loss,
                    tv_loss_type=tv_loss_type,
                )
                loss += attn_loss
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        if not (use_attention_tv_loss):
            return x + x_in
        else:
            return x + x_in, loss