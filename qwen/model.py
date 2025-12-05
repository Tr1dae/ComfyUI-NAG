from functools import partial
from types import MethodType

import torch

from comfy.ldm.qwen_image.model import (
    QwenImageTransformer2DModel,
    QwenImageTransformerBlock,
)

from ..utils import nag, cat_context, check_nag_activation, NAGSwitch


def _nag_qwen_attn_forward(self, hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None,
                           attention_mask=None, image_rotary_emb=None, transformer_options={}):
    """
    Wraps Qwen Attention forward to inject NAG on the image stream.
    Assumes batches are ordered as [positive..., nag_negative...].
    """
    img_attn_output, txt_attn_output = self._forward_origin(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        attention_mask=attention_mask,
        image_rotary_emb=image_rotary_emb,
        transformer_options=transformer_options,
    )

    origin_bsz = img_attn_output.shape[0] // 2
    if origin_bsz == 0 or self.nag_scale <= 1:
        return img_attn_output, txt_attn_output

    img_pos, img_neg = img_attn_output[:origin_bsz], img_attn_output[origin_bsz:]
    img_guidance = nag(img_pos, img_neg, self.nag_scale, self.nag_tau, self.nag_alpha)
    img_attn_output = torch.cat([img_pos, img_guidance], dim=0)
    return img_attn_output, txt_attn_output


class NAGQwenImageTransformer2DModel(QwenImageTransformer2DModel):
    def forward_nag(
        self,
        x,
        timestep,
        context,
        attention_mask=None,
        guidance=None,
        ref_latents=None,
        transformer_options={},
        nag_negative_context=None,
        nag_sigma_end=0.0,
        **kwargs
    ):
        apply_nag = check_nag_activation(transformer_options, nag_sigma_end)
        if apply_nag:
            # duplicate latents so we have positive and nag_negative batches
            x = torch.cat([x, x], dim=0)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
            # append nag_negative context after positive context
            context = cat_context(context, nag_negative_context, trim_context=True)

        return self.forward_origin(
            x,
            timestep,
            context,
            attention_mask=attention_mask,
            guidance=guidance,
            ref_latents=ref_latents,
            transformer_options=transformer_options,
            **kwargs,
        )


class NAGQwenImageTransformer2DModelSwitch(NAGSwitch):
    def set_nag(self):
        # patch model forward
        self.model.forward_origin = self.model.forward
        self.model.forward = MethodType(
            partial(
                NAGQwenImageTransformer2DModel.forward_nag,
                nag_negative_context=self.nag_negative_cond[0][0],
                nag_sigma_end=self.nag_sigma_end,
            ),
            self.model,
        )

        # patch attentions inside each transformer block
        for block in self.model.transformer_blocks:
            if isinstance(block, QwenImageTransformerBlock):
                attn = block.attn
                if not hasattr(attn, "_forward_origin"):
                    attn._forward_origin = attn.forward
                attn.nag_scale = self.nag_scale
                attn.nag_tau = self.nag_tau
                attn.nag_alpha = self.nag_alpha
                attn.forward = MethodType(
                    partial(
                        _nag_qwen_attn_forward,
                    ),
                    attn,
                )

    def set_origin(self):
        super().set_origin()
        # restore attn forwards if patched
        for block in self.model.transformer_blocks:
            if isinstance(block, QwenImageTransformerBlock):
                attn = block.attn
                if hasattr(attn, "_forward_origin"):
                    attn.forward = attn._forward_origin
                    del attn._forward_origin

