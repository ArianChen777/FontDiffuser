"""FreeU adaptation for FontDiffuser.

Reference: https://github.com/ChenyangSi/FreeU
Paper: "FreeU: Free Lunch in Diffusion U-Net" (Si et al., 2023)

FontDiffuser UNet architecture (channels = 64, 128, 256, 512):
  Down path: DownBlock2D → MCADownBlock2D → MCADownBlock2D → DownBlock2D
  Up   path: UpBlock2D   → StyleRSIUpBlock2D → StyleRSIUpBlock2D → UpBlock2D

FreeU is applied only to the two up_blocks closest to the bottleneck:
  - up_block[0] : UpBlock2D        (backbone hidden = 512 ch)
  - up_block[1] : StyleRSIUpBlock2D (backbone hidden = 512 → 256 → 256 ch)

Both the original UpBlock2D forward and the StyleRSI forward are monkey-patched
at the instance level so the rest of the model is untouched.

Design decisions vs. naive port:
  1. Backbone scaling is applied ONCE per block (before the resnet loop),
     not once per resnet iteration, to avoid cumulative b^N amplification.
  2. For StyleRSIUpBlock2D, Fourier_filter is applied AFTER DCN alignment
     so the structural offset computation sees unfiltered skip features.

Experiment sweep (fix s=0.9, threshold=1*, vary b):
  b = 1.0 → 1.2 → 1.4 → 1.6

* The user specified threshold=0.5; since Fourier_filter uses integer pixel
  indexing (crow ± threshold), 0.5 would truncate to 0 (no-op).
  We therefore enforce a minimum of 1 pixel, which is the standard FreeU value.
"""

import types

import torch
import torch.fft as fft


# ---------------------------------------------------------------------------
# Core Fourier filter
# ---------------------------------------------------------------------------

def Fourier_filter(x, threshold, scale):
    """Low-pass scale on skip-connection features via FFT.

    The center (low-frequency) region of the 2-D frequency spectrum is
    multiplied by `scale`.  High-frequency components stay at scale 1.0,
    so setting scale < 1 attenuates low frequencies (dampens texture/style
    leakage through skip connections while preserving structural edges).

    Args:
        x         (Tensor): [B, C, H, W] float or half tensor.
        threshold (float) : Half-width of the low-freq region in pixels.
                            Rounded up to int ≥ 1.
        scale     (float) : Multiplier for the low-freq region (e.g. 0.9).

    Returns:
        Tensor same shape and dtype as `x`.
    """
    dtype = x.dtype
    x = x.float()

    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device, dtype=torch.float32)

    crow, ccol = H // 2, W // 2
    t = max(1, int(threshold))          # enforce ≥ 1 pixel
    mask[..., crow - t:crow + t, ccol - t:ccol + t] = scale

    x_freq = x_freq * mask

    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(dtype)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_freeu(model, b=1.2, s=0.9, threshold=1):
    """Patch FontDiffuser's UNet in-place with FreeU at inference time.

    Call this once after loading the model and before running pipe.generate().
    Only up_block[0] and up_block[1] are patched; the rest of the UNet is
    unchanged.

    Args:
        model     : FontDiffuserModelDPM (has a .unet attribute).
        b (float) : Backbone amplification factor.  Try 1.0, 1.2, 1.4, 1.6.
        s (float) : Low-frequency scale for skip connections. Fixed at 0.9.
        threshold : FFT center half-width in pixels. Fixed at 1.
    """
    _patch_upblock(model.unet.up_blocks[0], b=b, s=s, threshold=threshold)
    _patch_stylersI_upblock(model.unet.up_blocks[1], b=b, s=s, threshold=threshold)
    print(f"[FreeU] Applied to up_block[0] (UpBlock2D) and "
          f"up_block[1] (StyleRSIUpBlock2D)  |  b={b}  s={s}  threshold={threshold}")


# ---------------------------------------------------------------------------
# Private patchers
# ---------------------------------------------------------------------------

def _scale_backbone(hidden_states, b):
    """Scale the first half of backbone channels by b (no in-place mutation)."""
    half = hidden_states.shape[1] // 2
    return torch.cat([hidden_states[:, :half] * b,
                      hidden_states[:, half:]], dim=1)


def _patch_upblock(block, b, s, threshold):
    """Monkey-patch UpBlock2D instance with FreeU.

    Backbone scaling is applied ONCE before the resnet loop (not per
    iteration) to avoid cumulative b^N amplification across the 3 resnets.
    Each skip connection is Fourier-filtered individually inside the loop.
    """
    block.freeu_b = b
    block.freeu_s = s
    block.freeu_threshold = threshold

    def forward(self, hidden_states, res_hidden_states_tuple,
                temb=None, upsample_size=None):
        # ── FreeU: backbone scaling applied ONCE per block ─────────────────
        hidden_states = _scale_backbone(hidden_states, self.freeu_b)
        # ───────────────────────────────────────────────────────────────────

        for resnet in self.resnets:
            # pop skip connection
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # ── FreeU: filter each skip connection individually ─────────────
            res_hidden_states = Fourier_filter(
                res_hidden_states, self.freeu_threshold, self.freeu_s)
            # ───────────────────────────────────────────────────────────────

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states

    block.forward = types.MethodType(forward, block)


def _patch_stylersI_upblock(block, b, s, threshold):
    """Monkey-patch StyleRSIUpBlock2D instance with FreeU.

    Backbone scaling is applied ONCE before the resnet loop.

    Fourier_filter is applied AFTER DCN alignment (not before), so that
    the DCN's offset computation uses the original unfiltered skip features
    for correct structural alignment. The FreeU filter then attenuates
    high-frequency noise in the already-aligned skip before concatenation.
    """
    block.freeu_b = b
    block.freeu_s = s
    block.freeu_threshold = threshold

    def forward(self, hidden_states, res_hidden_states_tuple,
                style_structure_features, temb=None,
                encoder_hidden_states=None, upsample_size=None):

        total_offset = 0
        style_content_feat = style_structure_features[-self.upblock_index - 2]

        # ── FreeU: backbone scaling applied ONCE per block ─────────────────
        hidden_states = _scale_backbone(hidden_states, self.freeu_b)
        # ───────────────────────────────────────────────────────────────────

        for sc_inter_offset, dcn_deform, resnet, attn in zip(
            self.sc_interpreter_offsets, self.dcn_deforms,
            self.resnets, self.attentions
        ):
            # pop skip connection
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # Original StyleRSI: DCN spatial alignment on unfiltered skip
            offset = sc_inter_offset(res_hidden_states, style_content_feat)
            offset = offset.contiguous()
            offset_sum = torch.mean(torch.abs(offset))
            total_offset += offset_sum

            res_hidden_states = res_hidden_states.contiguous()
            res_hidden_states = dcn_deform(res_hidden_states, offset)

            # ── FreeU: filter AFTER DCN so structural offsets are unaffected
            res_hidden_states = Fourier_filter(
                res_hidden_states, self.freeu_threshold, self.freeu_s)
            # ───────────────────────────────────────────────────────────────

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn), hidden_states,
                    encoder_hidden_states)
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states,
                                     context=encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        offset_out = total_offset / self.num_layers
        return hidden_states, offset_out

    block.forward = types.MethodType(forward, block)
