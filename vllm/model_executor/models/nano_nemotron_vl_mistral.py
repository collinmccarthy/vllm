# SPDX-License-Identifier: Apache-2.0
"""Mistral-Large-3 / Pixtral-Large vision encoder support for
`NemotronH_Nano_VL_V2`.

This module is imported lazily by `nano_nemotron_vl.py` only when the
checkpoint's `config.json` has `vision_model_type = "pixtral-vit-large"`.
RADIO checkpoints don't touch this file at all. The split keeps the
Pixtral-specific subclassing, the dynamic 2D RoPE module, the FA-varlen
mask metadata, and the encoder→merger adapter out of the central
nano_nemotron_vl.py file — the only Pixtral footprint left there is the
small per-call dispatches on `_encoder_did_spatial_merge` inside
`__init__` / `extract_feature*` / `load_weights`.

Public entry point: `build_pixtral_encoder(hf_config, *, prefix)` → returns
a `PixtralEncoderAdapter` `nn.Module` whose `(summary, all_feat)` forward
signature matches RADIO's, so the call sites in `nano_nemotron_vl.py` can
treat both encoders interchangeably.

# =============================================================================
# Pixtral path — design notes
#
# We subclass vLLM's **Mistral-format** Pixtral encoder family
# (`VisionTransformer` / `Transformer` / `TransformerBlock` / `Attention` /
# `apply_rotary_emb_vit`, all in `vllm/model_executor/models/pixtral.py`).
# That's the path `PixtralForConditionalGeneration` uses to serve
# Mistral-Large-3-675B-Base-2512 — raw-Mistral checkpoint format, 2D RoPE
# applied via `torch.view_as_complex` (interleaved pairing of adjacent
# dims). Mcore's `Pixtral2DRotaryEmbedding` (vit_model.py) preserves this
# convention via `rotary_interleaved=True` + `repeat_interleave(2)`, and our
# mcore→HF saver does only a head-grouping reorder (no within-head-dim
# permutation), so the weights land in HF-named files but still in
# Mistral-original head-dim layout. Pairing them with HF's rotate_half
# `PixtralHFVisionModel` would silently scramble RoPE numerics; pairing
# them with `VisionTransformer` matches end-to-end.
#
# Why we still subclass (rather than use `VisionTransformer` straight):
#   1. **FA-varlen attention** — parent's `Attention.forward` uses xformers
#      `BlockDiagonalMask` or HF's dense `(T,T)` fallback. We replace the
#      attention call with `MMEncoderAttention` over `cu_seqlens`/
#      `max_seqlen`, matching how RADIO is wired in vLLM. Avoids the
#      profile-time dense-mask OOM (no xformers in the container).
#   2. **Dynamic RoPE** — parent's `VisionTransformer` precomputes
#      `freqs_cis` at `max_patches_per_side = image_size / patch_size`,
#      which caps spatial dims. Dynamic-res images go past that bound at
#      runtime; mcore's `Pixtral2DRotaryEmbedding` composes freqs per call
#      to avoid the cap. `_PixtralDynamicRotaryEmbedding` mirrors mcore.
#   3. **Block / video frame chunking** — see `PixtralEncoderAdapter`
#      below for the 4-D path's 32-frame micro-batching.
#
# Open validation items (numeric feature diff vs mcore is the decisive
# test — see ~/.claude/plans/vision-encoder-study-parity-test.md):
#   * Per-head QKV layout vs `qkv_proj.chunk(3, dim=-1)` — confirm
#     `[Q_all, K_all, V_all]` from saver lines up correctly post-chunk.
#   * MLP fusion order: saver's `cat([W, V])` ↔ `gate_up_proj`'s
#     `MergedColumnParallelLinear(output_sizes=[I, I])` expects
#     `[gate, up]`. Verify W==gate==w1, V==up==w3.
#   * Spot-check: feed a fixed image through mcore Pixtral encoder vs
#     our adapter, diff pre-merger features.
# =============================================================================
"""

from typing import NamedTuple

import torch
import torch.nn as nn
from transformers import PixtralVisionConfig

from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.pixtral import (
    Attention as _PixtralAttention,
    PatchMerger as _PixtralNativePatchMerger,
    Transformer as _PixtralTransformer,
    TransformerBlock as _PixtralTransformerBlock,
    VisionEncoderArgs,
    VisionTransformer as _PixtralVisionTransformer,
    apply_rotary_emb_vit,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel


class _PixtralVarlenMaskMeta(NamedTuple):
    """Variable-length attention metadata threaded through the Pixtral
    encoder in place of the dense `(T, T)` attention mask.

    `cu_seqlens` is the FA-style cumulative per-image patch-count tensor
    (shape `(N+1,)`, int32, ends at total). `max_seqlen` is a scalar
    tensor — vLLM's FA wrapper requires `Tensor`, not `int`. Passed
    opaquely through TransformerBlock/Transformer (no isinstance check
    needed there) and dispatched on in `_PixtralVarlenAttention.forward`.
    """

    cu_seqlens: torch.Tensor
    max_seqlen: torch.Tensor


class _PixtralVarlenAttention(_PixtralAttention):
    """Mistral-format Pixtral attention with **flash-attn varlen** in place of
    the parent's xformers `BlockDiagonalMask` / HF dense `(T, T)` fallback.
    Receives `_PixtralVarlenMaskMeta` (`cu_seqlens`, `max_seqlen`) and routes
    to `MMEncoderAttention`. RoPE convention (view_as_complex / interleaved)
    is unchanged from the parent.
    """

    def __init__(
        self,
        args: VisionEncoderArgs,
        quant_config=None,
        *,
        prefix: str = "",
        disable_tp: bool = False,
    ) -> None:
        super().__init__(
            args,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=disable_tp,
        )
        self.varlen_attn = MMEncoderAttention(
            num_heads=self.n_heads,
            head_size=self.head_dim,
            prefix=f"{prefix}.varlen_attn",
        )

    def forward(
        self,
        x: torch.Tensor,
        mask,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        assert isinstance(mask, _PixtralVarlenMaskMeta), (
            "_PixtralVarlenAttention expects `_PixtralVarlenMaskMeta`; got "
            f"{type(mask).__name__}."
        )

        batch, patches, _ = x.shape
        assert batch == 1, (
            "_PixtralVarlenAttention assumes the encoder packs all patches "
            "into a single (1, total_patches, H) tensor; got batch="
            f"{batch}."
        )

        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch, patches, self.n_heads, self.head_dim)
        k = k.reshape(batch, patches, self.n_heads, self.head_dim)
        v = v.reshape(batch, patches, self.n_heads, self.head_dim)

        # Same `view_as_complex` rotation as the parent — pairs adjacent
        # dims (2i, 2i+1), matching mcore's `Pixtral2DRotaryEmbedding`.
        q, k = apply_rotary_emb_vit(q, k, freqs_cis=freqs_cis)

        # MMEncoderAttention auto-handles 4-D `(B, T, H, D)` inputs and
        # returns 4-D output of the same shape.
        out = self.varlen_attn(
            q,
            k,
            v,
            cu_seqlens=mask.cu_seqlens,
            max_seqlen=mask.max_seqlen,
        )  # (1, T, n_heads, head_dim)

        out = out.reshape(batch, patches, self.n_heads * self.head_dim)
        out, _ = self.o_proj(out)
        return out


class _PixtralVarlenTransformerBlock(_PixtralTransformerBlock):
    """Replace the parent's `Attention` instance with the varlen subclass.
    Weight-loading paths (`attention.qkv_proj.weight`, `attention.o_proj.weight`)
    are unchanged because the attribute name stays `attention`.
    """

    def __init__(
        self,
        args: VisionEncoderArgs,
        quant_config=None,
        *,
        prefix: str = "",
        disable_tp: bool = False,
    ) -> None:
        super().__init__(
            args,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=disable_tp,
        )
        self.attention = _PixtralVarlenAttention(
            args,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
            disable_tp=disable_tp,
        )


class _PixtralVarlenTransformer(_PixtralTransformer):
    """Use the varlen TransformerBlock instead of the default."""

    def __init__(
        self,
        args: VisionEncoderArgs,
        quant_config=None,
        *,
        prefix: str = "",
        disable_tp: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.layers = nn.ModuleList(
            [
                _PixtralVarlenTransformerBlock(
                    args,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                    disable_tp=disable_tp,
                )
                for i in range(args.num_hidden_layers)
            ]
        )


class _PixtralDynamicRotaryEmbedding(nn.Module):
    """2D RoPE for Pixtral in the **interleaved / view_as_complex** convention
    (the one Mistral-Large-3 was trained with and that mcore preserves), with
    NO precomputed max-bounded position table.

    Why complex / interleaved rather than HF rotate_half:
        Mistral-Large-3-675B-Base-2512 is distributed as a **raw Mistral
        checkpoint** (`consolidated-*.safetensors` + `params.json`). vLLM
        serves it via `PixtralForConditionalGeneration` → `VisionTransformer`
        → `apply_rotary_emb_vit` (`torch.view_as_complex`), pairing **adjacent
        dims** `(2i, 2i+1)`. Mcore's `Pixtral2DRotaryEmbedding` matches that
        with `rotary_interleaved=True` + `repeat_interleave(2)`, so mcore-
        trained weights expect this rotation. Our mcore→HF saver does only a
        head-grouping reorder (no within-head-dim permutation), so the weights
        landing in HF format are still in **interleaved-compatible head-dim
        layout**. We therefore apply the *same* `view_as_complex` rotation at
        inference rather than HF's `rotate_half` path.

    Why dynamic (no max-bounded table):
        The trained model is dynamic-res — single-image patch grids reach
        ~115×115 (square at max_num_patches=13312), well past any reasonable
        `max_patches_per_side` precompute. Mcore's `Pixtral2DRotaryEmbedding`
        composes freqs per call via `torch.outer(arange(h), inv_freq_h)` etc;
        we mirror that here.

    Output layout: `freqs_cis` (complex64) of shape `(total_patches, dim/2)`,
    consumable directly by vLLM's `apply_rotary_emb_vit` (after its internal
    `_reshape_for_broadcast`).
    """

    def __init__(self, args: VisionEncoderArgs) -> None:
        super().__init__()
        self.head_dim = args.hidden_size // args.num_attention_heads
        # Base inverse frequencies, then split into H and W halves matching
        # Mistral / mcore's freqs[::2] / freqs[1::2] convention.
        freqs = 1.0 / (
            args.rope_theta
            ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )  # (head_dim // 2,)
        self.register_buffer(
            "inv_freq_h", freqs[::2].contiguous(), persistent=False
        )
        self.register_buffer(
            "inv_freq_w", freqs[1::2].contiguous(), persistent=False
        )

    def forward(
        self,
        x: torch.Tensor,
        patch_embeds_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compose `freqs_cis` for the concatenated per-image patch stream.

        Args:
            x: dummy reference for device (dtype must remain complex64 —
                `apply_rotary_emb_vit` asserts this).
            patch_embeds_list: list of post-`patch_conv` per-image tensors
                of shape `(1, hidden, h, w)`; each `(h, w)` gives that
                image's actual patch-grid size.

        Returns:
            freqs_cis: complex64 tensor of shape `(total_patches, dim/2)`,
                ready for `apply_rotary_emb_vit`.
        """
        device = x.device
        inv_freq_h = self.inv_freq_h.to(device=device)
        inv_freq_w = self.inv_freq_w.to(device=device)

        per_image_freqs: list[torch.Tensor] = []
        for patch in patch_embeds_list:
            h, w = patch.shape[-2:]
            h_idx = torch.arange(h, device=device, dtype=torch.float32)
            w_idx = torch.arange(w, device=device, dtype=torch.float32)
            fh = torch.outer(h_idx, inv_freq_h)  # (h, dim // 4)
            fw = torch.outer(w_idx, inv_freq_w)  # (w, dim // 4)
            # Compose raster-order (h, w) freq vectors:
            #   pos (i, j) → cat([fh[i], fw[j]])  (length dim // 2)
            # Matches `precompute_freqs_cis_2d` in vllm/.../pixtral.py.
            fh_2d = fh[:, None, :].expand(h, w, -1)
            fw_2d = fw[None, :, :].expand(h, w, -1)
            freqs_2d = torch.cat([fh_2d, fw_2d], dim=-1).reshape(
                h * w, -1
            )  # (h*w, dim // 2)
            per_image_freqs.append(freqs_2d)

        freqs_2d_all = torch.cat(per_image_freqs, dim=0)  # (T, dim//2)
        # `polar(ones, freqs)` → complex `cos(freqs) + i*sin(freqs)`; this is
        # what `apply_rotary_emb_vit` consumes via its `view_as_complex`
        # multiplication path. dtype must be complex64 (asserted there).
        freqs_cis = torch.polar(
            torch.ones_like(freqs_2d_all), freqs_2d_all
        ).to(torch.complex64)
        return freqs_cis


class _PixtralVarlenVisionModel(_PixtralVisionTransformer):
    """Mistral-format Pixtral vision encoder (the one
    `PixtralForConditionalGeneration` instantiates for Mistral-Large-3) with
    two surgical overrides:

    1. **Dynamic 2D RoPE** — parent's `freqs_cis` property precomputes a
       `(max_patches_per_side, max_patches_per_side, dim/2)` table from
       `args.image_size / args.patch_size`. The trained model is
       dynamic-res; per-image patch grids exceed any reasonable bound. We
       replace it with `_PixtralDynamicRotaryEmbedding`, which composes
       freqs per call from each image's actual (h, w) — same pattern as
       mcore's `Pixtral2DRotaryEmbedding`.
    2. **Flash-attn varlen attention** — parent's `Attention` uses xformers
       `BlockDiagonalMask` (when available) or HF's dense `generate_block
       _attention_mask` fallback (when xformers isn't installed). We
       replace `self.transformer` with `_PixtralVarlenTransformer`, whose
       attention threads `cu_seqlens`/`max_seqlen` into vLLM's
       `MMEncoderAttention` (same FA-varlen primitive used by the RADIO
       encoder).
    """

    def __init__(
        self,
        args: VisionEncoderArgs,
        quant_config=None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(args, quant_config=quant_config, prefix=prefix)
        disable_tp = is_vit_use_data_parallel()
        # Replace transformer + rotary. Parent's instances are discarded;
        # `patch_conv`, `ln_pre`, and the attribute names `transformer` /
        # `patch_positional_embedding` are preserved so the param key
        # paths line up with the saver's `vision_model.*` output.
        self.transformer = _PixtralVarlenTransformer(
            args,
            quant_config=quant_config,
            prefix=f"{prefix}.transformer",
            disable_tp=disable_tp,
        )
        self.patch_positional_embedding = _PixtralDynamicRotaryEmbedding(
            args
        )

    def forward(
        self,
        images: list[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        # Per-image patch embedding (same shape contract as the parent).
        patch_embeds_list = [
            self.patch_conv(img.unsqueeze(0).to(self.dtype))
            for img in images
        ]
        patch_embeds = [
            p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list
        ]
        embed_sizes = [p.shape[1] for p in patch_embeds]

        patch_embeds = torch.cat(patch_embeds, dim=1)
        patch_embeds = self.ln_pre(patch_embeds)

        # Dynamic per-image freqs_cis (complex64); replaces the parent's
        # `position_meshgrid` + bounded `self.freqs_cis[positions]` lookup.
        freqs_cis = self.patch_positional_embedding(
            patch_embeds, patch_embeds_list
        )

        # Varlen attention metadata — replaces the BlockDiagonalMask /
        # dense `(T, T)` mask the parent forward would build here.
        seq_lens = [p.shape[-2] * p.shape[-1] for p in patch_embeds_list]
        cu_seqlens = torch.zeros(
            len(seq_lens) + 1, dtype=torch.int32, device=self.device
        )
        cu_seqlens[1:] = torch.tensor(
            seq_lens, dtype=torch.int32, device=self.device
        ).cumsum(0)
        mask_meta = _PixtralVarlenMaskMeta(
            cu_seqlens=cu_seqlens,
            # Keep max_seqlen on CPU to avoid a per-layer .item() sync — see
            # vllm/v1/attention/ops/vit_attn_wrappers.py. Matches RADIO's
            # `_inter_image_mask_metadata_from_seq_lens` convention.
            max_seqlen=torch.tensor(max(seq_lens), dtype=torch.int32),
        )

        out = self.transformer(
            patch_embeds, mask=mask_meta, freqs_cis=freqs_cis
        )

        # Same return contract as the parent: per-image tensors split out
        # of the concatenated stream.
        return torch.split(out.squeeze(0), embed_sizes)


class PixtralEncoderAdapter(nn.Module):
    """Adapt vLLM's Mistral-format `VisionTransformer` family (subclassed
    here as `_PixtralVarlenVisionModel` with FA-varlen attention and
    dynamic 2D RoPE) + native `PatchMerger` to the RADIO-style call
    interface used by `NemotronH_Nano_VL_V2.extract_feature_dynamic`:

        _, vit_embeds = self.vision_model(pixel_values, imgs_sizes=imgs_sizes)

    Pixtral's HF encoder forward takes `list[Tensor]` (one per image), so we
    unpack the dynamic-res tiler's packed `(1, total_patches, C*P*P)` tensor
    into per-image `(C, H_px, W_px)` tensors here. After the encoder, we run
    the mcore-trained merger (RMSNorm + Linear(4H→H)) per image and re-pack
    the result into `(1, total_post_merger_tokens, H)` so the call site can
    feed it straight into `mlp1` (skipping `pixel_shuffle_dynamic_res`).

    The adapter owns its own RMSNorm pre-merger because vLLM's `PatchMerger`
    is just the Linear — the trained mcore merger has RMSNorm sitting in
    front of it.

    Dynamic-res only: 4-D static-tiling / video-framewise inputs raise
    NotImplementedError. Pixtral mcore conv3d / video temporal compression
    isn't in mcore yet (Step 10 of vision-encoder-study plan).
    """

    def __init__(
        self,
        pixtral_config: "PixtralVisionConfig",
        *,
        patch_size: int,
        spatial_merge_size: int = 2,
        merger_norm_eps: float = 1e-5,
        prefix: str = "",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size

        # Translate the HF `PixtralVisionConfig` we received into the
        # `VisionEncoderArgs` dataclass that vLLM's Mistral-format encoder
        # family expects. `image_token_id` / `adapter_bias` / `mm_projector_id`
        # are not consulted by the encoder forward — we set safe defaults.
        encoder_args = VisionEncoderArgs(
            hidden_size=pixtral_config.hidden_size,
            num_channels=pixtral_config.num_channels,
            image_size=pixtral_config.image_size,
            patch_size=pixtral_config.patch_size,
            intermediate_size=pixtral_config.intermediate_size,
            num_hidden_layers=pixtral_config.num_hidden_layers,
            num_attention_heads=pixtral_config.num_attention_heads,
            rope_theta=pixtral_config.rope_theta,
            image_token_id=0,
            adapter_bias=False,
            spatial_merge_size=spatial_merge_size,
        )
        self.encoder = _PixtralVarlenVisionModel(
            encoder_args,
            prefix=f"{prefix}.encoder" if prefix else "encoder",
        )
        # The trained mcore merger is RMSNorm(H) + Linear(4H → H, no bias).
        # vLLM's `PatchMerger` exposes only the Linear; we host the RMSNorm.
        self.merger_pre_norm = RMSNorm(
            hidden_size=pixtral_config.hidden_size,
            eps=merger_norm_eps,
        )
        self.merger = _PixtralNativePatchMerger(
            vision_encoder_dim=pixtral_config.hidden_size,
            spatial_merge_size=spatial_merge_size,
            use_mlp_bias=False,
        )

    @property
    def hidden_size(self) -> int:
        # Post-merger hidden equals pre-merger hidden (Linear(4H→H)).
        return self.encoder.args.hidden_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        imgs_sizes: list[tuple[int, int]] | None = None,
        num_frames: int | None = None,
    ) -> tuple[None, torch.Tensor]:
        """Mirror RadioModel's `(summary, all_feat)` return so the caller is
        encoder-agnostic. `summary` is unused for our downstream `mlp1`/LM
        path, so we return None for it.

        Two accepted input shapes (mirroring RADIO's encoder):
        1. 3-D `(1, total_patches, C*P*P)` from the dynamic-res tiler, with
           `imgs_sizes=[(H_px_i, W_px_i), ...]`. Output:
           `(1, sum_i(n_post_merger_i), H)`.
        2. 4-D `(N, C, H, W)` static / framewise (no temporal compression).
           All entries share `(H, W)`. Output: `(N, n_post_merger, H)`.

        `num_frames > 1` raises NotImplementedError — Pixtral conv3d / video
        temporal compression isn't in mcore yet (Step 10 of
        vision-encoder-study plan). vLLM's profile dummy exercises the 4-D
        path with `num_frames=None`, so we accept it as plain framewise
        encoding to keep startup viable.
        """
        if num_frames is not None:
            raise NotImplementedError(
                "Pixtral conv3d/temporal compression is not implemented in "
                "mcore (see Step 10 of vision-encoder-study plan), so vLLM "
                "doesn't accept `num_frames` for Pixtral. Train without "
                "`--video-temporal-patch-size > 1` for now."
            )

        P = self.patch_size

        if pixel_values.ndim == 3:
            if imgs_sizes is None:
                raise ValueError(
                    "PixtralEncoderAdapter: dynamic-res input requires "
                    "`imgs_sizes`."
                )
            # Inverse of `DynamicResolutionImageTiler.stack`.
            flat = pixel_values.squeeze(0)  # (total_patches, C*P*P)
            per_image: list[torch.Tensor] = []
            cursor = 0
            for h_px, w_px in imgs_sizes:
                ph, pw = h_px // P, w_px // P
                n = ph * pw
                patches = flat[cursor : cursor + n]
                cursor += n
                img = (
                    patches.reshape(ph, pw, 3, P, P)
                    .permute(2, 0, 3, 1, 4)
                    .reshape(3, h_px, w_px)
                    .contiguous()
                )
                per_image.append(img)
            token_sizes = [
                (h_px // P, w_px // P) for h_px, w_px in imgs_sizes
            ]

            encoder_features = self.encoder(per_image)
            merged: list[torch.Tensor] = []
            for feat, token_size in zip(encoder_features, token_sizes):
                feat = self.merger_pre_norm(feat)
                feat = self.merger(feat, [token_size])  # (n/4, H)
                merged.append(feat)
            # (1, sum_n/4, H)
            all_feat = torch.cat(merged, dim=0).unsqueeze(0)
            return None, all_feat

        elif pixel_values.ndim == 4:
            if imgs_sizes is not None:
                raise ValueError(
                    "PixtralEncoderAdapter: don't pass `imgs_sizes` with 4-D "
                    "input (static-tiling / video-framewise — shapes are "
                    "uniform)."
                )
            # Static / framewise path — used by vLLM's profile dummy (128
            # frames at 448x448) and by real video requests. Pixtral has no
            # conv3d in mcore, so frames are encoded independently; accuracy
            # is degraded for video since the model wasn't trained for it,
            # but the path doesn't crash.
            #
            # Micro-batch to cap encoder peak memory. Per-call peak inside
            # the encoder forward is dominated by the SwiGLU MLP intermediate
            # at 2 × intermediate_size = 16384 dims (1024-patch frame ⇒
            # ~33.5 MB/frame in bf16). RADIO uses 128-frame chunks but with
            # a smaller encoder (hidden=1280, GLU-free MLP, intermediate=
            # 5120 ⇒ ~10.5 MB/frame). Matching RADIO's per-call peak (1.34
            # GB) gives Pixtral 128 / (33.5 / 10.5) ≈ 40 frames; we round
            # down to 32 for divisibility and a small safety margin. The
            # 32-frame chunk also keeps profile peak << available headroom
            # so vLLM can budget a usable KV cache.
            N, _C, H, W = pixel_values.shape
            token_size = (H // P, W // P)
            MICRO_BATCH_SIZE = 32

            merged_stack: list[torch.Tensor] = []
            for start in range(0, N, MICRO_BATCH_SIZE):
                end = min(start + MICRO_BATCH_SIZE, N)
                per_image = [pixel_values[i] for i in range(start, end)]
                encoder_features = self.encoder(per_image)
                for feat in encoder_features:
                    feat = self.merger_pre_norm(feat)
                    feat = self.merger(feat, [token_size])  # (n/4, H)
                    merged_stack.append(feat)
            # (N, n_post_merger, H) so the caller's reshape path can treat
            # it like RadioModel's batched output.
            return None, torch.stack(merged_stack, dim=0)

        else:
            raise ValueError(
                f"PixtralEncoderAdapter: expected 3-D dynamic-res or 4-D "
                f"static input; got shape {tuple(pixel_values.shape)}."
            )

    def load_weights(self, weights):
        """Route mcore-exported HF-named keys (after the outer
        `vision_model.` strip in `NemotronH_Nano_VL_V2.load_weights`) into
        the adapter submodules. Our saver emits already-merged tensors for
        `qkv_proj` (per-head reorder undoes TE's interleave) and
        `gate_up_proj` (`cat([W, V])`), so no q/k/v split-load is needed —
        vLLM's `QKVParallelLinear` / `MergedColumnParallelLinear`
        weight_loaders accept a fully-merged tensor when called without
        `shard_id`.

        Incoming key examples (post outer-strip):
          vision_model.patch_conv.weight
              → encoder.patch_conv.weight
          vision_model.transformer.layers.{i}.attention.qkv_proj.weight
              → encoder.transformer.layers.{i}.attention.qkv_proj.weight
          vision_model.merger.pre_norm.weight
              → merger_pre_norm.weight (host-owned, not vLLM PatchMerger)
          vision_model.merger.linear_fc1.weight
              → merger.merging_layer.weight (vLLM PatchMerger's Linear)
        """
        params_dict = dict(self.named_parameters())
        for name, w in weights:
            if not name.startswith("vision_model."):
                raise ValueError(
                    f"PixtralEncoderAdapter expected a `vision_model.` "
                    f"prefix on incoming weight key; got {name!r}"
                )
            rest = name[len("vision_model.") :]
            if rest == "merger.pre_norm.weight":
                target = "merger_pre_norm.weight"
            elif rest == "merger.linear_fc1.weight":
                target = "merger.merging_layer.weight"
            else:
                # All other keys land inside the encoder. Submodule names
                # (`patch_conv`, `ln_pre`, `transformer.layers.X.attention.
                # {qkv_proj,o_proj}`, `transformer.layers.X.feed_forward.
                # {gate_up_proj,down_proj}`, `attention_norm`, `ffn_norm`)
                # are identical between vLLM's HF and Mistral-format Pixtral
                # encoder families, so the same key shapes work for both.
                target = f"encoder.{rest}"
            if target not in params_dict:
                raise KeyError(
                    f"PixtralEncoderAdapter: no parameter named {target!r} "
                    f"(mapped from incoming {name!r})"
                )
            param = params_dict[target]
            weight_loader = getattr(
                param, "weight_loader", default_weight_loader
            )
            with torch.no_grad():
                weight_loader(param, w)


def build_pixtral_encoder(
    hf_config, *, prefix: str = ""
) -> PixtralEncoderAdapter:
    """Build the Pixtral-Large encoder + native 2×2 merger adapter.

    Architecture constants come from `megatron-lm/megatron/core/models/vision/
    encoder_registry.py`'s `pixtral-vit-large` entry (the single Pixtral
    variant in scope for the current vision-encoder study). If we ever need
    to support another Pixtral variant, parametrize from
    `hf_config.vision_config` instead of hardcoding here.

    Note: `image_size=448` is symbolic only. The adapter's encoder uses
    `_PixtralDynamicRotaryEmbedding`, which composes per-image 2D RoPE
    freqs from each image's actual `(h, w)` and ignores any fixed
    `max_patches_per_side` table — so the value of `image_size` here is
    not a runtime cap on dynamic-res inputs. We set it to the trained
    tile size (`force_image_size=448`) for symbolic alignment.
    """
    pixtral_config = PixtralVisionConfig(
        hidden_size=1664,
        intermediate_size=8192,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_channels=3,
        image_size=448,
        patch_size=14,
        hidden_act="silu",
        rope_theta=10000.0,
        attention_dropout=0.0,
    )
    return PixtralEncoderAdapter(
        pixtral_config,
        patch_size=14,
        spatial_merge_size=2,
        merger_norm_eps=getattr(hf_config, "projector_norm_eps", 1e-5),
        prefix=prefix,
    )