# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Nano-Nemotron-VL's same-resolution video batching.

These tests duck-type a minimal ``NemotronH_Nano_VL_V2`` instance with a
deterministic tubelet-local ``vision_model`` stub. They verify that
``_extract_video_embeddings_temporal_same_resolution`` (segment
microbatch packing, per-segment slicing, and per-video concatenation)
produces identical output whether videos are passed packed together or
one at a time.
"""

import math

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.nano_nemotron_vl import NemotronH_Nano_VL_V2

T = 4
PATCH_SIZE = 14
FRAME_H = 14
FRAME_W = 14
H_PATCHES = FRAME_H // PATCH_SIZE  # 1
W_PATCHES = FRAME_W // PATCH_SIZE  # 1
VIT_HIDDEN = 16
HIDDEN = VIT_HIDDEN  # mlp1 is Identity
DOWNSAMPLE_RATIO = 1.0  # keeps pixel_shuffle a shape no-op
TOKENS_PER_TUBELET = H_PATCHES * W_PATCHES  # = 1 at ratio 1.0


class _TubeletLocalVisionStub:
    """Stub ``vision_model`` that is tubelet-local by construction.

    Input frames carry a scalar id in every pixel (``x[i] == i``). For each
    per-video segment we pad to a multiple of ``T`` by repeating the last
    frame (matching ``forward_video``), then emit one row per tubelet whose
    value depends only on that tubelet's ``T`` frame ids. Replicated
    across the spatial sequence so the row count matches what the real
    encoder would produce.
    """

    def __call__(self, x, num_frames):
        tubelet_rows = []
        offset = 0
        for nf in num_frames:
            video = x[offset : offset + nf]
            offset += nf
            pad = (-nf) % T
            if pad > 0:
                video = torch.cat([video, video[-1:].expand(pad, -1, -1, -1)], dim=0)
            nt = video.shape[0] // T
            for k in range(nt):
                tubelet = video[k * T : (k + 1) * T]
                frame_ids = tubelet[:, 0, 0, 0]  # [T]
                row = torch.zeros(VIT_HIDDEN, dtype=x.dtype)
                row[:T] = frame_ids
                tubelet_rows.append(
                    row.unsqueeze(0)
                    .expand(H_PATCHES * W_PATCHES, VIT_HIDDEN)
                    .contiguous()
                )
        patches = torch.stack(tubelet_rows, dim=0)
        return None, patches


def _make_model() -> NemotronH_Nano_VL_V2:
    """Build a duck-typed model with only the attributes the extraction
    method touches. Avoids loading real weights or config."""
    model = NemotronH_Nano_VL_V2.__new__(NemotronH_Nano_VL_V2)
    # Initialize the nn.Module internals without running the full __init__,
    # so assigning submodules (mlp1) below does not raise.
    nn.Module.__init__(model)
    model.vision_model = _TubeletLocalVisionStub()
    model.mlp1 = nn.Identity()
    model.downsample_ratio = DOWNSAMPLE_RATIO
    model.ps_version = "v2"
    model.video_encoder_micro_batch_size = 128
    return model


def _make_pixel_values(num_frames_per_video: list[int]) -> torch.Tensor:
    """Frame ``i`` is a 3xHxW tensor filled with the scalar ``i``."""
    total = sum(num_frames_per_video)
    x = torch.zeros(total, 3, FRAME_H, FRAME_W)
    for i in range(total):
        x[i].fill_(float(i))
    return x


def _extract(
    model: NemotronH_Nano_VL_V2,
    pixel_values: torch.Tensor,
    num_frames_per_video: list[int],
) -> tuple[torch.Tensor, ...]:
    return model._extract_video_embeddings_temporal_same_resolution(
        pixel_values=pixel_values,
        num_frames_per_video=num_frames_per_video,
        hidden_size=HIDDEN,
        T=T,
        patch_size=PATCH_SIZE,
    )


def _solo_outputs(
    model: NemotronH_Nano_VL_V2,
    pixel_values: torch.Tensor,
    num_frames_per_video: list[int],
) -> list[torch.Tensor]:
    """Call the extraction method once per video, slicing ``pixel_values``
    so the exact same frame tensors are seen as in the packed call."""
    offsets = [0]
    for nf in num_frames_per_video:
        offsets.append(offsets[-1] + nf)

    outputs = []
    for i, nf in enumerate(num_frames_per_video):
        (out,) = _extract(
            model,
            pixel_values[offsets[i] : offsets[i + 1]],
            [nf],
        )
        outputs.append(out)
    return outputs


@pytest.mark.parametrize(
    "num_frames_per_video",
    [
        [10],
        [50, 70],
        [30, 40, 50, 60],
        [1, 51, 128, 4],
        # Long video that crosses the cap and triggers in-microbatch chunking.
        [200],
        # Long video joining the tail of a partial microbatch: the case
        # the previous whole-video-microbatch logic would silently mishandle.
        [30, 250],
        # Several videos that force multiple flush/new-microbatch boundaries.
        [100, 100, 100],
    ],
)
def test_per_video_equivalence_packed_vs_solo(
    num_frames_per_video: list[int],
) -> None:
    model = _make_model()
    pixel_values = _make_pixel_values(num_frames_per_video)

    packed = _extract(model, pixel_values, num_frames_per_video)
    solo = _solo_outputs(model, pixel_values, num_frames_per_video)

    assert len(packed) == len(num_frames_per_video)
    assert len(packed) == len(solo)
    for i, (p, s) in enumerate(zip(packed, solo)):
        assert torch.equal(p, s), (
            f"video {i} (nf={num_frames_per_video[i]}) packed output differs "
            f"from solo output"
        )


def test_tail_split_equivalence() -> None:
    """[30, 250] at T=4, cap=128 exercises the hardest case:
    video 1 is non-final-chunked across three microbatches, with its
    first chunk sharing a microbatch with the tail of video 0."""
    model = _make_model()
    num_frames = [30, 250]
    pixel_values = _make_pixel_values(num_frames)

    packed = _extract(model, pixel_values, num_frames)
    solo = _solo_outputs(model, pixel_values, num_frames)

    for i, (p, s) in enumerate(zip(packed, solo)):
        assert torch.equal(p, s), f"video {i} differs across packed/solo"


def test_output_length_invariant() -> None:
    """For every video, output row count equals
    ``ceil(nf_i / T) * tokens_per_tubelet``. Includes a long odd nf that
    crosses a microbatch boundary, catching dropped/duplicated final segments."""
    model = _make_model()
    num_frames = [1, 51, 303, 4]
    outputs = _extract(model, _make_pixel_values(num_frames), num_frames)

    assert len(outputs) == len(num_frames)
    for nf, out in zip(num_frames, outputs):
        expected_rows = math.ceil(nf / T) * TOKENS_PER_TUBELET
        assert out.shape == (expected_rows, HIDDEN), (
            f"nf={nf}: expected shape {(expected_rows, HIDDEN)}, got {tuple(out.shape)}"
        )


def test_single_video_matches_first_of_batch() -> None:
    """[A] equals the first output of [A, B]: smoke test for cross-video
    contamination that is subsumed by ``test_per_video_equivalence_packed_vs_solo``
    but cheaper to read when a regression lands here."""
    model = _make_model()
    nf = [50, 70]
    pixel_values = _make_pixel_values(nf)

    packed = _extract(model, pixel_values, nf)
    (solo_first,) = _extract(model, pixel_values[: nf[0]], [nf[0]])

    assert torch.equal(packed[0], solo_first)


def test_output_count_matches_input_count() -> None:
    model = _make_model()
    nf = [10, 20, 30, 40]
    outputs = _extract(model, _make_pixel_values(nf), nf)
    assert len(outputs) == 4
