# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pins the processor-side boundary for varying-resolution video.

The video field uses ``MultiModalFieldConfig.batched("video")``, which
requires the processor to hand vLLM a per-video ``list[Tensor]`` so the
batched field can build one item per video. vLLM's
``MultiModalProcessingContext`` defaults ``return_tensors="pt"``, which
makes the processor call ``BatchFeature(..., tensor_type="pt")``. Without
the pop/reassign escape hatch, ``BatchFeature`` tensorises the list —
silently collapsing same-shape videos into one stacked tensor, and
failing outright for mixed H/W.

This test duck-types a minimal ``NanoNemotronVLProcessor`` and calls
``__call__`` with mixed-resolution video inputs, then asserts that
``pixel_values_flat_video`` survives as a list of per-video tensors with
their original shapes.
"""

import torch

from vllm.transformers_utils.processors.nano_nemotron_vl import (
    NanoNemotronVLProcessor,
)


class _StubTokenizer:
    def __call__(self, text, add_special_tokens=False):
        # BatchFeature expects the usual HF tokenizer outputs. A single
        # token id per sample is enough — we only care about the video
        # field's fate through the BatchFeature boundary.
        return {
            "input_ids": [[0] for _ in text],
            "attention_mask": [[1] for _ in text],
        }


def _make_processor():
    """Build the minimal surface ``__call__`` touches. Nothing else is
    instantiated so we don't need real HF weights/config."""
    p = NanoNemotronVLProcessor.__new__(NanoNemotronVLProcessor)
    p.max_num_tiles = 1
    p.dynamic_tiler = None
    p.tokenizer = _StubTokenizer()

    p._preprocess_image = lambda *, text, images, max_num_tiles: (text, {})
    p._preprocess_audio = lambda *, text, audios: (text, {})
    return p


def test_pixel_values_flat_video_survives_batch_feature_mixed_resolution():
    landscape = torch.zeros(2, 3, 24, 36)
    portrait = torch.zeros(2, 3, 36, 24)
    pixel_values_lst_video = [landscape, portrait]

    def _fake_preprocess_video(*, text, videos):
        video_inputs = {
            "pixel_values_flat_video": pixel_values_lst_video,
            "video_num_patches": torch.tensor([2, 2]),
            # Same-length frames_indices keeps us out of the
            # ragged_frames_indices escape hatch; we want to isolate the
            # pixel_values_flat_video escape hatch under test.
            "frames_indices": [[0, 1], [0, 1]],
            "frame_duration_ms": torch.tensor([500, 500]),
        }
        return text, video_inputs

    p = _make_processor()
    p._preprocess_video = _fake_preprocess_video

    batch = p(text="x", videos=[object(), object()], return_tensors="pt")

    out = batch["pixel_values_flat_video"]
    assert isinstance(out, list), (
        f"batched('video') contract requires list[Tensor]; BatchFeature "
        f"tensorised to {type(out).__name__}"
    )
    assert len(out) == 2
    assert out[0].shape == landscape.shape
    assert out[1].shape == portrait.shape
    # And the object identity matches — nothing copied or reshaped.
    assert out[0] is landscape
    assert out[1] is portrait


def test_pixel_values_flat_video_survives_batch_feature_uniform_resolution():
    # Same-shape case: the batched reducer downstream will stack these,
    # but that's the reducer's job — the processor boundary must still
    # present a list so the field builds one elem per video.
    v0 = torch.zeros(2, 3, 24, 24)
    v1 = torch.zeros(2, 3, 24, 24)

    def _fake_preprocess_video(*, text, videos):
        return text, {
            "pixel_values_flat_video": [v0, v1],
            "video_num_patches": torch.tensor([2, 2]),
            "frames_indices": [[0, 1], [0, 1]],
            "frame_duration_ms": torch.tensor([500, 500]),
        }

    p = _make_processor()
    p._preprocess_video = _fake_preprocess_video

    batch = p(text="x", videos=[object(), object()], return_tensors="pt")

    out = batch["pixel_values_flat_video"]
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0] is v0 and out[1] is v1
