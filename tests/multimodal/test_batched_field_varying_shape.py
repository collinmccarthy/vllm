# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MultiModalBatchedField keeps varying-shape items as list[Tensor].

Pins the contract that models depending on dynamic per-item resolution
(e.g. Nano-Nemotron-VL's variable-resolution video path) rely on: when two
requests contribute video tensors with different spatial dims, the field
reducer must preserve them as a Python list rather than padding into one
dense tensor. Padding would both waste memory and silently hand the model
zero-filled frames that the dynamic-resolution code path is not meant to
see.
"""

import torch

from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalKwargsItem,
)
from vllm.multimodal.utils import group_and_batch_mm_kwargs

KEY = "pixel_values_flat_video"
MODALITY = "video"


def _make_item(tensor: torch.Tensor) -> MultiModalKwargsItem:
    elem = MultiModalFieldElem(data=tensor, field=MultiModalBatchedField())
    return MultiModalKwargsItem({KEY: elem})


def test_varying_shape_preserved_as_list():
    # Landscape and portrait per-video tensors with the same rank but
    # different spatial dims — the canonical cross-request failure mode.
    landscape = torch.zeros(1, 3, 24, 36)
    portrait = torch.zeros(1, 3, 36, 24)
    mm_kwargs = [
        (MODALITY, _make_item(landscape)),
        (MODALITY, _make_item(portrait)),
    ]

    groups = list(group_and_batch_mm_kwargs(mm_kwargs, device="cpu"))

    assert len(groups) == 1
    modality, num_items, batched = groups[0]
    assert modality == MODALITY
    assert num_items == 2

    reduced = batched[KEY]
    # The invariant: different shapes → list[Tensor], shapes preserved.
    assert isinstance(reduced, list)
    assert len(reduced) == 2
    assert reduced[0].shape == landscape.shape
    assert reduced[1].shape == portrait.shape


def test_matching_shape_stacks_to_tensor():
    # Baseline: when shapes match, the reducer stacks along a new leading
    # dim. _parse_and_validate_video_input already reshapes this form.
    a = torch.zeros(1, 3, 24, 36)
    b = torch.zeros(1, 3, 24, 36)
    mm_kwargs = [
        (MODALITY, _make_item(a)),
        (MODALITY, _make_item(b)),
    ]

    ((_, num_items, batched),) = list(
        group_and_batch_mm_kwargs(mm_kwargs, device="cpu")
    )

    assert num_items == 2
    reduced = batched[KEY]
    assert isinstance(reduced, torch.Tensor)
    assert reduced.shape == (2, 1, 3, 24, 36)
