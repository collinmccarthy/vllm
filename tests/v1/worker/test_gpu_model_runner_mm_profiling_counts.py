# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in hook for heterogeneous profiling dummies.

`_get_mm_dummy_batch` historically asked the registry for a single dummy
item and repeated it `max_items_per_batch` times. Models that need
cross-item variety during profiling (e.g. mixed-aspect-ratio videos to
exercise `MultiModalBatchedField`'s list-preserving reducer path) opt in
by exposing `get_dummy_mm_counts_for_profiling(modality, max_items)`
returning how many distinct items the runner should request. The runner
then cycles through those items to fill the batch.

These tests pin three behaviours:

1. When the model does not define the hook, the runner requests
   `mm_counts={modality: 1}` (legacy path, no profiling-cost regression
   for multimodal models that don't need diversity).
2. When the hook returns `N > 1`, the runner requests
   `mm_counts={modality: N}` and the resulting batch cycles through the
   distinct items.
3. The runner clamps the hook's return into `[1, max_items_per_batch]`
   so a model returning an absurd value can't break profiling.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _make_runner(
    distinct_items,
    model_hook=None,
    captured_mm_counts=None,
):
    """Build a GPUModelRunner with only the attributes
    `_get_mm_dummy_batch` reads."""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.mm_budget = SimpleNamespace(cache=MagicMock())
    runner.model_config = MagicMock()
    runner.device = "cpu"
    runner.pin_memory = False

    def fake_get_dummy_mm_inputs(model_config, *, mm_counts, cache):
        if captured_mm_counts is not None:
            captured_mm_counts.update(mm_counts)
        modality = next(iter(mm_counts))
        return {"mm_kwargs": {modality: distinct_items[: mm_counts[modality]]}}

    runner.mm_registry = SimpleNamespace(get_dummy_mm_inputs=fake_get_dummy_mm_inputs)

    model = SimpleNamespace()
    if model_hook is not None:
        model.get_dummy_mm_counts_for_profiling = model_hook
    runner.model = model
    return runner


def _capture_group_and_batch():
    """Capture the list of (modality, item) tuples handed to
    `group_and_batch_mm_kwargs`, and yield an empty dict as the only
    group so `_get_mm_dummy_batch` can return."""
    captured = []

    def fake_group(mm_kwargs, *, device, pin_memory):
        captured.append(list(mm_kwargs))
        yield ("dummy", len(mm_kwargs), {})

    return captured, fake_group


def test_default_path_requests_one_distinct_item():
    # Model doesn't implement the hook → legacy behaviour.
    items = [f"item_{i}" for i in range(4)]
    captured_counts = {}
    runner = _make_runner(
        distinct_items=items,
        model_hook=None,
        captured_mm_counts=captured_counts,
    )
    captured_dispatch, fake_group = _capture_group_and_batch()

    with patch(
        "vllm.v1.worker.gpu_model_runner.group_and_batch_mm_kwargs",
        fake_group,
    ):
        runner._get_mm_dummy_batch(modality="video", max_items_per_batch=4)

    assert captured_counts == {"video": 1}
    dispatched_items = [item for _, item in captured_dispatch[0]]
    # One distinct item, repeated — matches legacy behaviour.
    assert dispatched_items == [items[0]] * 4


def test_hook_requests_heterogeneous_items():
    items = [f"item_{i}" for i in range(3)]
    captured_counts = {}
    runner = _make_runner(
        distinct_items=items,
        model_hook=lambda modality, m: m if modality == "video" else 1,
        captured_mm_counts=captured_counts,
    )
    captured_dispatch, fake_group = _capture_group_and_batch()

    with patch(
        "vllm.v1.worker.gpu_model_runner.group_and_batch_mm_kwargs",
        fake_group,
    ):
        runner._get_mm_dummy_batch(modality="video", max_items_per_batch=3)

    assert captured_counts == {"video": 3}
    dispatched_items = [item for _, item in captured_dispatch[0]]
    # Three distinct items in order — no cycling needed when
    # distinct_count == max_items_per_batch.
    assert dispatched_items == items


def test_hook_cycles_when_distinct_count_is_less_than_batch():
    # Hook returns 2 distinct items for a batch of 5 → cycle.
    items = [f"item_{i}" for i in range(2)]
    captured_counts = {}
    runner = _make_runner(
        distinct_items=items,
        model_hook=lambda modality, m: 2,
        captured_mm_counts=captured_counts,
    )
    captured_dispatch, fake_group = _capture_group_and_batch()

    with patch(
        "vllm.v1.worker.gpu_model_runner.group_and_batch_mm_kwargs",
        fake_group,
    ):
        runner._get_mm_dummy_batch(modality="video", max_items_per_batch=5)

    assert captured_counts == {"video": 2}
    dispatched_items = [item for _, item in captured_dispatch[0]]
    assert dispatched_items == [items[0], items[1], items[0], items[1], items[0]]


def test_asserts_when_registry_returns_too_few_items():
    # Fail-fast contract: if the registry (or a misbehaving model
    # patching it) returns fewer items than requested, the profiling
    # path should raise a clear assertion before trying to cycle — not
    # IndexError its way into a confusing failure.
    items = ["item_0"]  # only one available
    runner = _make_runner(
        distinct_items=items,
        # Ask for 3 distinct items but registry will only return 1.
        model_hook=lambda modality, m: 3,
    )
    _, fake_group = _capture_group_and_batch()

    with pytest.raises(AssertionError, match="Expected at least 3 dummy items"):
        with patch(
            "vllm.v1.worker.gpu_model_runner.group_and_batch_mm_kwargs",
            fake_group,
        ):
            runner._get_mm_dummy_batch(modality="video", max_items_per_batch=4)


def test_asserts_when_batching_splits_into_multiple_groups():
    # Explicit contract: profiling dummies all share one modality and no
    # MultiModalSharedField distinctions, so group_and_batch_mm_kwargs
    # must yield exactly one group. If a future change causes a split,
    # we want the failure surfaced here, not silently-dropped items.
    items = ["item_0", "item_1"]
    runner = _make_runner(
        distinct_items=items,
        model_hook=lambda modality, m: 2,
    )

    def two_group_fake(mm_kwargs, *, device, pin_memory):
        # Simulate an unexpected split: yield two groups instead of one.
        half = len(mm_kwargs) // 2
        yield ("video", half, {})
        yield ("video", len(mm_kwargs) - half, {})

    with (
        pytest.raises(AssertionError, match="1 group, got 2"),
        patch(
            "vllm.v1.worker.gpu_model_runner.group_and_batch_mm_kwargs",
            two_group_fake,
        ),
    ):
        runner._get_mm_dummy_batch(modality="video", max_items_per_batch=4)


def test_hook_return_is_clamped():
    # Hook returns more than max_items_per_batch → clamp to max.
    # And a return < 1 is clamped to 1.
    items = [f"item_{i}" for i in range(4)]
    captured_counts = {}
    runner = _make_runner(
        distinct_items=items,
        model_hook=lambda modality, m: 999,
        captured_mm_counts=captured_counts,
    )
    _, fake_group = _capture_group_and_batch()

    with patch(
        "vllm.v1.worker.gpu_model_runner.group_and_batch_mm_kwargs",
        fake_group,
    ):
        runner._get_mm_dummy_batch(modality="video", max_items_per_batch=4)

    assert captured_counts == {"video": 4}

    # Zero / negative clamps up to 1.
    captured_counts.clear()
    runner.model.get_dummy_mm_counts_for_profiling = lambda modality, m: 0
    with patch(
        "vllm.v1.worker.gpu_model_runner.group_and_batch_mm_kwargs",
        fake_group,
    ):
        runner._get_mm_dummy_batch(modality="video", max_items_per_batch=4)
    assert captured_counts == {"video": 1}
