# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tower-LoRA mapping must match the modality-sorted encoder dispatch order.

`_execute_mm_encoder` sorts `mm_kwargs` by modality so that
`group_and_batch_mm_kwargs` can batch consecutive same-modality items
across requests. The tower/connector LoRA mapping is built from
`mm_lora_refs` and handed to `set_active_adapters`; this test pins that
the mapping is built **after** the sort, so that the adapter assigned to
the i-th encoded item corresponds to the request that actually owns the
i-th dispatched item.

Regression: previously the sort happened after the mapping, so a mixed
modality batch like A[img] A[vid] B[vid] B[img] was dispatched in order
A[img] B[img] A[vid] B[vid] but with the un-permuted LoRA mapping
[A, A, B, B] — the second and third items ended up running under the
wrong adapter.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from vllm.lora.layers.utils import LoRAMapping, LoRAMappingType
from vllm.lora.request import LoRARequest
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _make_runner(mm_lora_refs, request_lora_mapping, tokens_per_item):
    """Build a GPUModelRunner with only the attributes `_execute_mm_encoder`
    reads — avoids the full (GPU-requiring) __init__."""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.observability_config = None
    runner.lora_config = MagicMock()  # truthy
    runner.is_multimodal_pruning_enabled = False
    runner.model_handles_video_batching = False
    runner.encoder_cudagraph_manager = None
    runner.device = "cpu"
    runner.pin_memory = False

    # lora_manager: tower/connector enabled, capture set_active_adapters.
    runner.lora_manager = MagicMock()
    runner.lora_manager.supports_tower_connector_lora.return_value = True

    # input_batch: req_id → index, index → lora_id, lora_id → LoRARequest.
    req_ids = sorted({ref[0] for ref in mm_lora_refs})
    req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
    runner.input_batch = SimpleNamespace(
        req_id_to_index=req_id_to_index,
        request_lora_mapping=np.array(
            [request_lora_mapping[rid] for rid in req_ids], dtype=np.int64
        ),
        lora_id_to_lora_request={
            lora_id: LoRARequest(
                lora_name=f"lora_{lora_id}",
                lora_int_id=lora_id,
                lora_path=f"/tmp/lora_{lora_id}",
            )
            for lora_id in set(request_lora_mapping.values())
            if lora_id > 0
        },
    )

    # Model: tower-only (no connector), deterministic encoder-token count
    # keyed on num_embeds so prompt_mapping and token_mapping are
    # independently observable.
    model = MagicMock()
    model.get_num_mm_encoder_tokens.side_effect = lambda n: tokens_per_item[n]
    model.get_mm_mapping.return_value = SimpleNamespace(connector=False)
    runner.model = model

    return runner


def _fake_mm_input(modality, item_tag):
    # The sort only inspects the tuple's first element (modality string);
    # the item payload is never touched before the encoder runs, and we
    # patch group_and_batch_mm_kwargs to yield nothing.
    return (modality, SimpleNamespace(tag=item_tag))


def _fake_placeholder(num_embeds):
    # PlaceholderRange duck-type: only .get_num_embeds() is called.
    return SimpleNamespace(get_num_embeds=lambda n=num_embeds: n)


def test_tower_lora_mapping_follows_modality_sort():
    # Two requests, different LoRA adapters, each with one image and one
    # video. Original mm_kwargs order interleaves them, which is what
    # _batch_mm_inputs_from_scheduler produces when encoder inputs from
    # multiple requests are scheduled together.
    mm_hashes = ["A_img", "A_vid", "B_vid", "B_img"]
    mm_kwargs = [
        _fake_mm_input("image", "A_img"),
        _fake_mm_input("video", "A_vid"),
        _fake_mm_input("video", "B_vid"),
        _fake_mm_input("image", "B_img"),
    ]
    # Distinct num_embeds per item so we can tell them apart in
    # token_mapping; tokens_per_item maps num_embeds → token count.
    mm_lora_refs = [
        ("req_A", _fake_placeholder(10)),  # A_img
        ("req_A", _fake_placeholder(20)),  # A_vid
        ("req_B", _fake_placeholder(30)),  # B_vid
        ("req_B", _fake_placeholder(40)),  # B_img
    ]
    tokens_per_item = {10: 2, 20: 3, 30: 5, 40: 7}

    runner = _make_runner(
        mm_lora_refs=mm_lora_refs,
        request_lora_mapping={"req_A": 1, "req_B": 2},
        tokens_per_item=tokens_per_item,
    )
    runner._batch_mm_inputs_from_scheduler = MagicMock(
        return_value=(mm_hashes, mm_kwargs, mm_lora_refs)
    )

    # Short-circuit encoder dispatch: we only care about the LoRA mapping
    # that was constructed before it.
    with patch(
        "vllm.v1.worker.gpu_model_runner.group_and_batch_mm_kwargs",
        return_value=iter(()),
    ):
        runner._execute_mm_encoder(scheduler_output=MagicMock())

    # Exactly one set_active_adapters call (tower only; connector skipped
    # because mm_mapping.connector = False).
    assert runner.lora_manager.set_active_adapters.call_count == 1
    _, mapping = runner.lora_manager.set_active_adapters.call_args.args
    assert isinstance(mapping, LoRAMapping)
    assert mapping.type is LoRAMappingType.TOWER

    # Sorted dispatch order (stable modality sort): A_img, B_img, A_vid, B_vid.
    # Adapter ids: [1, 2, 1, 2] — the bug produced [1, 1, 2, 2].
    assert mapping.prompt_mapping == (1, 2, 1, 2)

    # token_mapping expands prompt_mapping by per-item token count, in
    # the same permuted order. Tokens: A_img=2, B_img=7, A_vid=3, B_vid=5.
    assert mapping.index_mapping == (1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2)


def test_tower_lora_mapping_same_modality_preserves_order():
    # Sanity check: when all items are the same modality the stable sort
    # is a no-op and mapping matches the original order.
    mm_hashes = ["A", "B"]
    mm_kwargs = [
        _fake_mm_input("video", "A"),
        _fake_mm_input("video", "B"),
    ]
    mm_lora_refs = [
        ("req_A", _fake_placeholder(4)),
        ("req_B", _fake_placeholder(4)),
    ]
    runner = _make_runner(
        mm_lora_refs=mm_lora_refs,
        request_lora_mapping={"req_A": 1, "req_B": 2},
        tokens_per_item={4: 1},
    )
    runner._batch_mm_inputs_from_scheduler = MagicMock(
        return_value=(mm_hashes, mm_kwargs, mm_lora_refs)
    )

    with patch(
        "vllm.v1.worker.gpu_model_runner.group_and_batch_mm_kwargs",
        return_value=iter(()),
    ):
        runner._execute_mm_encoder(scheduler_output=MagicMock())

    _, mapping = runner.lora_manager.set_active_adapters.call_args.args
    assert mapping.prompt_mapping == (1, 2)
    assert mapping.index_mapping == (1, 2)
