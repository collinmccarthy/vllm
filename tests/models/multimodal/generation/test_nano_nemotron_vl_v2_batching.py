# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression guard for NemotronH_Nano_VL_V2 video encoder batching.

When the scheduler admits multiple videos in the same encoder step, every
per-video embedding must match what the same model produces when the
video is processed alone. This file covers the paths introduced by the
PR that enables batched video encoding for NanoNemotron-VL:

* same-resolution fast path (frame-budgeted microbatches of ~128 frames,
  splits long videos across microbatches),
* varying-resolution dynamic path (patch-budgeted microbatches targeting
  ~32768 patches, whole-video grouping only),
* the packed ``pixel_values_flat`` tensor + ``num_patches`` splitting,
* the 128-frame microbatch boundary in the fast path,
* different-resolution videos batched together into dynamic microbatches,
* `_process_video_input` (EVS pruning + final embedding construction;
  same-resolution only, see the test docstring for why),
* the cache-by-hash loop in `_execute_mm_encoder` that runs after the
  modality-sort permutation.

Tolerances:

* ``REL_TOL_SAME_RES`` (~2e-2) is used for solo-vs-batched same-res
  comparisons. Different scenarios feed the encoder different total
  token counts (e.g. nf=129 solo splits 128 + 1; ``[129, 8]`` batched
  splits 128 + 9), which changes GEMM/attention tile shapes and bf16
  accumulation order. Small drift is the expected baseline — leakage
  has to clear that floor to be a regression.
* ``REL_TOL_DYNAMIC`` (~5e-3) is used for the dynamic path; per-item
  attention masking should keep numeric noise tighter there.
* For *true* cross-video leakage the file uses bit-exact
  (``torch.equal``) companion-invariance tests that hold the batch
  shape fixed and vary only a companion video's pixels — any non-bit-
  exact change in the queried video is then unambiguously a leak.

These tests intentionally call private methods such as
``_extract_video_embeddings_temporal`` to target the exact paths the PR
introduces. Expect to update this file whenever those methods are
renamed or restructured.

The heavy tests run on a real `NemotronH_Nano_VL_V2` checkpoint. Point
the env var ``VLLM_NEMOTRON_VL_V2_PATH`` at a local checkpoint directory
and have at least ``VLLM_TEST_TP`` (default 8) CUDA GPUs available;
otherwise they skip cleanly.

The cache-mapping test is CPU-only and always runs.

Invocation:

    VLLM_NEMOTRON_VL_V2_PATH=/path/to/hf-ckpt \
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    pytest -v \
      tests/models/multimodal/generation/test_nano_nemotron_vl_v2_batching.py
"""

from __future__ import annotations

import math
import os

import pytest
import torch

MODEL_PATH_ENV = "VLLM_NEMOTRON_VL_V2_PATH"
MODEL_PATH = os.environ.get(MODEL_PATH_ENV, "")
TP = int(os.environ.get("VLLM_TEST_TP", "8"))


def _accelerator_device_count() -> int:
    """Safe probe; never raises at import time."""
    try:
        return torch.accelerator.device_count()
    except Exception:
        return 0


needs_model = pytest.mark.skipif(
    not MODEL_PATH or not os.path.isdir(MODEL_PATH),
    reason=(
        f"Set {MODEL_PATH_ENV} to a local NemotronH_Nano_VL_V2 checkpoint "
        "directory to enable this integration test."
    ),
)
needs_gpus = pytest.mark.skipif(
    _accelerator_device_count() < TP,
    reason=(
        f"Need at least TP={TP} accelerator devices (override with VLLM_TEST_TP=<n>)."
    ),
)

SEED = 42
# Same-resolution solo-vs-batched comparison: solo and batched runs hand
# different total token counts to the encoder (e.g. nf=129 solo splits
# 128 + 1, but `[129-frame, 8-frame]` batched splits 128 + (1 + 8)),
# which changes varlen metadata, GEMM dimensions, FlashAttention/cuBLAS
# tile shapes, and bf16 accumulation order. The semantic result is the
# same but the bf16 outputs drift slightly. Keep this ceiling tight
# enough to catch real leakage but not so tight that legitimate kernel-
# shape drift looks like a regression.
REL_TOL_SAME_RES = 5e-3
# Dynamic path packs patches from multiple items into a single ViT
# forward. Bit-exactness relies on strict per-item attention masking;
# keep a looser ceiling so numerically-correct small differences do not
# look like regressions, but still catch genuine attention-leak bugs.
REL_TOL_DYNAMIC = 5e-3


# ---------------------------------------------------------------------------
# Module-level helpers. Must be picklable so `collective_rpc` can ship the
# per-test worker functions into each TP worker process.
# ---------------------------------------------------------------------------


def _sanity_check_model(model) -> list[str]:
    failures: list[str] = []
    if type(model).__name__ != "NemotronH_Nano_VL_V2":
        failures.append(f"wrong model class {type(model).__name__}")
    if not getattr(model, "handles_video_batching_internally", False):
        failures.append(
            "model missing handles_video_batching_internally=True; the "
            "scheduler would not exercise the batched encoder path"
        )
    return failures


def _model_ctx(worker):
    """Return ((model, device, T, patch_size, H, W), []) or (None, failures)."""
    model = worker.model_runner.get_model()
    failures = _sanity_check_model(model)
    if failures:
        return None, failures
    device = next(model.parameters()).device
    T = int(model.video_temporal_patch_size)
    patch_size = int(model.patch_size)
    H = W = patch_size * 16
    return (model, device, T, patch_size, H, W), []


def _video_embedder_loaded(model) -> bool:
    """`_extract_video_embeddings_temporal_dynamic` requires the video
    embedder weights. Returning False lets dynamic-path tests skip
    gracefully on checkpoints trained without temporal compression.
    """
    pg = getattr(getattr(model, "vision_model", None), "patch_generator", None)
    return bool(getattr(pg, "_video_embedder_loaded", False))


def _make_video(
    device: torch.device, nf: int, H: int, W: int, seed: int
) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(nf, 3, H, W, dtype=torch.bfloat16, device=device, generator=g)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max-abs relative difference; 1e-9 guards against a zero reference."""
    return ((a.float() - b.float()).abs().max() / (a.float().abs().max() + 1e-9)).item()


def _same_res_input(videos: list[torch.Tensor]) -> dict:
    """Packed same-resolution input (the shape the scheduler hands us)."""
    device = videos[0].device
    return {
        "pixel_values_flat": torch.cat(videos, dim=0),
        "num_patches": torch.tensor(
            [v.shape[0] for v in videos], dtype=torch.long, device=device
        ),
    }


def _varying_res_input(videos: list[torch.Tensor]) -> dict:
    """List-shaped input that triggers the dynamic-resolution path."""
    device = videos[0].device
    return {
        "pixel_values_flat": list(videos),
        "num_patches": torch.tensor(
            [v.shape[0] for v in videos], dtype=torch.long, device=device
        ),
    }


def _full_same_res_input(videos: list[torch.Tensor]) -> dict:
    """Full video input accepted by `_process_video_input` (same-res only;
    `_process_video_input` reads `pixel_values.shape[-2:]` so dynamic-res
    is not supported end-to-end at that entry point)."""
    device = videos[0].device
    nfs = [v.shape[0] for v in videos]
    return {
        "type": "pixel_values_videos",
        "pixel_values_flat": torch.cat(videos, dim=0),
        "num_patches": torch.tensor(nfs, dtype=torch.long, device=device),
        "frames_indices": torch.cat(
            [torch.arange(nf, dtype=torch.long, device=device) for nf in nfs]
        ),
        "frame_duration_ms": torch.full(
            (len(videos),), 500.0, dtype=torch.float32, device=device
        ),
    }


def _compare_tensor_lists(
    label: str,
    refs: list[torch.Tensor],
    actual,
    tol: float,
    failures: list[str],
    info: list[str],
) -> None:
    """Strict length + shape + rel-diff comparison that appends to the
    given failures/info lists. Length or shape mismatches are hard
    failures rather than silently skipped checks."""
    if len(actual) != len(refs):
        failures.append(f"{label}: expected {len(refs)} outputs, got {len(actual)}")
        return
    for i in range(len(refs)):
        if refs[i].shape != actual[i].shape:
            failures.append(
                f"{label}[{i}]: shape mismatch ref={tuple(refs[i].shape)} "
                f"actual={tuple(actual[i].shape)}"
            )
            continue
        r = _rel(refs[i], actual[i])
        info.append(f"{label}[{i}] rel_diff={r:.4g}")
        if r > tol:
            failures.append(f"{label}[{i}]: rel_diff {r:.4g} > {tol}")


# ---------------------------------------------------------------------------
# Per-test worker functions. Each returns (ok, failures, info) and is
# dispatched via `collective_rpc` so all TP ranks participate.
# ---------------------------------------------------------------------------


def _w_same_res_batched_equals_solo(worker):
    failures: list[str] = []
    info: list[str] = []
    ctx, fs = _model_ctx(worker)
    if fs:
        return False, fs, info
    model, device, T, patch_size, H, W = ctx
    if T <= 1:
        info.append("skipped: video_temporal_patch_size <= 1")
        return True, failures, info
    info.append(f"T={T} patch={patch_size} HxW={H}x{W}")

    nf = 8
    videos = [_make_video(device, nf, H, W, seed=SEED + i) for i in range(4)]
    with torch.no_grad():
        solos = [
            model._extract_video_embeddings_temporal(_same_res_input([v]))[0]
            for v in videos
        ]
        batched2 = model._extract_video_embeddings_temporal(_same_res_input(videos[:2]))
        batched4 = model._extract_video_embeddings_temporal(_same_res_input(videos))
    _compare_tensor_lists(
        "same_res_bs2", solos[:2], batched2, REL_TOL_SAME_RES, failures, info
    )
    _compare_tensor_lists(
        "same_res_bs4", solos, batched4, REL_TOL_SAME_RES, failures, info
    )
    return not failures, failures, info


def _w_dynamic_res_batched_equals_solo(worker):
    failures: list[str] = []
    info: list[str] = []
    ctx, fs = _model_ctx(worker)
    if fs:
        return False, fs, info
    model, device, T, patch_size, _H, _W = ctx
    if T <= 1:
        info.append("skipped: video_temporal_patch_size <= 1")
        return True, failures, info
    if not _video_embedder_loaded(model):
        info.append("skipped: video_embedder weights not loaded")
        return True, failures, info

    # Three distinct (H, W) so `_has_varying_resolution` is True.
    H1, W1 = patch_size * 16, patch_size * 16
    H2, W2 = patch_size * 20, patch_size * 16
    H3, W3 = patch_size * 16, patch_size * 20
    videos = [
        _make_video(device, 8, H1, W1, seed=SEED + 0),
        _make_video(device, 6, H2, W2, seed=SEED + 1),
        _make_video(device, 10, H3, W3, seed=SEED + 2),
    ]
    nfs = [v.shape[0] for v in videos]
    hidden_size = model.config.text_config.hidden_size
    info.append(f"dyn sizes=[{H1}x{W1}, {H2}x{W2}, {H3}x{W3}] nfs={nfs} T={T}")

    with torch.no_grad():
        # Force the dynamic path for the solo reference; with only one
        # video the top-level dispatcher would otherwise fall back to
        # same-res.
        solos = [
            model._extract_video_embeddings_temporal_dynamic(
                pixel_values=[v],
                num_frames_per_video=[v.shape[0]],
                hidden_size=hidden_size,
                T=T,
                patch_size=patch_size,
            )[0]
            for v in videos
        ]
        batched = model._extract_video_embeddings_temporal_dynamic(
            pixel_values=videos,
            num_frames_per_video=nfs,
            hidden_size=hidden_size,
            T=T,
            patch_size=patch_size,
        )
        # End-to-end: list-shaped pixel_values_flat must route to dynamic.
        via_top = model._extract_video_embeddings_temporal(_varying_res_input(videos))

    _compare_tensor_lists("dyn_direct", solos, batched, REL_TOL_DYNAMIC, failures, info)
    if len(via_top) != len(videos):
        failures.append(
            f"dyn_top_dispatch: expected {len(videos)} outputs, "
            f"got {len(via_top)} (varying-res input did not route to the "
            "dynamic path)"
        )
    else:
        _compare_tensor_lists(
            "dyn_top_dispatch",
            solos,
            via_top,
            REL_TOL_DYNAMIC,
            failures,
            info,
        )
    return not failures, failures, info


def _w_packed_tensor_splits(worker):
    failures: list[str] = []
    info: list[str] = []
    ctx, fs = _model_ctx(worker)
    if fs:
        return False, fs, info
    model, device, T, _patch_size, H, W = ctx
    if T <= 1:
        info.append("skipped: video_temporal_patch_size <= 1")
        return True, failures, info

    # Regression: pixel_values_flat arrives as ONE concatenated tensor
    # with num_patches=[n1, n2, n3]. A prior bug wrapped it in a
    # one-element list and zip-truncated to a single iteration, returning
    # just one embedding.
    nfs = [6, 4, 10]
    videos = [_make_video(device, nf, H, W, seed=SEED + i) for i, nf in enumerate(nfs)]
    with torch.no_grad():
        solos = [
            model._extract_video_embeddings_temporal(_same_res_input([v]))[0]
            for v in videos
        ]
        packed_in = _same_res_input(videos)
        assert torch.is_tensor(packed_in["pixel_values_flat"]), (
            "precondition: packed input must be a single tensor"
        )
        packed = model._extract_video_embeddings_temporal(packed_in)
    info.append(f"packed nfs={nfs}")
    _compare_tensor_lists("packed", solos, packed, REL_TOL_SAME_RES, failures, info)
    return not failures, failures, info


def _w_microbatch_boundaries_same_res(worker):
    failures: list[str] = []
    info: list[str] = []
    ctx, fs = _model_ctx(worker)
    if fs:
        return False, fs, info
    model, device, T, _patch_size, H, W = ctx
    if T <= 1:
        info.append("skipped: video_temporal_patch_size <= 1")
        return True, failures, info

    # Microbatch cap is 128 frames aligned to T. Cover the boundary, an
    # odd nf (tubelet-padding path), and cross-boundary batched
    # equivalence.
    boundary_nfs = [7, 127, 128, 129]
    for nf in boundary_nfs:
        with torch.no_grad():
            emb = model._extract_video_embeddings_temporal(
                _same_res_input([_make_video(device, nf, H, W, seed=SEED)])
            )[0]
        nt = math.ceil(nf / T)
        if emb.shape[0] % nt != 0:
            failures.append(
                f"nf={nf}: n_tokens={emb.shape[0]} not divisible by "
                f"num_tubelets={nt} -- breaks _process_video_input assertion"
            )
        else:
            info.append(
                f"nf={nf} nt={nt} n_tokens={emb.shape[0]} "
                f"(tokens/tubelet={emb.shape[0] // nt})"
            )

    # Cross-boundary batched-vs-solo: cover both 127 (last chunk below
    # the microbatch cap) and 129 (first chunk fills the cap, second
    # chunk is a single tubelet). In both cases the boundary-sized
    # video must be invariant when batched with a short companion.
    v8 = _make_video(device, 8, H, W, seed=SEED + 11)
    with torch.no_grad():
        solo8 = model._extract_video_embeddings_temporal(_same_res_input([v8]))[0]
        for long_nf, seed_offset in ((127, 10), (129, 12)):
            vlong = _make_video(device, long_nf, H, W, seed=SEED + seed_offset)
            solo_long = model._extract_video_embeddings_temporal(
                _same_res_input([vlong])
            )[0]
            both = model._extract_video_embeddings_temporal(
                _same_res_input([vlong, v8])
            )
            _compare_tensor_lists(
                f"boundary_mixed_nf{long_nf}",
                [solo_long, solo8],
                both,
                REL_TOL_SAME_RES,
                failures,
                info,
            )
    return not failures, failures, info


def _w_same_res_companion_invariance(worker):
    """Bit-exact: changing only a companion video's *pixels* (not its
    shape, frame count, or position in the batch) must not perturb the
    queried video's output by even one ULP. This holds the encoder
    execution shape fixed — same total token count, same varlen
    metadata, same kernel tile shapes — so any drift would have to come
    from cross-video computation, i.e. an attention masking or packing
    leak. ``torch.equal`` rather than rel-diff so no accumulation noise
    can hide a real leak.
    """
    failures: list[str] = []
    info: list[str] = []
    ctx, fs = _model_ctx(worker)
    if fs:
        return False, fs, info
    model, device, T, _patch_size, H, W = ctx
    if T <= 1:
        info.append("skipped: video_temporal_patch_size <= 1")
        return True, failures, info

    # Use nf=129 so the queried-A direction still spans the 128-frame
    # microbatch boundary — the same shape that produces solo-vs-batched
    # bf16 drift, but here the batch shape is identical between runs so
    # bit-exactness is the right contract.
    a_long = _make_video(device, 129, H, W, seed=SEED + 100)
    a_long_alt = _make_video(device, 129, H, W, seed=SEED + 101)
    b_short = _make_video(device, 8, H, W, seed=SEED + 200)
    b_short_alt = _make_video(device, 8, H, W, seed=SEED + 201)

    with torch.no_grad():
        # Direction 1: same A, different B. A's output must be identical.
        ab = model._extract_video_embeddings_temporal(
            _same_res_input([a_long, b_short])
        )
        ab_alt = model._extract_video_embeddings_temporal(
            _same_res_input([a_long, b_short_alt])
        )
        # Direction 2: same B, different A. B's output must be identical.
        cb = model._extract_video_embeddings_temporal(
            _same_res_input([a_long_alt, b_short])
        )

    info.append("companion-invariance shapes: A=129f, B=8f, fixed across runs")

    if ab[0].shape != ab_alt[0].shape:
        failures.append(
            f"queried-A shape changed across companion swap: "
            f"{tuple(ab[0].shape)} vs {tuple(ab_alt[0].shape)}"
        )
    elif not torch.equal(ab[0], ab_alt[0]):
        diff = (ab[0].float() - ab_alt[0].float()).abs().max().item()
        failures.append(
            f"queried-A perturbed by companion B's pixels (max abs diff "
            f"{diff:.4g}); same A, different B should be bit-exact under "
            "fixed batch shape"
        )

    if ab[1].shape != cb[1].shape:
        failures.append(
            f"queried-B shape changed across companion swap: "
            f"{tuple(ab[1].shape)} vs {tuple(cb[1].shape)}"
        )
    elif not torch.equal(ab[1], cb[1]):
        diff = (ab[1].float() - cb[1].float()).abs().max().item()
        failures.append(
            f"queried-B perturbed by companion A's pixels (max abs diff "
            f"{diff:.4g}); same B, different A should be bit-exact under "
            "fixed batch shape"
        )

    return not failures, failures, info


def _w_microbatch_boundaries_dynamic_res(worker):
    failures: list[str] = []
    info: list[str] = []
    ctx, fs = _model_ctx(worker)
    if fs:
        return False, fs, info
    model, device, T, patch_size, _H, _W = ctx
    if T <= 1:
        info.append("skipped: video_temporal_patch_size <= 1")
        return True, failures, info
    if not _video_embedder_loaded(model):
        info.append("skipped: video_embedder weights not loaded")
        return True, failures, info

    # For these inputs, no single video exceeds the 32768-patch target,
    # so the whole-video bin-packer must produce >1 microbatch.
    # Patches-per-frame is H_patches * W_patches (= 16*16 = 256 or
    # 20*16 = 320 for the sizes used here), so the chosen frame counts
    # overshoot a single microbatch in aggregate:
    #   video1  96 frames, 16x16 patches/frame -> 24576 patches
    #   video2  40 frames, 20x16 patches/frame -> 12800 patches
    #   video3   8 frames, 16x20 patches/frame ->  2560 patches
    H1, W1 = patch_size * 16, patch_size * 16
    H2, W2 = patch_size * 20, patch_size * 16
    H3, W3 = patch_size * 16, patch_size * 20
    videos = [
        _make_video(device, 96, H1, W1, seed=SEED + 0),
        _make_video(device, 40, H2, W2, seed=SEED + 1),
        _make_video(device, 8, H3, W3, seed=SEED + 2),
    ]
    nfs = [v.shape[0] for v in videos]
    hidden_size = model.config.text_config.hidden_size
    total_patches = 96 * 16 * 16 + 40 * 20 * 16 + 8 * 16 * 20
    info.append(
        f"dyn_boundary nfs={nfs} total_patches={total_patches} "
        f"(>32768 forces split across microbatches)"
    )

    with torch.no_grad():
        solos = [
            model._extract_video_embeddings_temporal_dynamic(
                pixel_values=[v],
                num_frames_per_video=[v.shape[0]],
                hidden_size=hidden_size,
                T=T,
                patch_size=patch_size,
            )[0]
            for v in videos
        ]
        batched = model._extract_video_embeddings_temporal_dynamic(
            pixel_values=videos,
            num_frames_per_video=nfs,
            hidden_size=hidden_size,
            T=T,
            patch_size=patch_size,
        )
    _compare_tensor_lists(
        "dyn_boundary", solos, batched, REL_TOL_DYNAMIC, failures, info
    )
    return not failures, failures, info


def _w_process_video_input_strict(worker):
    failures: list[str] = []
    info: list[str] = []
    ctx, fs = _model_ctx(worker)
    if fs:
        return False, fs, info
    model, device, T, _patch_size, H, W = ctx
    if T <= 1:
        info.append("skipped: video_temporal_patch_size <= 1")
        return True, failures, info

    nf = 8
    videos = [
        _make_video(device, nf, H, W, seed=SEED + 0),
        _make_video(device, nf, H, W, seed=SEED + 1),
    ]
    with torch.no_grad():
        solo1 = model._process_video_input(_full_same_res_input([videos[0]]))
        solo2 = model._process_video_input(_full_same_res_input([videos[1]]))
        batched = model._process_video_input(_full_same_res_input(videos))
    _compare_tensor_lists(
        "_process_video_input",
        [solo1[0], solo2[0]],
        batched,
        REL_TOL_SAME_RES,
        failures,
        info,
    )
    return not failures, failures, info


# ---------------------------------------------------------------------------
# Fixture + shared assertion helper
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nemotron_nano_vl_v2_llm():
    from vllm import LLM

    # collective_rpc needs to ship a callable into the worker; vLLM only
    # allows pickle-based serialization when this env var is set.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=TP,
        dtype="bfloat16",
        max_model_len=32768,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"video": 4, "audio": 1, "image": 16},
        mamba_ssm_cache_dtype="float32",
    )
    yield llm
    del llm


def _assert_all_workers_pass(results, label: str) -> None:
    """Fail if ANY TP worker reports failures.

    Prior versions only inspected `results[0]`, which silently hid
    rank-local divergences (e.g. a shape bug that only fires on ranks
    with a non-trivial TP shard). Print every worker's info lines before
    asserting so the first failing rank is obvious from pytest output.
    """
    aggregated: list[str] = []
    for rank, result in enumerate(results):
        ok, failures, info = result
        for line in info:
            print(f"[{label} rank={rank}] {line}")
        if not ok:
            aggregated.append(f"rank {rank}: " + " | ".join(failures))
    assert not aggregated, (
        f"{label} failed on {len(aggregated)}/{len(results)} worker(s):\n  "
        + "\n  ".join(aggregated)
    )


# ---------------------------------------------------------------------------
# Public tests -- GPU + checkpoint required
# ---------------------------------------------------------------------------


@needs_model
@needs_gpus
def test_video_same_res_batched_equals_solo(nemotron_nano_vl_v2_llm):
    """Same-resolution videos batched together: each per-video embedding
    equals its solo reference.

    Covers batch-of-2 and batch-of-4 through
    `_extract_video_embeddings_temporal_same_resolution`, the fast path
    with frame-budgeted encoder microbatches.
    """
    _assert_all_workers_pass(
        nemotron_nano_vl_v2_llm.collective_rpc(_w_same_res_batched_equals_solo),
        "same_res_batched",
    )


@needs_model
@needs_gpus
def test_video_dynamic_res_batched_equals_solo(nemotron_nano_vl_v2_llm):
    """Different-resolution videos batched together: each per-video
    embedding equals its solo reference via
    `_extract_video_embeddings_temporal_dynamic` (patch-budgeted
    microbatches).

    Also asserts that list-shaped ``pixel_values_flat`` dispatched
    through the top-level `_extract_video_embeddings_temporal` routes to
    the dynamic path and returns one output per video.

    Uses ``REL_TOL_DYNAMIC`` because the dynamic path concatenates
    patches across items into a single ViT forward; any deviation above
    that ceiling almost certainly means per-item attention masking
    leaked or broke.
    """
    _assert_all_workers_pass(
        nemotron_nano_vl_v2_llm.collective_rpc(_w_dynamic_res_batched_equals_solo),
        "dynamic_res_batched",
    )


@needs_model
@needs_gpus
def test_video_packed_tensor_with_num_patches_splits_correctly(
    nemotron_nano_vl_v2_llm,
):
    """Regression for the 'truncated to one iteration' bug: when the
    scheduler hands us a single concatenated ``pixel_values_flat`` tensor
    with ``num_patches=[n1, n2, n3]``, the encoder must emit exactly 3
    outputs in order, each matching its solo reference.
    """
    _assert_all_workers_pass(
        nemotron_nano_vl_v2_llm.collective_rpc(_w_packed_tensor_splits),
        "packed_tensor_splits",
    )


@needs_model
@needs_gpus
def test_video_microbatch_boundaries_same_res(nemotron_nano_vl_v2_llm):
    """Same-resolution fast path at/around the 128-frame microbatch cap.

    Verifies:
    * odd ``nf`` (tubelet padding) keeps the
      ``n_tokens % num_tubelets == 0`` invariant that
      `_process_video_input` relies on;
    * ``nf`` = 127 / 128 / 129 all produce correctly-shaped outputs;
    * both cross-boundary cases (``nf=127`` and ``nf=129``) match their
      solo embedding when batched with a short companion, i.e.
      same-resolution microbatch boundaries are safe.
    """
    _assert_all_workers_pass(
        nemotron_nano_vl_v2_llm.collective_rpc(_w_microbatch_boundaries_same_res),
        "microbatch_boundaries_same_res",
    )


@needs_model
@needs_gpus
def test_video_same_res_companion_invariance(nemotron_nano_vl_v2_llm):
    """Bit-exact leakage probe under fixed execution shape.

    Same-res solo-vs-batched comparisons can drift in bf16 because the
    encoder packs different total token counts in each scenario. That
    drift makes solo-vs-batched a poor leakage probe — small numerical
    differences are *expected* even with perfect masking. This test
    instead holds the batch shape constant and varies only a companion
    video's pixel content; any non-bit-exact change in the queried
    video's output is then provably a cross-video leak, since shape and
    masking are identical between runs.

    Covered:
    * queried = long (129 frames, spans the 128-frame microbatch
      boundary), companion = short (8 frames). Long's output must be
      bit-exact across companion changes, and short's output must be
      bit-exact across long changes.
    """
    _assert_all_workers_pass(
        nemotron_nano_vl_v2_llm.collective_rpc(_w_same_res_companion_invariance),
        "same_res_companion_invariance",
    )


@needs_model
@needs_gpus
def test_video_microbatch_boundaries_dynamic_res(nemotron_nano_vl_v2_llm):
    """Different-resolution videos forced across >1 dynamic microbatch.

    The dynamic path greedily groups whole videos by patch budget,
    targeting 32768 patches per microbatch. Constructed inputs push the
    total patch count over that budget so the greedy bin-packer groups
    different videos into different microbatches; each video's embedding
    must still equal its solo reference, i.e. dynamic-resolution
    microbatch boundaries are safe. (Whole-video grouping only: a single
    video that exceeds the budget is not split.)
    """
    _assert_all_workers_pass(
        nemotron_nano_vl_v2_llm.collective_rpc(_w_microbatch_boundaries_dynamic_res),
        "microbatch_boundaries_dynamic_res",
    )


@needs_model
@needs_gpus
def test_process_video_input_batched_equals_solo_strict_shapes(
    nemotron_nano_vl_v2_llm,
):
    """`_process_video_input` (EVS pruning + final-embedding construction)
    must be batch-invariant.

    Covers the same-resolution case only. ``_process_video_input`` reads
    ``pixel_values.shape[-2:]`` as a single (H, W) pair, so it is not
    structured to accept list-shaped varying-resolution input.
    Regressions in the dynamic post-encoder path have to be caught
    upstream by ``test_video_dynamic_res_batched_equals_solo``, or in a
    follow-up test once ``_process_video_input`` supports list input.
    """
    _assert_all_workers_pass(
        nemotron_nano_vl_v2_llm.collective_rpc(_w_process_video_input_strict),
        "process_video_input_strict",
    )


# ---------------------------------------------------------------------------
# CPU-only: runs on the driver process, no LLM/GPU required
# ---------------------------------------------------------------------------


def test_mm_encoder_cache_mapping_after_modality_sort():
    """Mirror of the permute-then-cache-by-hash loop in
    `gpu_model_runner._execute_mm_encoder`:

        perm = sorted(range(len(mm_kwargs)), key=lambda i: mm_kwargs[i][0])
        mm_kwargs = [mm_kwargs[i] for i in perm]
        mm_hashes = [mm_hashes[i] for i in perm]
        mm_lora_refs = [mm_lora_refs[i] for i in perm]
        ...
        for mm_hash, output in zip(mm_hashes, encoder_outputs):
            self.encoder_cache[mm_hash] = output

    Each simulated encoder output is tagged with the modality + value of
    the item that produced it, so the final cache must associate every
    hash with the output produced from its own item. If the permutation
    is ever applied to one of the parallel lists but not another, this
    test fails instead of silently caching embeddings under the wrong
    request's hash.
    """
    mm_kwargs = [("video", 1), ("audio", 2), ("video", 3), ("audio", 4)]
    mm_hashes = ["h_v1", "h_a2", "h_v3", "h_a4"]
    mm_refs = ["r_v1", "r_a2", "r_v3", "r_a4"]

    perm = sorted(range(len(mm_kwargs)), key=lambda i: mm_kwargs[i][0])
    perm_kwargs = [mm_kwargs[i] for i in perm]
    perm_hashes = [mm_hashes[i] for i in perm]
    perm_refs = [mm_refs[i] for i in perm]

    # The sort must actually reorder; otherwise this test no longer
    # exercises the permutation path.
    assert perm_kwargs != mm_kwargs, (
        "sort-by-modality did not reorder the interleaved input"
    )
    # Items of the same modality must be consecutive after sorting,
    # which is what enables `group_and_batch_mm_kwargs` to batch across
    # requests.
    modalities = [m for m, _ in perm_kwargs]
    assert modalities == sorted(modalities), (
        f"same-modality items not consecutive after sort: {modalities}"
    )

    # Simulate `embed_multimodal(**mm_kwargs_batch)` producing one output
    # per item, tagged with the item's modality + value so we can detect
    # cross-assignment.
    encoder_outputs = [f"emb_{m[0]}{v}" for (m, v) in perm_kwargs]

    cache: dict[str, str] = {}
    for mm_hash, output in zip(perm_hashes, encoder_outputs):
        cache[mm_hash] = output

    assert cache == {
        "h_v1": "emb_v1",
        "h_v3": "emb_v3",
        "h_a2": "emb_a2",
        "h_a4": "emb_a4",
    }, f"cache hash->output mapping diverged: {cache}"

    # `mm_lora_refs` is permuted alongside; spot-check it stays aligned.
    for i, (mod, val) in enumerate(perm_kwargs):
        expected = f"r_{mod[0]}{val}"
        assert perm_refs[i] == expected, f"ref[{i}]={perm_refs[i]!r} != {expected!r}"
