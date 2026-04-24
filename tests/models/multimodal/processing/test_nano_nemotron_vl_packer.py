# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure unit tests for Nano-Nemotron-VL's video segment microbatch packer."""

import math

import pytest

from vllm.model_executor.models.nano_nemotron_vl import (
    _pack_video_segments_into_microbatches,
    _VideoSegment,
)


def _bill(nf: int, T: int) -> int:
    return math.ceil(nf / T) * T


def _assert_microbatches_valid(
    microbatches: list[list[_VideoSegment]],
    num_frames_per_video: list[int],
    micro_batch_size: int,
    T: int,
) -> None:
    """Assert structural invariants that must hold for every packer output."""
    # Cap: no microbatch exceeds the billed budget.
    for mb in microbatches:
        billed = sum(_bill(s.num_frames, T) for s in mb)
        assert billed <= micro_batch_size, (
            f"microbatch billed {billed} exceeds cap {micro_batch_size}: {mb}"
        )

    # Per-video reconstruction: segments cover [0, nf) contiguously in order.
    segs_by_video: dict[int, list[_VideoSegment]] = {}
    for mb in microbatches:
        for s in mb:
            segs_by_video.setdefault(s.video_idx, []).append(s)

    for video_idx, nf in enumerate(num_frames_per_video):
        segs = segs_by_video.get(video_idx, [])
        assert segs, f"video {video_idx} (nf={nf}) produced no segments"

        cursor = 0
        for s in segs:
            assert s.frame_start == cursor, (
                f"video {video_idx} segments are not contiguous: "
                f"expected start={cursor}, got {s.frame_start}"
            )
            cursor += s.num_frames
        assert cursor == nf, (
            f"video {video_idx} reconstructed length {cursor} != nf {nf}"
        )

        # Non-final segments must be multiples of T.
        for s in segs[:-1]:
            assert s.num_frames % T == 0, f"non-final segment {s} has nf%T != 0"

    # Video-major global order: a segment for video i never appears in a
    # strictly later microbatch than a segment for video j > i.
    last_mb_idx_by_video: dict[int, int] = {}
    for mb_idx, mb in enumerate(microbatches):
        for s in mb:
            last_mb_idx_by_video[s.video_idx] = mb_idx
    first_mb_idx_by_video: dict[int, int] = {}
    for mb_idx, mb in enumerate(microbatches):
        for s in mb:
            first_mb_idx_by_video.setdefault(s.video_idx, mb_idx)

    video_indices = sorted(last_mb_idx_by_video.keys())
    for i, j in zip(video_indices, video_indices[1:]):
        assert last_mb_idx_by_video[i] <= first_mb_idx_by_video[j], (
            f"video {i} appears after video {j} (not video-major)"
        )


def test_empty_input() -> None:
    assert _pack_video_segments_into_microbatches([], micro_batch_size=128, T=4) == []


def test_single_atomic_video_fits() -> None:
    microbatches = _pack_video_segments_into_microbatches(
        [50], micro_batch_size=128, T=4
    )
    assert microbatches == [[_VideoSegment(0, 0, 50)]]
    _assert_microbatches_valid(microbatches, [50], 128, 4)


def test_two_videos_share_microbatch_in_order() -> None:
    microbatches = _pack_video_segments_into_microbatches(
        [40, 60], micro_batch_size=128, T=4
    )
    assert microbatches == [[_VideoSegment(0, 0, 40), _VideoSegment(1, 0, 60)]]
    _assert_microbatches_valid(microbatches, [40, 60], 128, 4)


def test_two_videos_exceed_cap_split_into_two_microbatches() -> None:
    # 100 + 100 = 200 billed > 128 -> two microbatches, each with one atomic video.
    microbatches = _pack_video_segments_into_microbatches(
        [100, 100], micro_batch_size=128, T=4
    )
    assert len(microbatches) == 2
    assert microbatches[0] == [_VideoSegment(0, 0, 100)]
    assert microbatches[1] == [_VideoSegment(1, 0, 100)]
    _assert_microbatches_valid(microbatches, [100, 100], 128, 4)


def test_exact_cap_video_occupies_microbatch_alone() -> None:
    microbatches = _pack_video_segments_into_microbatches(
        [128, 8], micro_batch_size=128, T=4
    )
    assert len(microbatches) == 2
    assert microbatches[0] == [_VideoSegment(0, 0, 128)]
    assert microbatches[1] == [_VideoSegment(1, 0, 8)]
    _assert_microbatches_valid(microbatches, [128, 8], 128, 4)


def test_long_video_splits_across_microbatches() -> None:
    # nf=200 at cap=128, T=4:
    #   non-final chunk of 128 frames  (billed 128)
    #   final tail of 72 frames         (billed 72)
    microbatches = _pack_video_segments_into_microbatches(
        [200], micro_batch_size=128, T=4
    )
    assert microbatches == [
        [_VideoSegment(0, 0, 128)],
        [_VideoSegment(0, 128, 72)],
    ]
    _assert_microbatches_valid(microbatches, [200], 128, 4)


def test_long_video_joins_tail_of_partial_microbatch() -> None:
    # video 0 uses 30 frames (billed 32); video 1 has 250 frames (billed 252).
    # Expected:
    #   mb 0: [video 0, non-final chunk of video 1 using (128-32)//4 * 4 = 96]
    #   mb 1: [non-final chunk of video 1 = 128]
    #   mb 2: [final tail of video 1 = 250 - 96 - 128 = 26]
    microbatches = _pack_video_segments_into_microbatches(
        [30, 250], micro_batch_size=128, T=4
    )
    assert microbatches == [
        [_VideoSegment(0, 0, 30), _VideoSegment(1, 0, 96)],
        [_VideoSegment(1, 96, 128)],
        [_VideoSegment(1, 224, 26)],
    ]
    _assert_microbatches_valid(microbatches, [30, 250], 128, 4)


def test_odd_final_segment_is_billed_with_ceil() -> None:
    # T=4, nf=51 -> final segment billed ceil(51/4)*4 = 52.
    microbatches = _pack_video_segments_into_microbatches(
        [51], micro_batch_size=128, T=4
    )
    assert microbatches == [[_VideoSegment(0, 0, 51)]]
    _assert_microbatches_valid(microbatches, [51], 128, 4)

    # Pair a 51-frame video with one that takes the rest minus the pad.
    # 52 + 76 = 128 exactly; both are atomic, same microbatch.
    microbatches = _pack_video_segments_into_microbatches(
        [51, 76], micro_batch_size=128, T=4
    )
    assert microbatches == [[_VideoSegment(0, 0, 51), _VideoSegment(1, 0, 76)]]
    _assert_microbatches_valid(microbatches, [51, 76], 128, 4)


def test_many_tiny_clips_use_padded_billing() -> None:
    # 128 videos of 1 frame each, T=4. Each bills 4 frames -> 32 videos fit
    # per microbatch of 128, so we expect exactly 4 microbatches.
    num_frames = [1] * 128
    microbatches = _pack_video_segments_into_microbatches(
        num_frames, micro_batch_size=128, T=4
    )
    assert len(microbatches) == 4
    assert all(len(mb) == 32 for mb in microbatches)
    _assert_microbatches_valid(microbatches, num_frames, 128, 4)


def test_non_positive_frame_counts_raise() -> None:
    with pytest.raises(ValueError):
        _pack_video_segments_into_microbatches([0], micro_batch_size=128, T=4)
    with pytest.raises(ValueError):
        _pack_video_segments_into_microbatches([20, 0], micro_batch_size=128, T=4)
    with pytest.raises(ValueError):
        _pack_video_segments_into_microbatches([-1], micro_batch_size=128, T=4)
    with pytest.raises(ValueError):
        _pack_video_segments_into_microbatches([10, -5, 20], micro_batch_size=128, T=4)


def test_tubelet_size_one_behaves_as_raw_frames() -> None:
    # T=1 means no padding, billing == raw frames.
    microbatches = _pack_video_segments_into_microbatches(
        [50, 50, 50], micro_batch_size=128, T=1
    )
    # 50 + 50 = 100 fits; +50 = 150 > 128 -> flush.
    assert microbatches == [
        [_VideoSegment(0, 0, 50), _VideoSegment(1, 0, 50)],
        [_VideoSegment(2, 0, 50)],
    ]
    _assert_microbatches_valid(microbatches, [50, 50, 50], 128, 1)


def test_very_long_video_across_many_microbatches() -> None:
    # Three full microbatches + a tail, with no other videos in play.
    nf = 3 * 128 + 17  # 401 frames, T=4 -> final billed 20
    microbatches = _pack_video_segments_into_microbatches(
        [nf], micro_batch_size=128, T=4
    )
    assert len(microbatches) == 4
    # Every non-final chunk is exactly 128 frames.
    for mb in microbatches[:-1]:
        assert len(mb) == 1
        assert mb[0].num_frames == 128
    # Final tail is the remainder.
    assert microbatches[-1] == [_VideoSegment(0, 3 * 128, 17)]
    _assert_microbatches_valid(microbatches, [nf], 128, 4)


def test_invalid_cap_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _pack_video_segments_into_microbatches([10], micro_batch_size=0, T=4)
    with pytest.raises(ValueError):
        _pack_video_segments_into_microbatches([10], micro_batch_size=-128, T=4)
    with pytest.raises(ValueError):
        _pack_video_segments_into_microbatches([10], micro_batch_size=130, T=4)
    with pytest.raises(ValueError):
        _pack_video_segments_into_microbatches([10], micro_batch_size=128, T=0)
    with pytest.raises(ValueError):
        _pack_video_segments_into_microbatches([10], micro_batch_size=128, T=-1)


@pytest.mark.parametrize(
    "num_frames_per_video,micro_batch_size,T",
    [
        ([], 128, 4),
        ([1], 128, 4),
        ([128], 128, 4),
        ([129], 128, 4),
        ([1, 1, 1, 1], 128, 4),
        ([30, 30, 30, 30, 30], 128, 4),
        ([500], 128, 4),
        ([500, 500], 128, 4),
        ([30, 250], 128, 4),
        ([51, 51, 51], 128, 4),
        ([1] * 200, 128, 4),
        ([100] * 10, 128, 4),
        ([128, 1, 127, 1], 128, 4),
        ([64, 64, 64, 64], 128, 4),
        # Non-default T values.
        ([31, 63, 128], 256 - (256 % 2), 2),
        ([100, 200, 300], 252, 6),
    ],
)
def test_invariants_hold_across_inputs(
    num_frames_per_video: list[int],
    micro_batch_size: int,
    T: int,
) -> None:
    microbatches = _pack_video_segments_into_microbatches(
        num_frames_per_video, micro_batch_size, T
    )
    _assert_microbatches_valid(microbatches, num_frames_per_video, micro_batch_size, T)
