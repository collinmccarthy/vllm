# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`ViTPatchGenerator._video_embedder_loaded` wiring.

`forward_video_dynamic` guards against running with an untrained
video_embedder by checking `self._video_embedder_loaded`. The flag must
be initialised by `ViTPatchGenerator.__init__` (when temporal
compression is enabled) and flipped to True by `RadioModel.load_weights`
when `model.patch_generator.video_embedder.*` weights are loaded.

Without this wiring the guard either AttributeError's at runtime or
silently masks integration tests via `getattr(..., False)`. These tests
pin both halves so a future refactor can't quietly break either.
"""

import torch
import torch.nn as nn

from vllm.model_executor.models.radio import RadioModel, ViTPatchGenerator


def test_init_marks_video_embedder_unloaded_when_temporal_compression_on():
    pg = ViTPatchGenerator(
        patch_size=14,
        embed_dim=32,
        input_dims=224,
        temporal_patch_size=4,
    )
    assert pg._video_embedder_loaded is False


def test_init_omits_flag_when_temporal_compression_off():
    # Without temporal compression there is no video_embedder to load,
    # and forward_video_dynamic is never reached. Guard the absence so
    # we don't accidentally start always-allocating the flag (which
    # would mask "guard never reached" semantics in the future).
    pg = ViTPatchGenerator(
        patch_size=14,
        embed_dim=32,
        input_dims=224,
        temporal_patch_size=1,
    )
    assert not hasattr(pg, "_video_embedder_loaded")


def _make_radio_model_with_pg() -> tuple[RadioModel, ViTPatchGenerator]:
    """Build the minimal RadioModel surface that `load_weights` touches:
    a top-level module with `self.model.patch_generator` set to a real
    ViTPatchGenerator (so its parameters appear in named_parameters)."""
    radio = RadioModel.__new__(RadioModel)
    nn.Module.__init__(radio)
    radio.model = nn.Module()
    pg = ViTPatchGenerator(
        patch_size=14,
        embed_dim=32,
        input_dims=224,
        temporal_patch_size=4,
    )
    radio.model.patch_generator = pg
    return radio, pg


def test_load_weights_flips_flag_when_video_embedder_weight_arrives():
    radio, pg = _make_radio_model_with_pg()
    assert pg._video_embedder_loaded is False

    # The HF source key is `radio_model.model.patch_generator.video_embedder.weight`;
    # load_weights strips the `radio_model.` prefix and remaps. The
    # weight shape mirrors ViTPatchLinear(patch_size=14, embed_dim=32,
    # temporal_patch_size=4): out=32, in=3*4*14*14.
    weight = torch.zeros(32, 3 * 4 * 14 * 14)
    radio.load_weights(
        [("radio_model.model.patch_generator.video_embedder.weight", weight)]
    )

    assert pg._video_embedder_loaded is True


def test_load_weights_keeps_flag_false_when_no_video_embedder_weight():
    radio, pg = _make_radio_model_with_pg()

    # Loading a non-video_embedder weight (here: the still-image
    # embedder) must not flip the guard. Shape mirrors
    # ViTPatchLinear(patch_size=14, embed_dim=32) without temporal
    # dimension: out=32, in=3*14*14.
    weight = torch.zeros(32, 3 * 14 * 14)
    radio.load_weights([("radio_model.model.patch_generator.embedder.weight", weight)])

    assert pg._video_embedder_loaded is False
