"""Task 3.2: GRU autoencoder masking, training, checkpointing and reload."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from dualscope.sequence.builder import build_chunks, gather_padded, load_day_sequences
from dualscope.sequence.config import SequenceDetectorConfig, SequencePolicy, input_feature_names
from dualscope.sequence.model import GRUSequenceAutoencoder, ModelSpec, masked_sequence_loss, per_feature_errors
from dualscope.sequence.scoring import score_chunks
from dualscope.sequence.training import (
    categorical_cardinalities,
    load_checkpoint,
    sample_chunks,
    save_checkpoint,
    train_autoencoder,
)


CONFIG = SequenceDetectorConfig().with_trial(8, "small")


def _spec(data) -> ModelSpec:
    return ModelSpec.from_settings(CONFIG.model, 5, 3, categorical_cardinalities(data.preprocessing))


def _days(data, days):
    return [
        load_day_sequences(data.features_dir, d, data.split_cfg, SequencePolicy(max_sequence_length=8), data.excluded_users)
        for d in days
    ]


def test_padded_positions_contribute_zero_loss(synthetic_sequences) -> None:
    torch.manual_seed(0)
    model = GRUSequenceAutoencoder(_spec(synthetic_sequences))
    day = _days(synthetic_sequences, [2])[0]
    chunks = build_chunks(day.offsets, day.counts, 8)
    pick = np.argsort(chunks.length)[[0, len(chunks) // 2, -1]]
    dense, cat, mask = gather_padded(day.numeric, day.categorical, chunks.start[pick], chunks.length[pick])
    lengths = torch.from_numpy(chunks.length[pick].astype(np.int64))
    assert (~mask).any(), "the test batch must contain padding"

    errors = per_feature_errors(model, torch.from_numpy(dense), torch.from_numpy(cat), lengths)
    assert errors.shape == (*mask.shape, len(input_feature_names()))
    assert torch.all(errors[torch.from_numpy(~mask)] == 0)

    noisy_dense, noisy_cat = dense.copy(), cat.copy()
    noisy_dense[~mask] = 123.0
    noisy_cat[~mask] = 1
    noisy = per_feature_errors(model, torch.from_numpy(noisy_dense), torch.from_numpy(noisy_cat), lengths)
    torch.testing.assert_close(errors, noisy)
    loss = masked_sequence_loss(errors, lengths)
    expected = errors.mean(dim=-1)[torch.from_numpy(mask)].mean()
    torch.testing.assert_close(loss, expected)


def test_training_saves_reloadable_checkpoint_with_finite_held_out_scores(synthetic_sequences, tmp_path) -> None:
    data = synthetic_sequences
    train_days = _days(data, [2])
    rng = np.random.default_rng(0)
    train, stats = sample_chunks(train_days, 8, 10_000, 2, rng, lambda d: d.fitting_eligible)
    monitor, _ = sample_chunks(_days(data, [3]), 8, 50, 2, np.random.default_rng(1), lambda d: d.scorable_mask())
    assert stats["chunks_sampled"] == len(train) > 0

    settings = replace(CONFIG.training, max_epochs=3, batch_size=16, early_stopping_patience=5)
    model, history = train_autoencoder(_spec(data), train, monitor, settings)
    assert len(history) == 3
    assert all(np.isfinite(h["train_loss"]) and np.isfinite(h["monitor_loss"]) for h in history)
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert sum(h["selected"] for h in history) == 1

    path = tmp_path / "checkpoint.pt"
    sha = save_checkpoint(path, model, {"seed": settings.seed, "history": history})
    reloaded, metadata = load_checkpoint(path, expected_sha256=sha)
    assert metadata["seed"] == settings.seed

    held_out = _days(data, [4])[0]
    chunks = build_chunks(held_out.offsets, held_out.counts, 8)
    before = {k: v.clone() for k, v in reloaded.state_dict().items()}
    original = score_chunks(model, held_out.numeric, held_out.categorical, chunks)[0]
    again = score_chunks(reloaded, held_out.numeric, held_out.categorical, chunks)[0]
    assert np.all(np.isfinite(again))
    np.testing.assert_allclose(original, again, rtol=0, atol=1e-6)
    assert all(torch.equal(before[k], v) for k, v in reloaded.state_dict().items())

    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(path, expected_sha256="0" * 64)


def test_training_is_reproducible_for_a_fixed_seed(synthetic_sequences) -> None:
    data = synthetic_sequences
    train, _ = sample_chunks(_days(data, [2]), 8, 200, 2, np.random.default_rng(0), lambda d: d.fitting_eligible)
    settings = replace(CONFIG.training, max_epochs=1, batch_size=32)
    first, h1 = train_autoencoder(_spec(data), train, train, settings)
    second, h2 = train_autoencoder(_spec(data), train, train, settings)
    assert h1[0]["train_loss"] == pytest.approx(h2[0]["train_loss"], abs=1e-9)
    for (k, a), (_, b) in zip(first.state_dict().items(), second.state_dict().items()):
        assert torch.allclose(a, b), k


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gpu_training_and_scoring_match_cpu(synthetic_sequences) -> None:
    data = synthetic_sequences
    train, _ = sample_chunks(_days(data, [2]), 8, 200, 2, np.random.default_rng(0), lambda d: d.fitting_eligible)
    settings = replace(CONFIG.training, max_epochs=1, batch_size=32)
    model, _ = train_autoencoder(_spec(data), train, train, settings, device="cuda")
    assert all(p.device.type == "cpu" for p in model.parameters())

    held_out = _days(data, [4])[0]
    chunks = build_chunks(held_out.offsets, held_out.counts, 8)
    on_cpu = score_chunks(model, held_out.numeric, held_out.categorical, chunks)
    on_gpu = score_chunks(model.to("cuda"), held_out.numeric, held_out.categorical, chunks)
    np.testing.assert_allclose(on_cpu[0], on_gpu[0], rtol=0, atol=1e-4)
    np.testing.assert_allclose(on_cpu[2], on_gpu[2], rtol=0, atol=1e-4)


def test_feature_scale_divides_each_feature_loss_before_averaging(synthetic_sequences) -> None:
    from dualscope.sequence.training import mean_feature_errors

    data = synthetic_sequences
    torch.manual_seed(0)
    model = GRUSequenceAutoencoder(_spec(data))
    day = _days(data, [3])[0]
    chunks = build_chunks(day.offsets, day.counts, 8)
    plain = score_chunks(model, day.numeric, day.categorical, chunks)
    ones = score_chunks(model, day.numeric, day.categorical, chunks, feature_scale=np.ones(model.spec.n_features))
    np.testing.assert_allclose(plain[2], ones[2], rtol=1e-6)  # float64 division; same values

    sample, _ = sample_chunks([day], 8, 50, 2, np.random.default_rng(0), lambda d: d.scorable_mask())
    scale = mean_feature_errors(model, sample)
    assert scale.shape == (model.spec.n_features,) and np.all(scale > 0)
    scaled = score_chunks(model, day.numeric, day.categorical, chunks, feature_scale=scale)

    d, c, mask = gather_padded(day.numeric, day.categorical, chunks.start[:3], chunks.length[:3])
    errors = per_feature_errors(model, torch.from_numpy(d), torch.from_numpy(c), torch.from_numpy(mask.sum(1))).detach().numpy()
    expected = (errors[0, : chunks.length[0]] / scale).mean(axis=-1)
    start = chunks.start[0]
    np.testing.assert_allclose(scaled[2][start : start + chunks.length[0]], expected, rtol=1e-5)
