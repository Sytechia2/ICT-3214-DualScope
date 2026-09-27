"""GRU sequence autoencoder and masked reconstruction objective (Task 3.2).

The encoder reads a user-hour chunk and compresses its final hidden state to
a latent vector. The decoder receives only that latent vector at every step,
so it cannot copy its inputs, and must reconstruct each event:

* numeric Task 2.4 inputs (standardised log counts and gaps): squared error;
* binary inputs (history and novelty flags): binary cross-entropy;
* categorical vocabulary IDs (authentication/logon type, orientation, result):
  cross-entropy.

An event's reconstruction error is the weighted mean of its per-feature losses.
Padded positions are removed by the mask and contribute exactly zero loss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from dualscope.sequence.config import ModelSettings


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to rebuild the network from a checkpoint."""

    n_numeric: int
    n_binary: int
    categorical_cardinalities: tuple[int, ...]
    hidden_size: int
    latent_size: int
    num_layers: int
    embedding_dim: int
    dropout: float
    numeric_weight: float
    binary_weight: float
    categorical_weight: float

    @classmethod
    def from_settings(
        cls,
        settings: ModelSettings,
        n_numeric: int,
        n_binary: int,
        categorical_cardinalities: tuple[int, ...],
    ) -> ModelSpec:
        return cls(
            n_numeric=n_numeric,
            n_binary=n_binary,
            categorical_cardinalities=tuple(int(c) for c in categorical_cardinalities),
            hidden_size=settings.hidden_size,
            latent_size=settings.latent_size,
            num_layers=settings.num_layers,
            embedding_dim=settings.embedding_dim,
            dropout=settings.dropout,
            numeric_weight=settings.numeric_weight,
            binary_weight=settings.binary_weight,
            categorical_weight=settings.categorical_weight,
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["categorical_cardinalities"] = list(self.categorical_cardinalities)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ModelSpec:
        values = dict(data)
        values["categorical_cardinalities"] = tuple(values["categorical_cardinalities"])
        return cls(**values)

    @property
    def n_features(self) -> int:
        return self.n_numeric + self.n_binary + len(self.categorical_cardinalities)


class GRUSequenceAutoencoder(nn.Module):
    """Encoder-decoder GRU reconstructing a variable-length event sequence."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, spec.embedding_dim) for cardinality in spec.categorical_cardinalities
        )
        input_size = spec.n_numeric + spec.n_binary + spec.embedding_dim * len(spec.categorical_cardinalities)
        recurrent_dropout = spec.dropout if spec.num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size, spec.hidden_size, spec.num_layers, batch_first=True, dropout=recurrent_dropout
        )
        self.to_latent = nn.Linear(spec.hidden_size, spec.latent_size)
        self.latent_to_hidden = nn.Linear(spec.latent_size, spec.hidden_size * spec.num_layers)
        self.decoder = nn.GRU(
            spec.latent_size, spec.hidden_size, spec.num_layers, batch_first=True, dropout=recurrent_dropout
        )
        self.numeric_head = nn.Linear(spec.hidden_size, spec.n_numeric)
        self.binary_head = nn.Linear(spec.hidden_size, spec.n_binary)
        self.categorical_heads = nn.ModuleList(
            nn.Linear(spec.hidden_size, cardinality) for cardinality in spec.categorical_cardinalities
        )

    def forward(
        self, dense: torch.Tensor, categorical: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Return numeric predictions, binary logits and categorical logits per step."""
        batch, steps, _ = dense.shape
        embedded = [emb(categorical[:, :, i]) for i, emb in enumerate(self.embeddings)]
        inputs = torch.cat([dense, *embedded], dim=-1)
        cpu_lengths = lengths.to("cpu", torch.int64)
        packed = pack_padded_sequence(inputs, cpu_lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        latent = torch.tanh(self.to_latent(hidden[-1]))

        initial = torch.tanh(self.latent_to_hidden(latent))
        initial = initial.view(batch, self.spec.num_layers, self.spec.hidden_size).transpose(0, 1).contiguous()
        repeated = latent.unsqueeze(1).expand(batch, steps, self.spec.latent_size)
        packed_decoder = pack_padded_sequence(repeated, cpu_lengths, batch_first=True, enforce_sorted=False)
        decoded, _ = self.decoder(packed_decoder, initial)
        decoded, _ = pad_packed_sequence(decoded, batch_first=True, total_length=steps)

        numeric = self.numeric_head(decoded)
        binary_logits = self.binary_head(decoded)
        categorical_logits = [head(decoded) for head in self.categorical_heads]
        return numeric, binary_logits, categorical_logits


def per_feature_errors(
    model: GRUSequenceAutoencoder,
    dense: torch.Tensor,
    categorical: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Return weighted per-event, per-feature reconstruction losses ``[B, T, F]``.

    Feature order is numeric, binary, then categorical (the order of
    ``dualscope.sequence.config.input_feature_names``). Padded steps are zero.
    """
    spec = model.spec
    numeric_hat, binary_logits, categorical_logits = model(dense, categorical, lengths)
    numeric_target = dense[:, :, : spec.n_numeric]
    binary_target = dense[:, :, spec.n_numeric : spec.n_numeric + spec.n_binary]
    numeric_loss = (numeric_hat - numeric_target).pow(2) * spec.numeric_weight
    binary_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_target, reduction="none") * spec.binary_weight
    categorical_loss = torch.stack(
        [
            F.cross_entropy(logits.transpose(1, 2), categorical[:, :, i], reduction="none")
            for i, logits in enumerate(categorical_logits)
        ],
        dim=-1,
    ) * spec.categorical_weight
    errors = torch.cat([numeric_loss, binary_loss, categorical_loss], dim=-1)
    steps = dense.shape[1]
    mask = torch.arange(steps, device=dense.device)[None, :] < lengths[:, None]
    return errors * mask.unsqueeze(-1)


def event_errors(feature_errors: torch.Tensor) -> torch.Tensor:
    """Per-event reconstruction error: the mean of its per-feature losses."""
    return feature_errors.mean(dim=-1)


def masked_sequence_loss(feature_errors: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean event error over all valid (unpadded) events in the batch."""
    return event_errors(feature_errors).sum() / lengths.sum().clamp_min(1)
