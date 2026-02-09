"""Decoder-only transformer with optional HTP-style between-layer rewiring.

Both the flat and hierarchical models use the exact same ArithmeticTransformer.
The only difference at runtime is whether rewire_src/rewire_dst are passed:

  Flat:         model(tokens)                    -- standard causal transformer
  Hierarchical: model(tokens, rewire_src=..., rewire_dst=...)  -- HTP rewiring

The rewiring copies hidden states from the last digit of each number group
to the corresponding <num> summary token position, between transformer layers.
This lets the transformer's own attention do the composition -- no separate
learned module needed.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class ArithmeticTransformer(nn.Module):
    """Decoder-only transformer with optional HTP-style rewiring.

    Architecture: token embedding -> positional encoding -> N transformer layers -> LN -> projection

    When rewire_src and rewire_dst are provided, the model copies hidden states
    from src positions (last digit of each number) to dst positions (<num> tokens)
    between layers 0 and 1, 1 and 2, etc. This is the "local prepending" mechanism
    from the HTP paper -- the transformer's own representations do the composition.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        d_ff: int = 256,
        max_seq_len: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor = None,
        rewire_src: torch.Tensor = None,
        rewire_dst: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: [batch, seq_len] token IDs.
            padding_mask: [batch, seq_len] True for padding positions.
            rewire_src: [batch, num_rewires] source positions (last digit of each number).
            rewire_dst: [batch, num_rewires] destination positions (<num> token positions).

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        seq_len = tokens.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=tokens.device,
        )

        x = self.token_embed(tokens) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        for i, layer in enumerate(self.layers):
            # HTP rewiring between layers (skip before the first layer, like the paper)
            if i > 0 and rewire_src is not None:
                x = _rewire(x, rewire_src, rewire_dst)

            x = layer(x, src_mask=causal_mask, src_key_padding_mask=padding_mask, is_causal=True)

        x = self.final_norm(x)
        return self.output_proj(x)


def _rewire(
    x: torch.Tensor,
    src_positions: torch.Tensor,
    dst_positions: torch.Tensor,
) -> torch.Tensor:
    """Copy hidden states from src to dst positions using gather.

    For each batch element, builds an index that is the identity everywhere
    except at dst positions, where it points to the corresponding src position.
    Then gathers from x using that index.
    """
    batch_size, seq_len, d_model = x.shape

    # Identity index: position j maps to j
    index = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1).clone()

    # Overwrite dst positions to point at src positions
    batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(1).expand_as(dst_positions)
    index[batch_idx, dst_positions] = src_positions

    return x.gather(1, index.unsqueeze(-1).expand(-1, -1, d_model))
