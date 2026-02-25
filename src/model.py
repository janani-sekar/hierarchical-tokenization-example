"""Decoder-only transformers with optional HTP rewiring and hierarchical attention.

Two model classes share the same layer structure (embedding -> layers -> LN -> proj):

  ArithmeticTransformer  -- standard causal, optional HTP rewiring (variants 1,2,4,5)
  HierAttnTransformer    -- local+global attention split with HTP rewiring (variants 3,6)

Both classes have identical parameter counts when given the same hyperparameters.
"""

import math

import torch
import torch.nn as nn

# Disable flash/mem-efficient SDP on CUDA -- the T4 fast path has CUBLAS bugs
# with custom attention masks. Falls back to the math-based SDP which works fine.
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)


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


def _make_layers(num_layers, d_model, nhead, d_ff, dropout):
    """Create a ModuleList of TransformerEncoderLayers."""
    return nn.ModuleList([
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


class ArithmeticTransformer(nn.Module):
    """Decoder-only transformer with optional HTP-style rewiring.

    When rewire_src and rewire_dst are provided, copies hidden states from
    src positions to dst positions between every pair of consecutive layers.
    Used for variants: L2R, L2R+HTP, R2L, R2L+HTP.
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
        self.layers = _make_layers(num_layers, d_model, nhead, d_ff, dropout)
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
        seq_len = tokens.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=tokens.device,
        )

        x = self.token_embed(tokens) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        for i, layer in enumerate(self.layers):
            if i > 0 and rewire_src is not None:
                x = _rewire(x, rewire_src, rewire_dst)
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=padding_mask)

        x = self.final_norm(x)
        return self.output_proj(x)


class HierAttnTransformer(nn.Module):
    """Decoder-only transformer with local/global attention split and HTP rewiring.

    First half of layers (local): digits only attend within their number group.
    Second half (global): standard causal attention over all positions.
    HTP rewiring only happens between local layers.

    Used for variants: L2R+HTP+HAttn, R2L+HTP+HAttn.
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
        self.nhead = nhead
        self.n_local = num_layers // 2
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len)
        self.layers = _make_layers(num_layers, d_model, nhead, d_ff, dropout)
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
        group_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_len = tokens.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=tokens.device,
        )

        local_mask = _build_local_mask(group_ids, self.nhead) if group_ids is not None else causal_mask

        x = self.token_embed(tokens) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        for i, layer in enumerate(self.layers):
            if i < self.n_local:
                # Local layer: group-restricted attention + HTP rewiring
                if i > 0 and rewire_src is not None:
                    x = _rewire(x, rewire_src, rewire_dst)
                x = layer(x, src_mask=local_mask, src_key_padding_mask=padding_mask)
            else:
                # Global layer: standard causal attention, no rewiring
                x = layer(x, src_mask=causal_mask, src_key_padding_mask=padding_mask)

        x = self.final_norm(x)
        return self.output_proj(x)


# -- Helpers ------------------------------------------------------------------

def _rewire(
    x: torch.Tensor,
    src_positions: torch.Tensor,
    dst_positions: torch.Tensor,
) -> torch.Tensor:
    """Copy hidden states from src to dst positions using gather.

    For each batch element, builds an index that is the identity everywhere
    except at dst positions, where it points to the corresponding src position.
    """
    batch_size, seq_len, d_model = x.shape
    index = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1).clone()
    batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(1).expand_as(dst_positions)
    index[batch_idx, dst_positions] = src_positions
    return x.gather(1, index.unsqueeze(-1).expand(-1, -1, d_model))


def _build_local_mask(group_ids: torch.Tensor, nhead: int) -> torch.Tensor:
    """Build group-restricted causal attention mask.

    Position i can attend to j iff:
      1. j <= i  (causal)
      2. AND one of:
         a. group_ids[i] == -1  (unrestricted: answer / = / EOS tokens)
         b. group_ids[i] == group_ids[j]  (same group)
         c. j == 0  (BOS is always attendable)

    Returns: [batch*nhead, seq_len, seq_len] float mask (0=attend, -inf=block).
    """
    batch, seq_len = group_ids.shape
    device = group_ids.device

    # causal_ok[i, j] = True iff j <= i
    causal_ok = ~torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1,
    )

    gi = group_ids.unsqueeze(2)  # [B, S, 1]
    gj = group_ids.unsqueeze(1)  # [B, 1, S]
    same_group = (gi == gj)      # [B, S, S]

    unrestricted = (group_ids == -1).unsqueeze(2)  # [B, S, 1]

    bos_col = torch.zeros(1, 1, seq_len, device=device, dtype=torch.bool)
    bos_col[0, 0, 0] = True

    allowed = causal_ok.unsqueeze(0) & (unrestricted | same_group | bos_col)

    mask = torch.zeros(batch, seq_len, seq_len, device=device)
    mask[~allowed] = float('-inf')

    return mask.unsqueeze(1).expand(-1, nhead, -1, -1).reshape(batch * nhead, seq_len, seq_len)
