# Hierarchical Tokenization for Arithmetic

Does giving a transformer structural hints about multi-digit numbers help it learn arithmetic? Does reversing digit order to match the addition algorithm help?

We compare six variants crossing two axes: digit ordering (L2R vs R2L) and attention structure (flat, HTP, HTP + hierarchical attention). Inspired by [Ding et al., 2025](https://arxiv.org/pdf/2511.14868).

## The grid

|                    | Flat causal       | HTP (rewiring)      | HTP + H-Attention       |
|--------------------|-------------------|---------------------|-------------------------|
| **L2R** (standard) | 1. L2R            | 2. L2R + HTP        | 3. L2R + HTP + H-Attn  |
| **R2L** (reversed) | 4. R2L            | 5. R2L + HTP        | 6. R2L + HTP + H-Attn  |

## Digit ordering

**L2R (left-to-right)**: standard digit order. `25 + 13 = 38`.

**R2L (right-to-left)**: reverse each number's digits. `52 + 31 = 83`. This aligns the generation order with how addition is actually computed: ones place first, then tens (with carries), then hundreds. The model can predict the ones digit immediately without needing to anticipate carries from ungenerated digits.

## Attention structure

### Flat causal (variants 1, 4)

Standard decoder-only transformer with causal attention. Every token attends to all previous tokens.

```
[BOS, 2, 5, +, 1, 3, =, 3, 8, EOS]
```

### HTP -- Hierarchical Token Prepending (variants 2, 5)

Inserts `<num>` summary tokens before each digit group. Between transformer layers, a rewiring step copies the hidden state of the last digit in each number to its `<num>` position. Later layers attend to `<num>` to get a number-level summary.

```
[BOS, <num>, 2, 5, +, <num>, 1, 3, =, 3, 8, EOS]
```

After layer 0, the hidden state at position "5" (which has attended to `<num>`, `2`, `5`) encodes the full number "25". The rewiring copies that state to `<num>`. Layer 1 and beyond see a composed number representation at `<num>`.

### HTP + Hierarchical Attention (variants 3, 6)

Same tokens and rewiring as HTP, but the first half of layers (0-1) use group-restricted attention: digits can only attend within their own number group. This forces each number to compose in isolation before interacting with other numbers. The second half (layers 2-3) use standard causal attention for cross-number reasoning.

Answer tokens (after `=`) use standard causal attention in all layers so they can see the composed representations.

## Architecture

All six models use the same parameter count (~202K). `ArithmeticTransformer` (variants 1, 2, 4, 5) and `HierAttnTransformer` (variants 3, 6) share identical layer structures -- the only difference is the attention masking strategy.

**Model details**: 4 transformer layers, d_model=64, 4 attention heads, FFN dim=256.

**Training**: Next-token prediction with cross-entropy loss on answer tokens only. Adam optimizer, lr=3e-4. Early stopping with patience=10 on validation accuracy, max 50 epochs.

## Datasets

Three data generating processes, selected via `--dgp`:

### Addition (`--dgp addition`)

Simple integer addition with operands 1-999.

`418+273=691`

### Mixed arithmetic (`--dgp mixed`)

All four operations (+, -, *, /) with operands 1-99. Subtraction ensures non-negative results. Division rounds to 2 decimal places.

`42-17=25`, `7/3=2.33`, `12*8=96`

### Algebraic (`--dgp algebraic`)

Single-variable expressions of the form `ax OP bx` with coefficients 1-99. Multiplication produces `x^2` terms; division cancels `x`.

`2x+3x=5x`, `2x*4x=8x^2`, `5x-3x=2x`, `2x/4x=0.5`

All datasets: 50K train / 5K val / 5K test.

## Results

### Addition

Test set exact-match accuracy (autoregressive decoding, 5K held-out problems):

| Variant            | Test Accuracy |
|--------------------|---------------|
| L2R                | 98.68%        |
| L2R + HTP          | 99.62%        |
| L2R + HTP + HAttn  | 99.68%        |
| R2L                | 99.94%        |
| R2L + HTP          | 100.00%       |
| R2L + HTP + HAttn  | 100.00%       |

R2L digit ordering is the single biggest win -- it aligns generation order with the addition algorithm (ones first, then carries propagate left). HTP rewiring gives a consistent boost on top of both orderings. Hierarchical attention adds a small further improvement for L2R but all R2L variants are near-perfect.

### Mixed arithmetic

Results pending.

### Algebraic

Results pending.

## Setup

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Then run scripts from the repo root, e.g. `uv run python src/train.py --dgp addition`, or activate `.venv` and run `python src/train.py` from `src/`.

## AWS (training / eval)

Training and eval run on a persistent AWS g4dn.xlarge GPU instance. Access via AWS SSO.

**Connect:**

1. Log in: `aws sso login --profile AWSPowerUserAccess-939693385233`
2. SSH: `ssh -i ~/.ssh/jsekar-htp-train.pem ubuntu@3.235.236.135` (instance `i-058718ba50fff1d38`)
3. On the instance the project lives in `~/htp/`. Activate env and run from `htp/src`: `source venv/bin/activate && cd src && python train.py --dgp addition`

**Sync plots back:** `rsync -avz -e "ssh -i ~/.ssh/jsekar-htp-train.pem" ubuntu@3.235.236.135:htp/plots/ plots/`

## Training

```bash
cd src
python train.py --dgp addition    # default
python train.py --dgp mixed
python train.py --dgp algebraic
```

## Project structure

```
src/
  data.py        # Dataset generation (addition, mixed, algebraic)
  tokenizer.py   # Character-level vocab, L2R/R2L, flat/hierarchical sequences
  model.py       # ArithmeticTransformer + HierAttnTransformer
  train.py       # Training loop comparing all 6 variants
  eval.py        # Evaluate saved checkpoints and plot results
```

## Reference

[Hierarchical Token Prepending: Enhancing Information Flow in Decoder-based LLM Embeddings](https://arxiv.org/pdf/2511.14868) (Ding et al., 2025)
