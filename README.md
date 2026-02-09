# Hierarchical Tokenization for Arithmetic

Does giving a transformer structural hints about multi-digit numbers help it learn arithmetic?

We compare two approaches: a standard character-level transformer versus one that uses hierarchical token prepending (HTP) to compose digit groups into number-level representations. Inspired by [Ding et al., 2025](https://arxiv.org/pdf/2511.14868).

## The idea

Consider the expression `25 + 13 = 38`.

**Flat tokenization** treats every character as its own token:

```
[BOS, 2, 5, +, 1, 3, =, 3, 8, EOS]
```

The transformer has to figure out from raw characters that `2, 5` together mean "twenty-five."

**Hierarchical tokenization** inserts a `<num>` summary token before each digit group:

```
[BOS, <num>, 2, 5, +, <num>, 1, 3, =, 3, 8, EOS]
```

Between transformer layers, a rewiring step copies the hidden state of the last digit in each number (which has attended to all preceding digits via causal attention) back into the `<num>` position. Later layers can then attend to `<num>` to get a number-level summary. The transformer's own attention does the composition -- no separate learned module is needed.

## How rewiring works

After layer 0 processes the sequence with causal attention, the hidden state at position `5` (the digit "5" in "25") has attended to `<num>`, `2`, and `5`. It now encodes information about the full number "25".

The rewiring step copies that hidden state to the `<num>` position. In layer 1, any later token (like `+` or `=`) that attends to position `<num>` now sees a number-level summary of "25", even though `<num>` comes *before* the digits in the sequence. This creates backward information flow within the causal attention framework.

```
Layer 0:  [BOS, <num>, 2, 5, +, <num>, 1, 3, =, ...]
                 |           |
                 |     (5 attends to <num>, 2, 5)
                 |           |
Rewire:          <--- copy ---
                 |
Layer 1:  [BOS, <num>=summary(25), 2, 5, +, ...]
                      ^
                      |
              (+ can now attend to number-level summary of 25)
```

This rewiring happens between every pair of consecutive layers (skipping before layer 0).

## Architecture

Both models use the **exact same** `ArithmeticTransformer` class -- a decoder-only transformer with causal attention. The only difference at runtime is whether the rewiring step is active:

- **Flat model**: `model(tokens)` -- standard causal transformer, no rewiring
- **Hierarchical model**: `model(tokens, rewire_src=..., rewire_dst=...)` -- same transformer, but with between-layer rewiring

Both models have identical parameter counts. The comparison isolates the effect of the structural `<num>` tokens and rewiring.

**Model details**: 4 transformer layers, d_model=64, 4 attention heads, FFN dim=256, ~202K parameters.

**Task**: Given an expression like `418+273=`, predict the answer `691` character by character.

**Training**: Next-token prediction with cross-entropy loss, computed only on the answer tokens (after `=`). Adam optimizer, learning rate 3e-4.

**Data**: 50K random addition problems with operands from 1 to 999. The space of possible problems (~998K) is much larger than the training set, so the model must generalize.

## Results

With 1-999 addition (50K train, 5K test, 50 epochs):

| Epoch | Flat acc | Hier acc |
|-------|----------|----------|
| 5     | 1.2%     | **4.3%** |
| 10    | 76.5%    | **83.6%** |
| 20    | 95.0%    | **97.2%** |
| 30    | 97.9%    | **98.7%** |
| 50    | 98.8%    | **99.6%** |

The hierarchical model learns slightly faster.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
cd src
python train.py
```

## Project structure

```
src/
  data.py        # Generate arithmetic expression datasets
  tokenizer.py   # Character-level vocab + hierarchical sequence construction
  model.py       # Decoder-only transformer with optional HTP rewiring
  train.py       # Training loop comparing flat vs hierarchical
```

## Reference

[Hierarchical Token Prepending: Enhancing Information Flow in Decoder-based LLM Embeddings](https://arxiv.org/pdf/2511.14868) (Ding et al., 2025)
