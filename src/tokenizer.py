"""Two-level tokenizer for arithmetic expressions.

Level 1: Character-level tokenization.
Level 2: Inserts <num> summary tokens before each digit group,
         enabling HTP-style between-layer rewiring in the transformer.

Supports both L2R (standard) and R2L (reversed digits) orderings.
"""

# Special tokens
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
NUM_ID = 3  # hierarchical number summary token

# Character vocabulary
DIGIT_CHARS = '0123456789'
OP_CHARS = '+-*/=.x^'

CHAR_TO_ID = {}
ID_TO_CHAR = {}
_next_id = 4  # after PAD, BOS, EOS, NUM

for _c in DIGIT_CHARS + OP_CHARS:
    CHAR_TO_ID[_c] = _next_id
    ID_TO_CHAR[_next_id] = _c
    _next_id += 1

VOCAB_SIZE = _next_id
DIGIT_IDS = frozenset(CHAR_TO_ID[c] for c in DIGIT_CHARS)


def encode(s: str) -> list[int]:
    """Encode a string into character-level token IDs."""
    return [CHAR_TO_ID[c] for c in s]


def decode(ids: list[int]) -> str:
    """Decode token IDs back to a string, skipping special tokens."""
    return ''.join(ID_TO_CHAR[i] for i in ids if i in ID_TO_CHAR)


def is_digit_token(tid: int) -> bool:
    return tid in DIGIT_IDS


def reverse_digits(s: str) -> str:
    """Reverse each contiguous run of digit characters in a string.

    Example: "25+13" -> "52+31", "38" -> "83", "1998" -> "8991"
    """
    result = []
    i = 0
    while i < len(s):
        if s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            result.append(s[i:j][::-1])
            i = j
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


# -- Flat sequences (no <num> tokens) ----------------------------------------

def build_flat_sequence(expr: str, answer: str) -> tuple[list[int], int]:
    """Build a flat character-level sequence for decoder-only next-token prediction.

    Returns:
        tokens: [BOS, ...expr_chars, =, ...answer_chars, EOS]
        prefix_len: number of tokens before the first answer digit
    """
    expr_ids = encode(expr)
    eq_id = CHAR_TO_ID['=']
    ans_ids = encode(answer)

    prefix = [BOS_ID] + expr_ids + [eq_id]
    tokens = prefix + ans_ids + [EOS_ID]
    return tokens, len(prefix)


def build_r2l_flat_sequence(expr: str, answer: str) -> tuple[list[int], int]:
    """Flat sequence with reversed digit ordering."""
    return build_flat_sequence(reverse_digits(expr), reverse_digits(answer))


# -- Hierarchical sequences (<num> tokens + group IDs) -----------------------

def build_hierarchical_sequence(
    expr: str, answer: str,
) -> tuple[list[int], int, list[tuple[int, int]], list[int]]:
    """Build a hierarchical sequence with <num> tokens before each digit group.

    For "25+13", produces: [BOS, <num>, 2, 5, +, <num>, 1, 3, =, 3, 8, EOS]

    Returns:
        tokens: full sequence including <num> tokens
        prefix_len: number of tokens before the first answer digit
        rewire_pairs: list of (dst_pos, src_pos) for HTP rewiring
        group_ids: per-token group assignment (-1 = unrestricted / answer tokens)
    """
    expr_ids = encode(expr)
    eq_id = CHAR_TO_ID['=']
    ans_ids = encode(answer)

    tokens = [BOS_ID]
    group_ids = [0]  # BOS gets its own group
    rewire_pairs = []
    next_group = 1

    i = 0
    while i < len(expr_ids):
        if is_digit_token(expr_ids[i]):
            # Start a number group: <num> token + all consecutive digits
            num_pos = len(tokens)
            tokens.append(NUM_ID)
            group_ids.append(next_group)
            while i < len(expr_ids) and is_digit_token(expr_ids[i]):
                tokens.append(expr_ids[i])
                group_ids.append(next_group)
                i += 1
            last_digit_pos = len(tokens) - 1
            rewire_pairs.append((num_pos, last_digit_pos))
            next_group += 1
        else:
            # Operator: gets its own group
            tokens.append(expr_ids[i])
            group_ids.append(next_group)
            next_group += 1
            i += 1

    # = and answer tokens are unrestricted (standard causal in all layers)
    tokens.append(eq_id)
    group_ids.append(-1)
    prefix_len = len(tokens)

    for aid in ans_ids:
        tokens.append(aid)
        group_ids.append(-1)

    tokens.append(EOS_ID)
    group_ids.append(-1)

    return tokens, prefix_len, rewire_pairs, group_ids


def build_r2l_hier_sequence(
    expr: str, answer: str,
) -> tuple[list[int], int, list[tuple[int, int]], list[int]]:
    """Hierarchical sequence with reversed digit ordering."""
    return build_hierarchical_sequence(reverse_digits(expr), reverse_digits(answer))
