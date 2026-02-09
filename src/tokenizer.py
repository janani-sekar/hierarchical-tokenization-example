"""Two-level tokenizer for arithmetic expressions.

Level 1: Character-level tokenization.
Level 2: Inserts <num> summary tokens before each digit group,
         enabling HTP-style between-layer rewiring in the transformer.
"""

# Special tokens
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
NUM_ID = 3  # hierarchical number summary token

# Character vocabulary
DIGIT_CHARS = '0123456789'
OP_CHARS = '+-*='

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


def build_flat_sequence(expr: str, answer: str) -> tuple[list[int], int]:
    """Build a flat character-level sequence for decoder-only next-token prediction.

    Returns:
        tokens: [BOS, ...expr_chars, =, ...answer_chars, EOS]
        prefix_len: number of tokens before the first answer digit
                    (used to identify where generation starts at inference)
    """
    expr_ids = encode(expr)
    eq_id = CHAR_TO_ID['=']
    ans_ids = encode(answer)

    prefix = [BOS_ID] + expr_ids + [eq_id]
    tokens = prefix + ans_ids + [EOS_ID]
    return tokens, len(prefix)


def build_hierarchical_sequence(
    expr: str, answer: str,
) -> tuple[list[int], int, list[tuple[int, int]]]:
    """Build a hierarchical sequence with <num> tokens before each digit group.

    For "25+13", produces: [BOS, <num>, 2, 5, +, <num>, 1, 3, =, ...]
    The <num> tokens serve as "summary slots" that receive the hidden state
    of the last digit in their group via between-layer rewiring.

    Returns:
        tokens: full sequence including <num> tokens
        prefix_len: number of tokens before the first answer digit
        rewire_pairs: list of (dst_pos, src_pos) where
            dst_pos = position of <num> token
            src_pos = position of last digit in that number group
    """
    expr_ids = encode(expr)
    eq_id = CHAR_TO_ID['=']
    ans_ids = encode(answer)

    prefix = [BOS_ID]
    rewire_pairs = []

    i = 0
    while i < len(expr_ids):
        if is_digit_token(expr_ids[i]):
            num_pos = len(prefix)
            prefix.append(NUM_ID)
            while i < len(expr_ids) and is_digit_token(expr_ids[i]):
                prefix.append(expr_ids[i])
                i += 1
            last_digit_pos = len(prefix) - 1
            rewire_pairs.append((num_pos, last_digit_pos))
        else:
            prefix.append(expr_ids[i])
            i += 1

    prefix.append(eq_id)
    tokens = prefix + ans_ids + [EOS_ID]
    return tokens, len(prefix), rewire_pairs
