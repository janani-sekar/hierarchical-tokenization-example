"""Generate arithmetic expression datasets for training."""

import random
from typing import List, Tuple


def generate_addition_dataset(
    n_samples: int,
    min_val: int = 1,
    max_val: int = 99,
    seed: int = 0,
) -> List[Tuple[str, str]]:
    """Generate addition problems as (expression, answer) string pairs.

    Example: ("25+13", "38")
    """
    rng = random.Random(seed)
    data = []
    for _ in range(n_samples):
        a = rng.randint(min_val, max_val)
        b = rng.randint(min_val, max_val)
        c = a + b
        data.append((f"{a}+{b}", str(c)))
    return data


def generate_mixed_dataset(
    n_samples: int,
    min_val: int = 1,
    max_val: int = 99,
    ops: str = "+-",
    seed: int = 0,
) -> List[Tuple[str, str]]:
    """Generate arithmetic problems with mixed operations.

    For subtraction, ensures the result is non-negative.
    Example: ("42-17", "25") or ("25+13", "38")
    """
    rng = random.Random(seed)
    data = []
    for _ in range(n_samples):
        op = rng.choice(ops)
        a = rng.randint(min_val, max_val)
        b = rng.randint(min_val, max_val)

        if op == '+':
            c = a + b
        elif op == '-':
            if a < b:
                a, b = b, a
            c = a - b
        elif op == '*':
            c = a * b
        else:
            raise ValueError(f"Unsupported operation: {op}")

        data.append((f"{a}{op}{b}", str(c)))
    return data
