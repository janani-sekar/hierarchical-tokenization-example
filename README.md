# Hierarchical Tokenization for Mathematical Expressions

This repository implements hierarchical tokenization techniques for processing mathematical equations, inspired by recent research in structured tokenization approaches (based on https://arxiv.org/pdf/2511.14868).

## Overview

Traditional tokenization methods treat mathematical expressions as flat sequences of characters or symbols, losing important structural and hierarchical information. This project explores **hierarchical tokenization** - a method that preserves the inherent tree-like structure of mathematical expressions during the tokenization process.

### What is Hierarchical Tokenization?

Hierarchical tokenization decomposes mathematical expressions into multi-level token representations that respect:
- **Operator precedence** (e.g., multiplication before addition)
- **Nested structures** (e.g., parentheses, fractions, exponents)
- **Semantic groupings** (e.g., function applications, subscripts)

For example, the expression `2 * (3 + 4)` would be tokenized hierarchically as:
```
[MULT]
  ├── [NUM: 2]
  └── [ADD]
      ├── [NUM: 3]
      └── [NUM: 4]
```

## Motivation

Mathematical expressions have inherent hierarchical structure that is critical for:
- **Understanding mathematical relationships**: Preserving operator precedence and grouping
- **Symbolic computation**: Enabling symbolic manipulation and simplification
- **Machine learning on math**: Better representations for models processing mathematical content
- **Formula parsing**: Accurate interpretation of complex expressions

## Project Structure

```
hierarchical-tokenization-example/
├── src/                    # Source code
│   ├── tokenizer/         # Hierarchical tokenization implementation
│   ├── parser/            # Expression parsing utilities
│   ├── models/            # Data structures for hierarchical tokens
│   └── utils/             # Helper functions
├── data/                   # Example datasets and test cases
│   ├── raw/               # Raw mathematical expressions
│   ├── processed/         # Tokenized outputs
│   └── examples/          # Example use cases
├── tests/                  # Unit tests (to be added)
├── notebooks/              # Jupyter notebooks for demos (to be added)
├── .gitignore             # Git ignore patterns
└── README.md              # This file
```

## Features

- **Multi-level tokenization**: Breaks down expressions into hierarchical token trees
- **Operator precedence handling**: Respects mathematical operator precedence rules
- **Nested structure support**: Handles parentheses, brackets, and other grouping symbols
- **Extensible design**: Easy to add support for new mathematical operators and functions
- **Math-specific tokenization**: Designed specifically for mathematical notation

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management

### Installation

```bash
# Clone the repository
git clone https://github.com/janani-sekar/hierarchical-tokenization-example.git
cd hierarchical-tokenization-example

# Install dependencies (to be specified)
pip install -r requirements.txt
```

### Basic Usage

```python
# Example usage (implementation to be added)
from src.tokenizer import HierarchicalTokenizer

# Initialize tokenizer
tokenizer = HierarchicalTokenizer()

# Tokenize a mathematical expression
expression = "2 * (3 + 4) / 5"
tokens = tokenizer.tokenize(expression)

# Access hierarchical structure
print(tokens.tree_representation())
```

## Implementation Roadmap

- [ ] Core tokenizer implementation
- [ ] Expression parser
- [ ] Hierarchical token data structures
- [ ] Support for basic arithmetic operators (+, -, *, /)
- [ ] Support for parentheses and grouping
- [ ] Support for advanced operators (exponents, roots, etc.)
- [ ] Support for mathematical functions (sin, cos, log, etc.)
- [ ] Visualization tools for token hierarchies
- [ ] Example datasets
- [ ] Unit tests
- [ ] Documentation and tutorials

## Use Cases

1. **Mathematical Expression Parsing**: Convert LaTeX or plain text math into structured representations
2. **Symbolic Computation**: Enable computer algebra systems to manipulate expressions
3. **Math Education Tools**: Help students understand expression structure and evaluation order
4. **Machine Learning**: Create better representations for models that process mathematical content
5. **Formula Search**: Enable semantic search over mathematical expressions

## Research Background

This implementation is based on hierarchical tokenization approaches from recent research, particularly focusing on applications to mathematical expressions. The key insight is that preserving hierarchical structure during tokenization leads to better downstream performance in tasks involving mathematical reasoning and computation.

**Reference**: https://arxiv.org/pdf/2511.14868

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas where contributions would be particularly valuable:

- Additional operator support
- Performance optimizations
- More comprehensive test cases
- Documentation improvements
- Example notebooks

## License

(To be specified)

## Acknowledgments

- Based on hierarchical tokenization research
- Inspired by advances in structured representation learning
