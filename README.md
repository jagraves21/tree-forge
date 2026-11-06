# Tree Forge

A modular, general-purpose Python game search framework for deterministic, turn-based, two-player, zero-sum games.

## Goals

Tree Forge aims to provide a clear, extensible, and efficient foundation for game-search algorithms.

Planned capabilities include:

- Minimax and Negamax.
- Alpha-beta pruning.
- Iterative deepening.
- Beam search.
- Transposition tables.
- Position hashing.
- Optional position canonicalization.
- Configurable evaluation and move ordering.
- Search limits and cancellation.
- Search statistics.

## Project Structure

```text
tree-forge/
├── pyproject.toml
├── README.md
│
├── src/
│   └── tree_forge/
│       ├── __init__.py
│       ├── interfaces.py
│       ├── models.py
│       ├── search/
│       └── core/
│
├── tests/
│
└── examples/
    └── tic_tac_toe/
        ├── pyproject.toml
        └── README.md
```

## Requirements

- Python 3.11+

## Installation

From the project root:

```bash
pip install -e .
```

## Examples

Examples are maintained independently from the core package.

Each example has its own directory, dependencies, and installation instructions.

Available examples:

- [Tic-Tac-Toe](examples/tic_tac_toe/README.md)

## Testing

Run the test suite with:

```bash
pytest
```

## Development Philosophy

Development priorities:

1. Correctness.
2. Clear interfaces and modularity.
3. Testability.
4. Extensibility.
5. Measurable performance.
6. Optimization based on benchmarks.

The framework will prioritize reliable reference implementations before introducing advanced search optimizations.

## License

Tree Forge is distributed under the [MIT License](LICENSE).

