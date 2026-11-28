# TreeForge

A modular, general-purpose Python game search framework for deterministic, turn-based, two-player, zero-sum games.

## Goals

TreeForge aims to provide a clear, extensible, and efficient foundation for game-search algorithms.

Planned capabilities include:

- Minimax and Negamax.
- Alpha-beta pruning.
- Iterative deepening.
- Beam search.
- Transposition tables.
- Position hashing.
- Optional position canonicalization.
- Configurable evaluation and move ordering.
- Evaluators that value every position, terminal ones included.
- Search limits and cancellation.
- Search statistics.

## Outcomes and Evaluation

A `Game` reports whether a position is terminal and, if so, its `Outcome`
(`WIN`, `LOSS` or `DRAW`) relative to a given player. It never assigns
numbers. An `Evaluator` turns every position into a score, and searches
only compare those scores.

The default `Evaluator` values wins at positive infinity, losses at negative
infinity, and draws and nonterminal positions at 0.0. Subclass it and
override `evaluate_win`, `evaluate_loss`, `evaluate_draw` or
`evaluate_non_terminal` to prefer faster wins, value draws differently, or
add heuristics. Each receives the state, player, `ply` and remaining `depth`.

```python
class SoonerIsBetter(Evaluator):
	def evaluate_win(self, state, player, ply, depth):
		return 1000.0 - ply
```

Cached transposition-table scores are not adjusted for ply or depth, so
keeping ply or depth dependent values consistent is up to the evaluator.

## Project Structure

```text
tree-forge/
├── pyproject.toml
├── README.md
├── LICENSE
│
├── src/
│   └── tree_forge/
│       ├── __init__.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── interfaces.py
│       │   └── models.py
│       │
│       ├── algorithms/
│       │   ├── __init__.py
│       │   ├── _base.py
│       │   ├── minimax.py
│       │   └── random_search.py
│       │
│       └── caching/
│           ├── __init__.py
│           ├── search_dependency_tracker.py
│           └── transposition_table.py
│
├── tests/
│
└── examples/
    ├── tic_tac_toe/
    │   ├── pyproject.toml
    │   ├── README.md
    │   └── src/
    │       └── tic_tac_toe/
    │
    └── connect_four/
        ├── pyproject.toml
        ├── README.md
        └── src/
            └── connect_four/
```

## Requirements

- Python 3.11+

## Installation

From the project root:

```bash
pip install -e ".[test]"
pip install -e ./examples/tic_tac_toe
pip install -e ./examples/connect_four
```

## Examples

Examples are maintained independently from the core package.

Each example has its own directory, dependencies, and installation instructions.

Available examples:

- [Tic-Tac-Toe](examples/tic_tac_toe/)
- [Connect Four](examples/connect_four/)

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

TreeForge is distributed under the [MIT License](LICENSE).

