# Connect Four

A simple Connect Four implementation demonstrating the TreeForge game interface.

## Requirements

- Python 3.11+
- TreeForge

## Installation

From the project root:

```bash
pip install -e .
pip install -e ./examples/connect_four
```

## Running

From the example directory:

```bash
python -m connect_four.main
```

## Features

- Standard 6-row by 7-column Connect Four board.
- Two-player gameplay with alternating turns.
- Column-based legal move generation.
- Gravity-based piece placement.
- Horizontal, vertical, and diagonal win detection.
- Draw detection when the board is full.
- Win, loss, and draw outcomes (valued by an `Evaluator`).
- Make/unmake move support.
- Position key generation for position identification.

## Purpose

This example serves as a reference implementation of the `Game` interface.

It is intended to validate the TreeForge framework independently of any particular search algorithm.

Future iterations may demonstrate Minimax, Negamax, alpha-beta pruning, transposition tables, and symmetry-based canonicalization.

