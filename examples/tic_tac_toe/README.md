# Tic-Tac-Toe

A simple Tic-Tac-Toe implementation demonstrating the Tree Forge game interface.

## Requirements

- Python 3.11+
- Tree Forge

## Installation

From the project root:

```bash
pip install -e .
pip install -e ./examples/tic_tac_toe
```

## Running

From the example directory:

```bash
python -m tic_tac_toe.main
```

## Features

- Standard 3x3 Tic-Tac-Toe rules.
- Two-player turn-based gameplay.
- Legal move generation.
- Terminal-state detection.
- Win, loss, and draw scoring.
- Make/unmake move support.

## Purpose

This example serves as a reference implementation of the `Game` interface.

It is intended to validate the Tree Forge framework independently of any particular search algorithm.

Future iterations may demonstrate Minimax, Negamax, alpha-beta pruning, and other search capabilities.

