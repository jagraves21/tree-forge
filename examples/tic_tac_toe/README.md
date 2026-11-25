# Tic-Tac-Toe

A Tic-Tac-Toe implementation demonstrating the TreeForge game interface and search capabilities.

The example provides a playable 3x3 game with human and computer-controlled players. Computer players use TreeForge's Minimax search, with optional search enhancements that can be enabled globally or configured independently for each player.

## Requirements

- Python 3.11+
- TreeForge

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

By default, the game starts with two human players.

### Choosing Players

Each player can independently be configured as a human, Minimax AI, or random player.

```bash
python -m tic_tac_toe.main --x-player minimax --o-player human
```

Run two computer players against each other:

```bash
python -m tic_tac_toe.main --x-player minimax --o-player minimax
```

### Search Configuration

The Minimax search depth can be configured globally or independently for each player.

```bash
python -m tic_tac_toe.main --x-player minimax --o-player minimax --depth 9
```

Player-specific settings override the corresponding global settings.

| Option | Description | Default |
|---|---|---|
| `--depth` | Maximum search depth. | `9` |
| `--x-depth` | X's maximum search depth. | Global depth |
| `--o-depth` | O's maximum search depth. | Global depth |
| `--alpha-beta` | Enable alpha-beta pruning. | Disabled |
| `--move-ordering` | Select a move-ordering strategy. | `none` |
| `--evaluator` | Enable the Tic-Tac-Toe heuristic evaluator. | Disabled |
| `--canonicalization` | Enable board-symmetry canonicalization. | Disabled |
| `--transposition-table` | Enable transposition-table caching. | Disabled |
| `--dependency-tracking` | Enable transposition-table dependency tracking. | Disabled |

Each search enhancement also supports player-specific options using the `--x-` and `--o-` prefixes.

For example:

```bash
python -m tic_tac_toe.main \
	--x-player minimax \
	--o-player minimax \
	--x-depth 9 \
	--o-depth 5 \
	--x-alpha-beta \
	--o-move-ordering positional
```

### Move Ordering

Three move-ordering strategies are provided:

| Strategy | Description |
|---|---|
| `none` | Use the game's original legal-move order. |
| `positional` | Prefer the center, then corners, then edges. |
| `randomized` | Preserve positional priorities while shuffling moves within each priority group. |
| `random` | Randomly shuffle all legal moves. |

Example:

```bash
python -m tic_tac_toe.main \
	--x-player minimax \
	--o-player minimax \
	--move-ordering positional
```

### Transposition Tables and Canonicalization

Transposition tables can reuse previously evaluated positions. Tic-Tac-Toe's canonicalizer recognizes equivalent board positions created by rotations and reflections.

Enable both together:

```bash
python -m tic_tac_toe.main \
	--x-player minimax \
	--o-player minimax \
	--canonicalization \
	--transposition-table
```

Dependency tracking can also be enabled when using a transposition table:

```bash
python -m tic_tac_toe.main \
	--x-player minimax \
	--o-player minimax \
	--canonicalization \
	--transposition-table \
	--dependency-tracking
```

Dependency tracking requires the transposition table, and the transposition table requires canonicalization in this example.

### Playing Multiple Games

Run a series of games between computer-controlled players:

```bash
python -m tic_tac_toe.main \
	--x-player minimax \
	--o-player minimax \
	--games 100
```

Minimax instances and their associated transposition tables are reused across games. Each game starts with a fresh board, while search-related cached information can persist.

At the end of a multi-game run, the program reports match results, aggregate search statistics, and final transposition-table sizes.

### Quiet Mode

Use `--quiet` to suppress boards, individual search results, and aggregate statistics. Only the winner of each game is printed.

```bash
python -m tic_tac_toe.main \
	--x-player minimax \
	--o-player minimax \
	--games 100 \
	--quiet
```

### Move Delay

The delay between moves can be configured with `--delay`. The default is 0.5 seconds.

```bash
python -m tic_tac_toe.main \
	--x-player minimax \
	--o-player minimax \
	--delay 0
```

### Watching the AIs Learn

One particularly interesting way to use this example is to run two Minimax players against each other with persistent transposition tables and dependency tracking:

```bash
python -m tic_tac_toe.main \
	--x-player minimax \
	--o-player minimax \
	--alpha-beta \
	--canonicalization \
	--transposition-table \
	--dependency-tracking \
	--move-ordering random \
	--depth 2 \
	--games 1000 \
	--delay 0 \
	--quiet
```

This configuration creates an interesting experiment:

- **Persistent transposition tables** allow each player to reuse search information from previous games.
- **Dependency tracking** helps maintain the validity of cached search information as positions are revisited.
- **Random move ordering** introduces variation in exploration, allowing the players to encounter different search paths.
- **Depth 2** keeps each individual search relatively shallow, making the accumulated knowledge from previous games especially interesting to observe.
- **Zero delay** allows games to run quickly.
- **Quiet mode** prints only the winner of each game, making the evolving match results easy to follow.

Initially, X and O may win games as they encounter positions they have not explored before. As their transposition tables accumulate useful information, more games may end in draws.

This is especially interesting in Tic-Tac-Toe because either player can force a draw with perfect play. The increasing frequency of draws provides a visible indication that the players are benefiting from information accumulated across games.

It is a simple way to experiment with persistent search knowledge and observe how caching, dependency tracking, and move ordering interact over repeated games.

## Features

### Game Implementation

- Standard 3x3 Tic-Tac-Toe rules.
- Human, random, and Minimax players.
- Legal move generation.
- Terminal-state detection.
- Win, loss, and draw outcomes.
- Mutable game-state representation.
- Make/unmake move support with undo data.

### Search Capabilities

- TreeForge Minimax integration.
- Configurable search depth.
- Optional alpha-beta pruning.
- Configurable move ordering.
- Heuristic position evaluation.
- Board-symmetry canonicalization.
- Transposition-table caching.
- Optional transposition-table dependency tracking.
- Persistent search instances across multiple games.

### Search Statistics

Search results and aggregate match statistics include:

- Nodes visited.
- Leaf nodes.
- Terminal nodes.
- Alpha-beta cutoffs.
- Transposition-table hits.
- Transposition-table stores.
- Transposition-table invalidations and refreshes, when exposed by the search result.
- Search duration.
- Transposition-table sizes.

## Project Structure

```text
tic_tac_toe/
├── src/
│   └── tic_tac_toe/
│       ├── __init__.py
│       ├── main.py
│       ├── game.py
│       ├── evaluator.py
│       ├── canonicalizer.py
│       └── move_ordering.py
├── pyproject.toml
└── README.md
```

## Purpose

This example serves as both a reference implementation of TreeForge's `Game` interface and a demonstration of its search functionality.

It illustrates how a game can integrate with TreeForge's Minimax implementation and optionally extend search behavior through evaluators, canonicalizers, move ordering, alpha-beta pruning, and transposition tables.

It also provides a practical way to compare search configurations across multiple games and observe their effects through search statistics.

