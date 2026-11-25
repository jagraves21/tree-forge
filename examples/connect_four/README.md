# Connect Four

A Connect Four implementation demonstrating the TreeForge game interface and search capabilities.

The example provides a playable 6-row by 7-column game with human and computer-controlled players. Computer players use TreeForge's Minimax search, with optional search enhancements that can be enabled globally or configured independently for each player.

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

By default, the game starts with two human players. Human players choose a column by entering its number, as shown above the board.

### Choosing Players

Each player can independently be configured as a human, Minimax AI, or random player. The players are Red, who moves first, and Yellow.

```bash
python -m connect_four.main --red-player minimax --yellow-player human
```

Run two computer players against each other:

```bash
python -m connect_four.main --red-player minimax --yellow-player minimax
```

### Search Configuration

The Minimax search depth can be configured globally or independently for each player.

```bash
python -m connect_four.main --red-player minimax --yellow-player minimax --depth 4
```

Player-specific settings override the corresponding global settings.

| Option | Description | Default |
|---|---|---|
| `--depth` | Maximum search depth. | `4` |
| `--red-depth` | Red's maximum search depth. | Global depth |
| `--yellow-depth` | Yellow's maximum search depth. | Global depth |
| `--alpha-beta` | Enable alpha-beta pruning. | Disabled |
| `--move-ordering` | Select a move-ordering strategy. | `none` |
| `--evaluator` | Enable the Connect Four heuristic evaluator. | Disabled |
| `--canonicalization` | Enable horizontal-reflection canonicalization. | Disabled |
| `--transposition-table` | Enable transposition-table caching. | Disabled |
| `--dependency-tracking` | Enable transposition-table dependency tracking. | Disabled |

Each search enhancement also supports player-specific options using the `--red-` and `--yellow-` prefixes.

For example:

```bash
python -m connect_four.main \
	--red-player minimax \
	--yellow-player minimax \
	--red-depth 6 \
	--yellow-depth 4 \
	--red-alpha-beta \
	--yellow-move-ordering tactical
```

A full Connect Four game is too deep to search to the end, so the search depth matters much more than in smaller games. Without alpha-beta pruning, depths above about 6 become slow.

### Evaluator

Without `--evaluator`, Minimax uses TreeForge's default evaluator, which only distinguishes wins, losses, and draws. A depth-limited search then sees every unfinished position as equal and can only react to forced results within its depth.

With `--evaluator`, positions are scored by `ConnectFourEvaluator`:

- Faster wins score higher and slower losses score higher.
- Unfinished positions are scored by counting open lines for each player, with a bonus for the center column.

```bash
python -m connect_four.main \
	--red-player minimax \
	--yellow-player minimax \
	--evaluator \
	--alpha-beta
```

### Move Ordering

Three move-ordering strategies are provided:

| Strategy | Description |
|---|---|
| `none` | Use the game's original legal-move order. |
| `tactical` | Try immediate wins first, then moves that leave the opponent the fewest immediate winning replies. Ties prefer lower columns for Red and higher columns for Yellow. |
| `randomized` | Preserve tactical priorities while shuffling moves within each priority group. |
| `random` | Randomly shuffle all legal moves. |

Example:

```bash
python -m connect_four.main \
	--red-player minimax \
	--yellow-player minimax \
	--alpha-beta \
	--move-ordering tactical
```

Move ordering never changes the score Minimax finds, but it affects which of several equally good moves is chosen. The tactical strategy is what makes a player block a threat when every move would lose at the same search depth.

### Transposition Tables and Canonicalization

Transposition tables can reuse previously evaluated positions. Connect Four's canonicalizer recognizes a position and its horizontal mirror image as equivalent.

Enable both together:

```bash
python -m connect_four.main \
	--red-player minimax \
	--yellow-player minimax \
	--canonicalization \
	--transposition-table
```

Dependency tracking can also be enabled when using a transposition table:

```bash
python -m connect_four.main \
	--red-player minimax \
	--yellow-player minimax \
	--canonicalization \
	--transposition-table \
	--dependency-tracking
```

Dependency tracking requires the transposition table, and the transposition table requires canonicalization in this example.

### Playing Multiple Games

Run a series of games between computer-controlled players:

```bash
python -m connect_four.main \
	--red-player minimax \
	--yellow-player minimax \
	--games 100
```

Minimax instances and their associated transposition tables are reused across games. Each game starts with a fresh board, while search-related cached information can persist.

At the end of a multi-game run, the program reports match results, aggregate search statistics, and final transposition-table sizes.

### Quiet Mode

Use `--quiet` to suppress boards and individual search results. Only the winner of each game is printed, followed by the aggregate statistics when more than one game is played.

```bash
python -m connect_four.main \
	--red-player minimax \
	--yellow-player minimax \
	--games 100 \
	--quiet
```

### Move Delay

The delay between moves can be configured with `--delay`. The default is 0.5 seconds.

```bash
python -m connect_four.main \
	--red-player minimax \
	--yellow-player minimax \
	--delay 0
```

### Comparing Search Configurations

One useful way to use this example is to give the two players different configurations and compare their results over many games. For example, a deeper search against a shallower one, both with the heuristic evaluator and a persistent transposition table:

```bash
python -m connect_four.main \
	--red-player minimax \
	--yellow-player minimax \
	--evaluator \
	--alpha-beta \
	--canonicalization \
	--transposition-table \
	--move-ordering randomized \
	--red-depth 5 \
	--yellow-depth 3 \
	--games 100 \
	--delay 0 \
	--quiet
```

This configuration creates an interesting experiment:

- **Different depths** show how much search depth matters, since the deeper player can see threats the shallower one cannot.
- **Randomized move ordering** keeps the tactical priorities but varies which equally ranked move is tried first, so the games are not all identical.
- **Persistent transposition tables** let each player reuse search information from previous games. The aggregate statistics show table hits and final table sizes.
- **Quiet mode and zero delay** keep the run fast and print only the winner of each game and the final summary.

Swapping the depths or move-ordering strategies between runs is a simple way to see how each setting affects the results and the search statistics.

## Features

### Game Implementation

- Standard 6-row by 7-column Connect Four board.
- Human, random, and Minimax players.
- Column-based legal move generation.
- Gravity-based piece placement.
- Horizontal, vertical, and diagonal win detection.
- Draw detection when the board is full.
- Win, loss, and draw outcomes (valued by an `Evaluator`).
- Mutable game-state representation.
- Make/unmake move support with undo data.
- Position key generation for position identification.

### Search Capabilities

- TreeForge Minimax integration.
- Configurable search depth.
- Optional alpha-beta pruning.
- Configurable move ordering: tactical, randomized, or random.
- Heuristic position evaluation.
- Horizontal-reflection canonicalization.
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
connect_four/
├── src/
│   └── connect_four/
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
