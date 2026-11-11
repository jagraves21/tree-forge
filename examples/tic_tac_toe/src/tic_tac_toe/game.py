from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum

from tree_forge import Game


BOARD_SIZE = 3


class Player(Enum):
	"""Players in a Tic-Tac-Toe game."""

	X = 1
	O = 2

	@property
	def opponent(self) -> Player:
		"""Return the opposing player."""
		return Player.O if self is Player.X else Player.X


@dataclass(frozen=True, slots=True)
class Move:
	"""A move represented by a board coordinate."""

	row: int
	column: int


@dataclass(slots=True)
class GameState:
	"""Mutable Tic-Tac-Toe game state."""

	board: list[list[Player | None]] = field(
		default_factory=lambda: [
			[None for _ in range(BOARD_SIZE)]
			for _ in range(BOARD_SIZE)
		]
	)
	current_player: Player = Player.X

	def __str__(self) -> str:
		"""Return a human-readable representation of the board."""
		symbols = {
			"X": "X",
			"O": "O",
		}

		lines = [
			"    0   1   2",
			"  ┌───┬───┬───┐",
		]

		for row_index, row in enumerate(self.board):
			cells = [
				symbols.get(cell.name, cell.name) if cell is not None else " "
				for cell in row
			]

			lines.append(
				f"{row_index} │ {' │ '.join(cells)} │"
			)

			if row_index < len(self.board) - 1:
				lines.append("  ├───┼───┼───┤")

		lines.append("  └───┴───┴───┘")

		current_player = symbols.get(
			self.current_player.name,
			self.current_player.name,
		)
		lines.append(f"Current player: {current_player}")

		return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class UndoData:
	"""Information required to reverse a move."""

	row: int
	column: int
	player: Player
	previous_player: Player


class TicTacToeGame(
	Game[GameState, Move, Player, UndoData]
):
	"""Implementation of deterministic Tic-Tac-Toe."""

	def initial_state(self) -> GameState:
		"""Create a new game with an empty board."""
		return GameState()

	def current_player(self, state: GameState) -> Player:
		"""Return the player whose turn it is."""
		return state.current_player

	def legal_moves(self, state: GameState) -> tuple[Move, ...]:
		"""Return all available board positions."""
		if self.is_terminal(state):
			return ()

		return tuple(
			Move(row, column)
			for row in range(BOARD_SIZE)
			for column in range(BOARD_SIZE)
			if state.board[row][column] is None
		)

	def make_move(
		self,
		state: GameState,
		move: Move,
	) -> UndoData:
		"""Apply a legal move and return its undo information."""
		if move not in self.legal_moves(state):
			raise ValueError(f"Illegal move: {move}")

		player = state.current_player

		undo_data = UndoData(
			row=move.row,
			column=move.column,
			player=player,
			previous_player=player,
		)

		state.board[move.row][move.column] = player
		state.current_player = player.opponent

		return undo_data

	def unmake_move(
		self,
		state: GameState,
		move: Move,
		undo_data: UndoData,
	) -> None:
		"""Restore the state to its previous position."""
		if (
			move.row != undo_data.row
			or move.column != undo_data.column
		):
			raise ValueError("Move does not match undo information")

		if state.board[move.row][move.column] is not undo_data.player:
			raise ValueError("Cannot undo move: board state does not match")

		if state.current_player is not undo_data.player.opponent:
			raise ValueError("Cannot undo move: unexpected current player")

		state.board[move.row][move.column] = None
		state.current_player = undo_data.previous_player

	def is_terminal(self, state: GameState) -> bool:
		"""Return whether the game has ended."""
		if self.winner(state) is not None:
			return True

		return all(
			cell is not None
			for row in state.board
			for cell in row
		)

	def terminal_score(
		self,
		state: GameState,
		player: Player,
	) -> float:
		"""Return the terminal score from a player's perspective."""
		if not self.is_terminal(state):
			raise ValueError("Cannot score a nonterminal position")

		winner = self.winner(state)

		if winner is None:
			return 0.0

		return 1.0 if winner is player else -1.0

	def winner(self, state: GameState) -> Player | None:
		"""Return the winning player, or None if there is no winner."""
		board = state.board

		lines = (
			*(board[row] for row in range(BOARD_SIZE)),
			*(
				[board[row][column] for row in range(BOARD_SIZE)]
				for column in range(BOARD_SIZE)
			),
			[board[i][i] for i in range(BOARD_SIZE)],
			[
				board[i][BOARD_SIZE - 1 - i]
				for i in range(BOARD_SIZE)
			],
		)

		for line in lines:
			first = line[0]

			if first is not None and all(cell is first for cell in line):
				return first

		return None

	def display(self, state: GameState) -> None:
		"""Print a human-readable representation of the board."""
		print(state)

