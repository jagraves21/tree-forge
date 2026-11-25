
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from tree_forge import Game, Outcome


BOARD_ROWS = 6
BOARD_COLUMNS = 7
CONNECT_LENGTH = 4


class Player(Enum):
	"""Players in a Connect Four game."""

	RED = 1
	YELLOW = 2

	@property
	def opponent(self) -> Player:
		"""Return the opposing player."""
		return (
			Player.YELLOW
			if self is Player.RED
			else Player.RED
		)


@dataclass(frozen=True, slots=True)
class Move:
	"""A move represented by a column."""

	column: int


@dataclass(slots=True)
class GameState:
	"""Mutable Connect Four game state."""

	board: list[list[Player | None]] = field(
		default_factory=lambda: [
			[None for _ in range(BOARD_COLUMNS)]
			for _ in range(BOARD_ROWS)
		]
	)
	current_player: Player = Player.RED
	move_count: int = 0

	def __str__(self) -> str:
		"""Return a human-readable board representation."""
		symbols = {
			Player.RED: "R",
			Player.YELLOW: "Y",
		}

		lines = [
			"  " + "   ".join(
				str(column)
				for column in range(BOARD_COLUMNS)
			),
			"┌" + "───┬" * (BOARD_COLUMNS - 1) + "───┐",
		]

		for row_index, row in enumerate(self.board):
			cells = [
				symbols.get(cell, " ")
				for cell in row
			]

			lines.append(
				"│ " + " │ ".join(cells) + " │"
			)

			if row_index < BOARD_ROWS - 1:
				lines.append(
					"├" + "───┼" * (BOARD_COLUMNS - 1) + "───┤"
				)

		lines.append(
			"└" + "───┴" * (BOARD_COLUMNS - 1) + "───┘"
		)

		lines.append(
			f"Current player: {symbols[self.current_player]}"
		)
		lines.append(f"Moves played: {self.move_count}")

		return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class UndoData:
	"""Information required to reverse a move."""

	row: int
	column: int
	player: Player
	previous_player: Player
	previous_move_count: int


class ConnectFourGame(
	Game[GameState, Move, Player, UndoData]
):
	"""Implementation of deterministic Connect Four."""

	def initial_state(self) -> GameState:
		return GameState()

	def current_player(self, state: GameState) -> Player:
		return state.current_player

	def legal_moves(self, state: GameState) -> tuple[Move, ...]:
		if self.is_terminal(state):
			return ()

		return tuple(
			Move(column)
			for column in range(BOARD_COLUMNS)
			if state.board[0][column] is None
		)

	def make_move(
		self,
		state: GameState,
		move: Move,
	) -> UndoData:
		if move not in self.legal_moves(state):
			raise ValueError(f"Illegal move: {move}")

		column = move.column

		row = next(
			row_index
			for row_index in range(BOARD_ROWS - 1, -1, -1)
			if state.board[row_index][column] is None
		)

		player = state.current_player

		undo_data = UndoData(
			row=row,
			column=column,
			player=player,
			previous_player=player,
			previous_move_count=state.move_count,
		)

		state.board[row][column] = player
		state.current_player = player.opponent
		state.move_count += 1

		return undo_data

	def unmake_move(
		self,
		state: GameState,
		move: Move,
		undo_data: UndoData,
	) -> None:
		if move.column != undo_data.column:
			raise ValueError("Move does not match undo information")

		if state.board[undo_data.row][undo_data.column] is not undo_data.player:
			raise ValueError("Cannot undo move: board state does not match")

		if state.current_player is not undo_data.player.opponent:
			raise ValueError("Cannot undo move: unexpected current player")

		if any(
			state.board[row][move.column] is not None
			for row in range(undo_data.row)
		):
			raise ValueError("Cannot undo move: piece is not on top")

		state.board[undo_data.row][undo_data.column] = None
		state.current_player = undo_data.previous_player
		state.move_count = undo_data.previous_move_count

	def is_terminal(self, state: GameState) -> bool:
		if self.winner(state) is not None:
			return True

		return all(
			state.board[0][column] is not None
			for column in range(BOARD_COLUMNS)
		)

	def outcome(
		self,
		state: GameState,
		player: Player,
	) -> Outcome:
		if not self.is_terminal(state):
			raise ValueError(
				"Cannot determine the outcome of a nonterminal position"
			)

		winner = self.winner(state)

		if winner is None:
			return Outcome.DRAW

		return Outcome.WIN if winner is player else Outcome.LOSS

	def position_key(self, state: GameState) -> tuple:
		"""Return a collision-free representation of the position."""
		return (
			tuple(
				tuple(
					0 if cell is None else cell.value
					for cell in row
				)
				for row in state.board
			),
			state.current_player.value,
			state.move_count,
		)

	def position_hash(self, state: GameState) -> int:
		return hash(self.position_key(state))

	def winner(self, state: GameState) -> Player | None:
		"""Return the winning player, if one exists."""
		directions = (
			(0, 1),
			(1, 0),
			(1, 1),
			(1, -1),
		)

		for row in range(BOARD_ROWS):
			for column in range(BOARD_COLUMNS):
				player = state.board[row][column]

				if player is None:
					continue

				for row_delta, column_delta in directions:
					end_row = row + (CONNECT_LENGTH - 1) * row_delta
					end_column = column + (CONNECT_LENGTH - 1) * column_delta

					if not (
						0 <= end_row < BOARD_ROWS
						and 0 <= end_column < BOARD_COLUMNS
					):
						continue

					if all(
						state.board[
							row + offset * row_delta
						][
							column + offset * column_delta
						] is player
						for offset in range(1, CONNECT_LENGTH)
					):
						return player

		return None

	def display(self, state: GameState) -> None:
		print(state)

