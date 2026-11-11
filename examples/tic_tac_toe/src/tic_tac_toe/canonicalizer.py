from collections.abc import Callable, Iterator

from tree_forge import CanonicalPosition, Canonicalizer

from .game import (
	BOARD_SIZE,
	GameState,
	Move,
	Player,
	TicTacToeGame,
)


class TicTacToeCanonicalizer(Canonicalizer[GameState, Move]):
	"""Canonicalize Tic-Tac-Toe positions using board symmetries."""

	def __init__(self, game: TicTacToeGame) -> None:
		self.game = game

	def position_key(self, state: GameState) -> str:
		"""Return the complete position as a string."""
		return self._board_to_string(
			state.board,
			state.current_player,
		)

	def position_hash(self, state: GameState) -> int:
		"""Return a collision-free integer encoding of the position."""
		return self._position_string_to_int(
			self.position_key(state)
		)

	def canonical_position_key(self, state: GameState) -> str:
		"""Return the lexicographically smallest symmetric position."""
		return min(
			key
			for key, _ in self.iter_symmetries(state)
		)

	def canonical_position_hash(self, state: GameState) -> int:
		"""Return a collision-free integer encoding of the canonical position."""
		return self._position_string_to_int(
			self.canonical_position_key(state)
		)

	def canonicalize(
		self,
		state: GameState,
	) -> CanonicalPosition[Move]:
		"""Return the canonical position and move transformations."""
		canonical_key, to_canonical = min(
			self.iter_symmetries(state),
			key=lambda candidate: candidate[0],
		)

		def from_canonical(move: Move) -> Move:
			for row in range(BOARD_SIZE):
				for column in range(BOARD_SIZE):
					original = Move(row, column)

					if to_canonical(original) == move:
						return original

			raise ValueError(f"Invalid canonical move: {move}")

		return CanonicalPosition(
			key=canonical_key,
			to_canonical_move=to_canonical,
			from_canonical_move=from_canonical,
		)

	def iter_symmetries(
		self,
		state: GameState,
	) -> Iterator[tuple[str, Callable[[Move], Move]]]:
		"""Yield symmetric position keys and their move transformations."""
		for rotation in range(4):
			for reflected in (False, True):

				def transform(
					move: Move,
					rotation: int = rotation,
					reflected: bool = reflected,
				) -> Move:
					row, column = move.row, move.column

					if reflected:
						column = BOARD_SIZE - 1 - column

					for _ in range(rotation):
						row, column = column, BOARD_SIZE - 1 - row

					return Move(row, column)

				board = [
					[None for _ in range(BOARD_SIZE)]
					for _ in range(BOARD_SIZE)
				]

				for row in range(BOARD_SIZE):
					for column in range(BOARD_SIZE):
						canonical_move = transform(
							Move(row, column)
						)

						board[canonical_move.row][canonical_move.column] = (
							state.board[row][column]
						)

				yield (
					self._board_to_string(
						board,
						state.current_player,
					),
					transform,
				)

	@staticmethod
	def _board_to_string(
		board: list[list[Player | None]],
		current_player: Player,
	) -> str:
		"""Encode the current player and board as a string."""
		symbols = {
			None: ".",
			Player.X: "X",
			Player.O: "O",
		}

		return current_player.name + "".join(
			symbols[cell]
			for row in board
			for cell in row
		)

	@staticmethod
	def _position_string_to_int(position: str) -> int:
		"""Convert a position string into a collision-free integer."""
		values = {
			".": 0,
			"X": 1,
			"O": 2,
		}

		result = 0

		for symbol in position:
			result = result * 3 + values[symbol]

		return result

