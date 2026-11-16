from collections.abc import Callable, Iterator

from tree_forge import CanonicalPosition, Canonicalizer

from .game import (
	BOARD_COLUMNS,
	GameState,
	Move,
	Player,
)


class ConnectFourCanonicalizer(Canonicalizer[GameState, Move]):
	"""Canonicalize Connect Four positions using horizontal reflection."""

	def position_key(self, state: GameState) -> str:
		"""Return the complete position as a string."""
		return self._board_to_string(
			state.board,
			state.current_player,
			state.move_count,
		)

	def position_hash(self, state: GameState) -> int:
		"""Return an integer encoding of the position."""
		return self._position_string_to_int(
			self.position_key(state)
		)

	def canonical_position_key(self, state: GameState) -> str:
		"""Return the lexicographically smaller symmetric position."""
		return min(
			key
			for key, _ in self.iter_symmetries(state)
		)

	def canonical_position_hash(self, state: GameState) -> int:
		"""Return an integer encoding of the canonical position."""
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
			for column in range(BOARD_COLUMNS):
				original = Move(column)

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
		for reflected in (False, True):

			def transform(
				move: Move,
				reflected: bool = reflected,
			) -> Move:
				if reflected:
					return Move(BOARD_COLUMNS - 1 - move.column)

				return move

			board = [
				[None for _ in range(BOARD_COLUMNS)]
				for _ in range(len(state.board))
			]

			for row in range(len(state.board)):
				for column in range(BOARD_COLUMNS):
					canonical_move = transform(Move(column))

					board[row][canonical_move.column] = (
						state.board[row][column]
					)

			yield (
				self._board_to_string(
					board,
					state.current_player,
					state.move_count,
				),
				transform,
			)

	@staticmethod
	def _board_to_string(
		board: list[list[Player | None]],
		current_player: Player,
		move_count: int,
	) -> str:
		"""Encode the current player, move count, and board."""
		symbols = {
			None: ".",
			Player.RED: "R",
			Player.YELLOW: "Y",
		}

		return (
			f"{current_player.value}:{move_count}:"
			+ "".join(
				symbols[cell]
				for row in board
				for cell in row
			)
		)

	@staticmethod
	def _position_string_to_int(position: str) -> int:
		"""Convert a position string into an injective integer encoding."""
		return int.from_bytes(
			position.encode("utf-8"),
			byteorder="big",
		)

