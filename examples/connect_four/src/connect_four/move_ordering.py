import random
from collections.abc import Sequence

from tree_forge import MoveOrdering

from .game import (
	BOARD_COLUMNS,
	BOARD_ROWS,
	CONNECT_LENGTH,
	GameState,
	Move,
	Player,
)


class ConnectFourMoveOrdering(
	MoveOrdering[GameState, Move, Player]
):
	"""Order Connect Four moves by tactical priority and player preference.

	Moves are ordered as follows:

	1. Moves that win immediately.
	2. Other moves, by how many immediate winning replies they leave
	   the opponent, fewest first.
	3. Ties by side preference: Red prefers lower columns, Yellow higher.

	Minimax keeps the first of several equally good moves, so this is also
	what makes it prefer to stop threats when every move loses.

	The ordering never makes Minimax choose a move with a worse score,
	so it does not interfere with preferring faster wins and slower losses.

	The state is not modified.
	"""

	_DIRECTIONS = (
		(0, 1),
		(1, 0),
		(1, 1),
		(1, -1),
	)

	_OFF_BOARD = object()

	@staticmethod
	def _playable_row(
		board: list[list[Player | None]],
		column: int,
	) -> int | None:
		"""Return the row a piece dropped in a column lands in, if any."""
		for row in range(BOARD_ROWS - 1, -1, -1):
			if board[row][column] is None:
				return row

		return None

	@classmethod
	def _completes_line(
		cls,
		board: list[list[Player | None]],
		row: int,
		column: int,
		player: Player,
		placed: tuple[int, int, Player] | None = None,
	) -> bool:
		"""Return whether a piece at a cell would win the game.

		placed is an extra piece, as (row, column, player), that is
		assumed to be on the board as well. The board is never modified.
		"""

		def owner(cell_row: int, cell_column: int):
			if not (
				0 <= cell_row < BOARD_ROWS
				and 0 <= cell_column < BOARD_COLUMNS
			):
				return cls._OFF_BOARD

			if placed is not None and (cell_row, cell_column) == placed[:2]:
				return placed[2]

			return board[cell_row][cell_column]

		for row_delta, column_delta in cls._DIRECTIONS:
			length = 1

			for sign in (1, -1):
				step_row = row + sign * row_delta
				step_column = column + sign * column_delta

				while owner(step_row, step_column) is player:
					length += 1
					step_row += sign * row_delta
					step_column += sign * column_delta

			if length >= CONNECT_LENGTH:
				return True

		return False

	@classmethod
	def _winning_columns(
		cls,
		board: list[list[Player | None]],
		player: Player,
	) -> set[int]:
		"""Return the columns in which the player could win right now."""
		columns = set()

		for column in range(BOARD_COLUMNS):
			row = cls._playable_row(board, column)

			if row is not None and cls._completes_line(
				board,
				row,
				column,
				player,
			):
				columns.add(column)

		return columns

	@classmethod
	def _immediate_threats_left(
		cls,
		board: list[list[Player | None]],
		column: int,
		player: Player,
		opponent_wins: set[int],
	) -> int:
		"""Count the opponent's winning replies after a move.

		Dropping a piece in an opponent winning column blocks that threat.
		Dropping a piece elsewhere can open a winning cell immediately
		above it, which also counts as an opponent threat.
		"""
		threats = len(opponent_wins - {column})
		row = cls._playable_row(board, column)

		if (
			row is not None
			and row > 0
			and cls._completes_line(
				board,
				row - 1,
				column,
				player.opponent,
				placed=(row, column, player),
			)
		):
			threats += 1

		return threats

	@classmethod
	def _tactical_key(
		cls,
		board: list[list[Player | None]],
		column: int,
		player: Player,
		opponent_wins: set[int],
	) -> tuple[int, int]:
		"""Return the tactical group of a move, lower sorting first.

		(0, 0) is a winning move. Other playable moves are (1, n),
		where n is the number of immediate winning replies left for
		the opponent. Unplayable moves come last.
		"""
		row = cls._playable_row(board, column)

		if row is None:
			return (2, 0)

		if cls._completes_line(board, row, column, player):
			return (0, 0)

		return (
			1,
			cls._immediate_threats_left(
				board,
				column,
				player,
				opponent_wins,
			),
		)

	def order_moves(
		self,
		state: GameState,
		moves: Sequence[Move],
		player: Player,
	) -> list[Move]:
		board = state.board
		opponent_wins = self._winning_columns(board, player.opponent)

		def priority(move: Move) -> tuple[int, int, int]:
			group, threats = self._tactical_key(
				board,
				move.column,
				player,
				opponent_wins,
			)
			side = (
				move.column
				if player is Player.RED
				else -move.column
			)

			return (group, threats, side)

		return sorted(moves, key=priority)


class RandomizedConnectFourMoveOrdering(
	ConnectFourMoveOrdering
):
	"""Order Connect Four moves tactically, randomly shuffling
	moves within each priority group.

	Moves are grouped by tactical priority as in
	ConnectFourMoveOrdering. The order within each group
	is random.

	The state is not modified.
	"""

	def __init__(
		self,
		rng: random.Random | None = None,
	) -> None:
		self.rng = rng or random.Random()

	def order_moves(
		self,
		state: GameState,
		moves: Sequence[Move],
		player: Player,
	) -> list[Move]:
		board = state.board
		opponent_wins = self._winning_columns(board, player.opponent)

		groups: dict[tuple[int, int], list[Move]] = {}

		for move in moves:
			key = self._tactical_key(
				board,
				move.column,
				player,
				opponent_wins,
			)

			groups.setdefault(key, []).append(move)

		ordered: list[Move] = []

		for key in sorted(groups):
			group = groups[key]
			self.rng.shuffle(group)
			ordered.extend(group)

		return ordered


class RandomConnectFourMoveOrdering(
	MoveOrdering[GameState, Move, Player]
):
	"""Order Connect Four moves randomly, ignoring tactical
	priorities and priority groups.

	The state is not modified.
	"""

	def __init__(
		self,
		rng: random.Random | None = None,
	) -> None:
		self.rng = rng or random.Random()

	def order_moves(
		self,
		state: GameState,
		moves: Sequence[Move],
		player: Player,
	) -> list[Move]:
		ordered = list(moves)
		self.rng.shuffle(ordered)

		return ordered

