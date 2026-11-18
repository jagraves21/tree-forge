from collections.abc import Callable, Iterable

import pytest

from connect_four import ConnectFourGame, GameState, Move, Player


SYMBOLS = {
	".": None,
	"R": Player.RED,
	"Y": Player.YELLOW,
}


@pytest.fixture
def game() -> ConnectFourGame:
	return ConnectFourGame()


@pytest.fixture
def make_state() -> Callable[..., GameState]:
	"""Build a state from six rows, top first, using "." for empty cells.

	The move count defaults to the number of pieces on the board. Pieces
	are placed exactly as written, so gravity is not enforced.
	"""

	def factory(
		*rows: str,
		current_player: Player = Player.RED,
		move_count: int | None = None,
	) -> GameState:
		board = [[SYMBOLS[cell] for cell in row] for row in rows]

		if move_count is None:
			move_count = sum(
				cell is not None
				for row in board
				for cell in row
			)

		return GameState(
			board=board,
			current_player=current_player,
			move_count=move_count,
		)

	return factory


@pytest.fixture
def play(game: ConnectFourGame) -> Callable[..., GameState]:
	"""Play a sequence of columns from the initial position."""

	def factory(columns: Iterable[int]) -> GameState:
		state = game.initial_state()

		for column in columns:
			game.make_move(state, Move(column))

		return state

	return factory
