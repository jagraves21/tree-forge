from collections.abc import Callable

import pytest

from tic_tac_toe import GameState, Player, TicTacToeGame


SYMBOLS = {
	".": None,
	"X": Player.X,
	"O": Player.O,
}


@pytest.fixture
def game() -> TicTacToeGame:
	return TicTacToeGame()


@pytest.fixture
def make_state() -> Callable[..., GameState]:
	"""Build a state from rows such as "XO.", using "." for empty cells."""

	def factory(
		*rows: str,
		current_player: Player = Player.X,
	) -> GameState:
		return GameState(
			board=[[SYMBOLS[cell] for cell in row] for row in rows],
			current_player=current_player,
		)

	return factory
