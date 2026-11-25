import pytest

import tic_tac_toe
from tic_tac_toe import canonicalizer, evaluator, game, move_ordering


EXPECTED_EXPORTS = {
	"BOARD_SIZE": game,
	"GameState": game,
	"Move": game,
	"Player": game,
	"TicTacToeGame": game,
	"UndoData": game,
	"TicTacToeEvaluator": evaluator,
	"TicTacToeCanonicalizer": canonicalizer,
	"TicTacToeMoveOrdering": move_ordering,
	"RandomizedTicTacToeMoveOrdering": move_ordering,
	"RandomTicTacToeMoveOrdering": move_ordering,
}


class TestTicTacToePackageExports:
	def test_all_declares_expected_public_names(self) -> None:
		assert set(tic_tac_toe.__all__) == set(EXPECTED_EXPORTS)

	def test_all_contains_no_duplicates(self) -> None:
		assert len(tic_tac_toe.__all__) == len(set(tic_tac_toe.__all__))

	@pytest.mark.parametrize("name", sorted(EXPECTED_EXPORTS))
	def test_exports_object_from_defining_module(self, name: str) -> None:
		defining_module = EXPECTED_EXPORTS[name]

		assert getattr(tic_tac_toe, name) is getattr(defining_module, name)
