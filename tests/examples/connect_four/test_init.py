import pytest

import connect_four
from connect_four import canonicalizer, evaluator, game, move_ordering


EXPECTED_EXPORTS = {
	"BOARD_ROWS": game,
	"BOARD_COLUMNS": game,
	"CONNECT_LENGTH": game,
	"ConnectFourGame": game,
	"GameState": game,
	"Move": game,
	"Player": game,
	"UndoData": game,
	"ConnectFourEvaluator": evaluator,
	"ConnectFourCanonicalizer": canonicalizer,
	"ConnectFourMoveOrdering": move_ordering,
	"RandomizedConnectFourMoveOrdering": move_ordering,
	"RandomConnectFourMoveOrdering": move_ordering,
}


class TestConnectFourPackageExports:
	def test_all_declares_expected_public_names(self) -> None:
		assert set(connect_four.__all__) == set(EXPECTED_EXPORTS)

	def test_all_contains_no_duplicates(self) -> None:
		assert len(connect_four.__all__) == len(set(connect_four.__all__))

	@pytest.mark.parametrize("name", sorted(EXPECTED_EXPORTS))
	def test_exports_object_from_defining_module(self, name: str) -> None:
		defining_module = EXPECTED_EXPORTS[name]

		assert getattr(connect_four, name) is getattr(defining_module, name)
