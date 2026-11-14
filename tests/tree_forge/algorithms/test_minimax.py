from collections.abc import Sequence

import pytest

from tree_forge.algorithms.minimax import Minimax
from tree_forge.core import MoveOrdering, SearchLimits

from .conftest import SimpleGame, SimpleMove, SimpleState


class ReverseMoveOrdering(MoveOrdering[SimpleState, SimpleMove, int]):
	"""Reverses legal move order and records the positions visited."""

	def __init__(self) -> None:
		self.ordered_positions: list[str] = []

	def order_moves(
		self,
		state: SimpleState,
		moves: Sequence[SimpleMove],
		player: int,
	) -> Sequence[SimpleMove]:
		self.ordered_positions.append(state.node)
		return list(reversed(moves))


class InvalidMoveOrdering(MoveOrdering[SimpleState, SimpleMove, int]):
	"""Returns an invalid move ordering for validation tests."""

	def __init__(self, omit_move: bool = False) -> None:
		self.omit_move = omit_move

	def order_moves(
		self,
		state: SimpleState,
		moves: Sequence[SimpleMove],
		player: int,
	) -> Sequence[SimpleMove]:
		if self.omit_move:
			return list(moves[:-1])

		return [*moves, SimpleMove("invalid")]


class TestMinimaxMoveOrdering:
	@pytest.fixture(autouse=True)
	def setup(self, game: SimpleGame) -> None:
		self.game = game

	def test_move_ordering_preserves_minimax_result(self) -> None:
		plain_search = Minimax(
			game=self.game,
			limits=SearchLimits(max_depth=2),
		)

		ordering = ReverseMoveOrdering()

		ordered_search = Minimax(
			game=self.game,
			limits=SearchLimits(max_depth=2),
			move_ordering=ordering,
		)

		plain_result = plain_search.search(self.game.initial_state())
		ordered_result = ordered_search.search(self.game.initial_state())

		assert ordered_result.score == plain_result.score
		assert ordered_result.best_move == plain_result.best_move

	def test_move_ordering_is_applied_at_root(self) -> None:
		ordering = ReverseMoveOrdering()

		search = Minimax(
			game=self.game,
			limits=SearchLimits(max_depth=2),
			move_ordering=ordering,
		)

		search.search(self.game.initial_state())

		assert ordering.ordered_positions[0] == "root"

	def test_move_ordering_is_applied_at_recursive_nodes(self) -> None:
		ordering = ReverseMoveOrdering()

		search = Minimax(
			game=self.game,
			limits=SearchLimits(max_depth=2),
			move_ordering=ordering,
		)

		search.search(self.game.initial_state())

		assert "a" in ordering.ordered_positions
		assert "b" in ordering.ordered_positions

	def test_move_ordering_preserves_result_with_alpha_beta(self) -> None:
		plain_search = Minimax(
			game=self.game,
			limits=SearchLimits(max_depth=2),
			alpha_beta=True,
		)

		ordered_search = Minimax(
			game=self.game,
			limits=SearchLimits(max_depth=2),
			alpha_beta=True,
			move_ordering=ReverseMoveOrdering(),
		)

		plain_result = plain_search.search(self.game.initial_state())
		ordered_result = ordered_search.search(self.game.initial_state())

		assert ordered_result.score == plain_result.score
		assert ordered_result.best_move == plain_result.best_move

	def test_move_ordering_rejects_added_moves(self) -> None:
		search = Minimax(
			game=self.game,
			limits=SearchLimits(max_depth=2),
			move_ordering=InvalidMoveOrdering(),
		)

		with pytest.raises(ValueError):
			search.search(self.game.initial_state())

	def test_move_ordering_rejects_omitted_moves(self) -> None:
		search = Minimax(
			game=self.game,
			limits=SearchLimits(max_depth=2),
			move_ordering=InvalidMoveOrdering(omit_move=True),
		)

		with pytest.raises(ValueError):
			search.search(self.game.initial_state())

