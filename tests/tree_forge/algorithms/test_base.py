from unittest.mock import patch

import pytest

from tree_forge.algorithms import minimax as minimax_module
from tree_forge.algorithms._base import _Counters, _Node, _TreeSearch
from tree_forge.algorithms.minimax import Minimax
from tree_forge.caching import TranspositionTable
from tree_forge.core.models import (
	SearchLimits,
	SearchTerminationReason,
)

from .conftest import (
	InvalidMoveOrdering,
	ReversingMoveOrdering,
	TreeCanonicalizer,
	TreeGame,
	TreeMove,
)


class FixedSearch(_TreeSearch):
	"""Minimal algorithm that records how the base class calls it."""

	def __init__(self, *arguments, **keywords) -> None:
		super().__init__(*arguments, **keywords)

		self.calls = []

	def _search_root(self, state, root_depth, counters):
		self.calls.append((state, root_depth))

		counters.nodes = 5
		counters.leaves = 3
		counters.terminals = 1
		counters.cutoffs = 2
		counters.max_ply = 4

		return _Node(1.5, False, TreeMove("a"))


class TestTreeSearchContract:
	def test_the_base_class_cannot_be_instantiated(
		self,
		game: TreeGame,
	) -> None:
		with pytest.raises(TypeError):
			_TreeSearch(game)

	def test_minimax_is_built_on_the_base_class(self) -> None:
		assert issubclass(Minimax, _TreeSearch)

	def test_the_base_class_is_not_exported(self) -> None:
		import tree_forge.algorithms as algorithms

		assert "_TreeSearch" not in algorithms.__all__
		assert not hasattr(minimax_module, "perf_counter")


class TestTreeSearchConfiguration:
	def test_defaults(self, game: TreeGame) -> None:
		search = FixedSearch(game)

		assert search.game is game
		assert search.limits == SearchLimits()
		assert search.alpha_beta is False
		assert search.canonicalizer is None
		assert search.transposition_table is None
		assert search.move_ordering is None
		assert search.evaluator is not None

	def test_a_table_requires_a_canonicalizer(self, game: TreeGame) -> None:
		with pytest.raises(ValueError, match="canonicalizer"):
			FixedSearch(game, transposition_table=TranspositionTable())

	def test_a_canonicalizer_creates_a_table(self, game: TreeGame) -> None:
		search = FixedSearch(game, canonicalizer=TreeCanonicalizer())

		assert isinstance(search.transposition_table, TranspositionTable)

	def test_a_supplied_table_is_used(self, game: TreeGame) -> None:
		table = TranspositionTable()

		search = FixedSearch(
			game,
			canonicalizer=TreeCanonicalizer(),
			transposition_table=table,
		)

		assert search.transposition_table is table


class TestTreeSearchResult:
	def test_the_root_is_searched_to_the_depth_limit(
		self,
		game: TreeGame,
	) -> None:
		search = FixedSearch(game, limits=SearchLimits(max_depth=3))
		state = game.initial_state()

		search.search(state)

		assert search.calls == [(state, 3)]

	def test_an_unlimited_depth_is_passed_on_as_none(
		self,
		game: TreeGame,
	) -> None:
		search = FixedSearch(game, limits=SearchLimits(max_depth=None))
		state = game.initial_state()

		search.search(state)

		assert search.calls == [(state, None)]

	def test_the_result_reports_the_root_node(self, game: TreeGame) -> None:
		result = FixedSearch(game).search(game.initial_state())

		assert result.best_move == TreeMove("a")
		assert result.score == 1.5
		assert result.completed is True
		assert result.termination_reason is SearchTerminationReason.COMPLETED

	def test_the_statistics_are_built_from_the_counters(
		self,
		game: TreeGame,
	) -> None:
		search = FixedSearch(
			game,
			alpha_beta=True,
			limits=SearchLimits(max_depth=3),
		)

		statistics = search.search(game.initial_state()).statistics

		assert statistics.nodes_visited == 5
		assert statistics.leaf_nodes == 3
		assert statistics.terminal_nodes == 1
		assert statistics.cutoffs == 2
		assert statistics.max_ply_reached == 4
		assert statistics.completed_depth == 3
		assert statistics.iterations_completed is None

	def test_cutoffs_are_not_measured_without_alpha_beta(
		self,
		game: TreeGame,
	) -> None:
		statistics = FixedSearch(game).search(game.initial_state()).statistics

		assert statistics.cutoffs is None

	def test_table_statistics_are_not_measured_without_a_table(
		self,
		game: TreeGame,
	) -> None:
		statistics = FixedSearch(game).search(game.initial_state()).statistics

		assert statistics.transposition_table_hits is None
		assert statistics.transposition_table_stores is None
		assert statistics.transposition_table_invalidations is None
		assert statistics.transposition_table_refreshes is None
		assert statistics.transposition_table_size is None

	def test_table_statistics_are_measured_with_a_table(
		self,
		game: TreeGame,
	) -> None:
		search = FixedSearch(game, canonicalizer=TreeCanonicalizer())

		statistics = search.search(game.initial_state()).statistics

		assert statistics.transposition_table_hits == 0
		assert statistics.transposition_table_stores == 0
		assert statistics.transposition_table_size == 0
		assert statistics.transposition_table_invalidations is None
		assert statistics.transposition_table_refreshes is None

	def test_dependency_statistics_need_dependency_tracking(
		self,
		game: TreeGame,
	) -> None:
		search = FixedSearch(
			game,
			canonicalizer=TreeCanonicalizer(),
			transposition_table=TranspositionTable(dependency_tracking=True),
		)

		statistics = search.search(game.initial_state()).statistics

		assert statistics.transposition_table_invalidations == 0
		assert statistics.transposition_table_refreshes == 0

	def test_elapsed_time_is_measured_once_around_the_search(
		self,
		game: TreeGame,
	) -> None:
		with patch(
			"tree_forge.algorithms._base.perf_counter",
			side_effect=[10.0, 12.5],
		):
			result = FixedSearch(game).search(game.initial_state())

		assert result.statistics.elapsed_time_seconds == 2.5


class TestTreeSearchMoveOrdering:
	def moves(self, game: TreeGame):
		return game.legal_moves(game.initial_state())

	def test_without_an_ordering_the_moves_keep_their_order(
		self,
		game: TreeGame,
	) -> None:
		search = FixedSearch(game)
		state = game.initial_state()
		moves = self.moves(game)

		assert search._order_moves(state, moves, 1) == list(moves)

	def test_an_ordering_rearranges_the_moves(self, game: TreeGame) -> None:
		search = FixedSearch(game, move_ordering=ReversingMoveOrdering())
		state = game.initial_state()
		moves = self.moves(game)

		assert search._order_moves(state, moves, 1) == list(reversed(moves))

	@pytest.mark.parametrize(
		"mode",
		["add", "omit", "duplicate", "substitute"],
	)
	def test_an_ordering_must_not_change_the_moves(
		self,
		game: TreeGame,
		mode: str,
	) -> None:
		search = FixedSearch(game, move_ordering=InvalidMoveOrdering(mode))
		state = game.initial_state()

		with pytest.raises(ValueError, match="MoveOrdering"):
			search._order_moves(state, self.moves(game), 1)


class TestCounters:
	def test_every_counter_starts_at_zero(self) -> None:
		counters = _Counters()

		assert [getattr(counters, name) for name in _Counters.__slots__] == (
			[0] * len(_Counters.__slots__)
		)
