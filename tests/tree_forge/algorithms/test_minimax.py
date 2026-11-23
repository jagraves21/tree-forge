from itertools import permutations
from unittest.mock import patch

import pytest

from tree_forge.algorithms.minimax import Minimax
from tree_forge.caching import (
	TranspositionBound,
	TranspositionEntry,
	TranspositionTable,
)
from tree_forge.core.models import (
	SearchLimits,
	SearchResult,
	SearchTerminationReason,
)

from .conftest import (
	DEEP_SCORES,
	DEEP_TREE,
	SHALLOW_SCORES,
	SHALLOW_TREE,
	TRANSPOSITION_SCORES,
	TRANSPOSITION_TREE,
	InvalidMoveOrdering,
	PrefixingCanonicalizer,
	RaisingEvaluator,
	ReversingMoveOrdering,
	TableEvaluator,
	TreeCanonicalizer,
	TreeGame,
	TreeMove,
	TreeState,
	TupleMoveOrdering,
)


cached = TranspositionEntry.from_bound


def make_search(game: TreeGame, depth: int | None = 2, **kwargs) -> Minimax:
	return Minimax(game, limits=SearchLimits(max_depth=depth), **kwargs)


class TestMinimaxInit:
	def test_default_configuration(self, game: TreeGame) -> None:
		search = Minimax(game)

		assert search.game is game
		assert search.evaluator is None
		assert search.limits == SearchLimits()
		assert search.alpha_beta is False
		assert search.canonicalizer is None
		assert search.transposition_table is None
		assert search.move_ordering is None

	def test_stores_supplied_configuration(self, game: TreeGame) -> None:
		evaluator = TableEvaluator()
		limits = SearchLimits(max_depth=7)
		ordering = ReversingMoveOrdering()

		search = Minimax(
			game,
			evaluator=evaluator,
			limits=limits,
			alpha_beta=True,
			move_ordering=ordering,
		)

		assert search.evaluator is evaluator
		assert search.limits is limits
		assert search.alpha_beta is True
		assert search.move_ordering is ordering

	def test_uses_default_limits_when_none_are_supplied(
		self,
		game: TreeGame,
	) -> None:
		assert Minimax(game, limits=None).limits == SearchLimits()

	def test_table_without_canonicalizer_is_rejected(
		self,
		game: TreeGame,
	) -> None:
		with pytest.raises(ValueError, match="canonicalizer is required"):
			Minimax(game, transposition_table=TranspositionTable())

	def test_canonicalizer_creates_a_default_table(
		self,
		game: TreeGame,
	) -> None:
		search = Minimax(game, canonicalizer=TreeCanonicalizer())

		assert isinstance(search.transposition_table, TranspositionTable)
		assert len(search.transposition_table) == 0
		assert search.transposition_table.dependency_tracking is False

	def test_default_tables_are_not_shared_between_searches(
		self,
		game: TreeGame,
	) -> None:
		canonicalizer = TreeCanonicalizer()

		first = Minimax(game, canonicalizer=canonicalizer)
		second = Minimax(game, canonicalizer=canonicalizer)

		assert first.transposition_table is not second.transposition_table

	def test_uses_supplied_table(self, game: TreeGame) -> None:
		table = TranspositionTable(dependency_tracking=True)

		search = Minimax(
			game,
			canonicalizer=TreeCanonicalizer(),
			transposition_table=table,
		)

		assert search.transposition_table is table


class TestMinimaxSearch:
	def test_returns_best_move_and_score(self, game: TreeGame) -> None:
		result = make_search(game).search(game.initial_state())

		assert isinstance(result, SearchResult)
		assert result.best_move == TreeMove("a")
		assert result.score == 3.0

	def test_reports_a_completed_search(self, game: TreeGame) -> None:
		result = make_search(game).search(game.initial_state())

		assert result.completed is True
		assert result.termination_reason is SearchTerminationReason.COMPLETED

	def test_alternates_between_maximizing_and_minimizing(
		self,
		deep_game: TreeGame,
	) -> None:
		result = make_search(deep_game, depth=3).search(
			deep_game.initial_state()
		)

		assert result.best_move == TreeMove("A")
		assert result.score == 5.0

	def test_scores_from_the_perspective_of_the_player_to_move(
		self,
		game: TreeGame,
	) -> None:
		# For player 2 the leaves are worth -3, -5, -2, and -9, so the
		# values of "a" and "b" are -5 and -9.
		state = TreeState(current_player=2)

		result = make_search(game).search(state)

		assert result.best_move == TreeMove("a")
		assert result.score == -5.0

	def test_searches_from_a_position_other_than_the_start(
		self,
		game: TreeGame,
	) -> None:
		state = TreeState(node="a", current_player=2)

		result = make_search(game, depth=1).search(state)

		assert result.best_move == TreeMove("a1")
		assert result.score == -3.0

	def test_first_best_move_wins_ties(self) -> None:
		game = TreeGame(
			SHALLOW_TREE,
			{"a1": 1.0, "a2": 1.0, "b1": 1.0, "b2": 1.0},
		)

		result = make_search(game).search(game.initial_state())

		assert result.best_move == TreeMove("a")
		assert result.score == 1.0

	def test_depth_beyond_the_tree_is_harmless(self, game: TreeGame) -> None:
		result = make_search(game, depth=10).search(game.initial_state())

		assert result.best_move == TreeMove("a")
		assert result.score == 3.0

	def test_unlimited_depth_searches_to_terminal_positions(
		self,
		game: TreeGame,
	) -> None:
		result = make_search(game, depth=None).search(game.initial_state())

		assert result.best_move == TreeMove("a")
		assert result.score == 3.0

	def test_depth_limit_stops_before_terminal_positions(
		self,
		game: TreeGame,
	) -> None:
		evaluator = TableEvaluator({"a": 1.0, "b": 4.0})

		result = make_search(game, depth=1, evaluator=evaluator).search(
			game.initial_state()
		)

		assert result.best_move == TreeMove("b")
		assert result.score == 4.0

	def test_repeated_searches_give_identical_results(
		self,
		game: TreeGame,
	) -> None:
		search = make_search(game)

		first = search.search(game.initial_state())
		second = search.search(game.initial_state())

		assert first.best_move == second.best_move
		assert first.score == second.score
		assert first.statistics.nodes_visited == (
			second.statistics.nodes_visited
		)
		assert first.statistics is not second.statistics

	def test_default_depth_limit_applies_when_no_limits_are_given(
		self,
		deep_game: TreeGame,
	) -> None:
		result = Minimax(deep_game).search(deep_game.initial_state())

		assert result.statistics.completed_depth == 4


class TestMinimaxRootPositions:
	def test_terminal_root_has_no_move_and_its_terminal_score(self) -> None:
		game = TreeGame({}, {"root": 7.0})

		result = make_search(game).search(game.initial_state())

		assert result.best_move is None
		assert result.score == 7.0
		assert result.completed is True
		assert result.termination_reason is SearchTerminationReason.COMPLETED

	def test_terminal_root_is_scored_for_the_player_to_move(self) -> None:
		game = TreeGame({}, {"root": 7.0})

		result = make_search(game).search(TreeState(current_player=2))

		assert result.score == -7.0

	def test_terminal_root_statistics(self) -> None:
		game = TreeGame({}, {"root": 7.0})

		statistics = make_search(game).search(game.initial_state()).statistics

		assert statistics.nodes_visited == 1
		assert statistics.leaf_nodes == 1
		assert statistics.terminal_nodes == 1
		assert statistics.max_ply_reached == 0

	def test_root_without_moves_has_no_move_and_a_neutral_score(self) -> None:
		game = TreeGame({}, {})

		result = make_search(game).search(game.initial_state())

		assert result.best_move is None
		assert result.score == 0.0
		assert result.completed is True

	def test_root_without_moves_uses_the_evaluator(self) -> None:
		game = TreeGame({}, {})
		evaluator = TableEvaluator({"root": 2.5})

		result = make_search(game, evaluator=evaluator).search(
			game.initial_state()
		)

		assert result.score == 2.5
		assert evaluator.calls == [("root", 1)]

	def test_root_without_moves_statistics(self) -> None:
		game = TreeGame({}, {})

		statistics = make_search(game).search(game.initial_state()).statistics

		assert statistics.nodes_visited == 1
		assert statistics.leaf_nodes == 1
		assert statistics.terminal_nodes == 0

	def test_interior_position_without_moves_is_evaluated(self) -> None:
		# "dead" is neither terminal nor expandable. "ok" leads to a
		# terminal score of 5 for the maximizing player.
		game = TreeGame(
			{"root": ("dead", "ok"), "ok": ("ok1",)},
			{"ok1": 5.0},
		)
		evaluator = TableEvaluator({"dead": 2.0})

		result = make_search(game, depth=3, evaluator=evaluator).search(
			game.initial_state()
		)

		assert result.best_move == TreeMove("ok")
		assert result.score == 5.0
		assert evaluator.calls == [("dead", 1)]
		assert result.statistics.leaf_nodes == 2
		assert result.statistics.terminal_nodes == 1


class TestMinimaxEvaluator:
	def test_leaves_are_scored_zero_without_an_evaluator(
		self,
		game: TreeGame,
	) -> None:
		result = make_search(game, depth=1).search(game.initial_state())

		assert result.score == 0.0
		assert result.best_move == TreeMove("a")

	def test_evaluator_is_called_at_the_depth_limit_only(
		self,
		deep_game: TreeGame,
	) -> None:
		evaluator = TableEvaluator()

		make_search(deep_game, depth=2, evaluator=evaluator).search(
			deep_game.initial_state()
		)

		assert sorted(node for node, _ in evaluator.calls) == [
			"A1",
			"A2",
			"B1",
			"B2",
		]

	def test_evaluator_always_receives_the_root_player(
		self,
		deep_game: TreeGame,
	) -> None:
		evaluator = TableEvaluator()

		make_search(deep_game, depth=1, evaluator=evaluator).search(
			deep_game.initial_state()
		)

		assert evaluator.calls == [("A", 1), ("B", 1)]

	def test_evaluator_is_not_called_for_terminal_positions(
		self,
		game: TreeGame,
	) -> None:
		evaluator = TableEvaluator()

		make_search(game, depth=5, evaluator=evaluator).search(
			game.initial_state()
		)

		assert evaluator.calls == []

	def test_evaluated_scores_are_minimaxed_through_the_tree(
		self,
		deep_game: TreeGame,
	) -> None:
		# A = min(1, 4) = 1 and B = min(0, 3) = 0.
		evaluator = TableEvaluator({"A1": 1.0, "A2": 4.0, "B1": 0.0, "B2": 3.0})

		result = make_search(deep_game, depth=2, evaluator=evaluator).search(
			deep_game.initial_state()
		)

		assert result.best_move == TreeMove("A")
		assert result.score == 1.0


class TestMinimaxStatistics:
	def test_plain_search_counts(self, game: TreeGame) -> None:
		statistics = make_search(game).search(game.initial_state()).statistics

		# The root and its six descendants.
		assert statistics.nodes_visited == 7
		assert statistics.leaf_nodes == 4
		assert statistics.terminal_nodes == 4
		assert statistics.max_ply_reached == 2
		assert statistics.completed_depth == 2

	def test_every_position_is_counted_including_the_root(self) -> None:
		game = TreeGame({"root": ("x",)}, {"x": 1.0})

		statistics = make_search(game, depth=3).search(
			game.initial_state()
		).statistics

		assert statistics.nodes_visited == 2

	def test_alpha_beta_prunes_the_second_reply_of_a_refuted_move(
		self,
		game: TreeGame,
	) -> None:
		# After "a" scores 3, "b1" scores 2 and refutes "b", so "b2" is
		# never searched: root, a, a1, a2, b, b1.
		statistics = make_search(game, alpha_beta=True).search(
			game.initial_state()
		).statistics

		assert statistics.nodes_visited == 6
		assert statistics.leaf_nodes == 3
		assert statistics.terminal_nodes == 3
		assert statistics.cutoffs == 1

	def test_alpha_beta_cuts_off_when_the_window_collapses_to_a_point(
		self,
	) -> None:
		# "a" is worth 1. At "b" the first reply is also worth 1, which
		# makes beta equal to alpha, so the second reply is skipped even
		# though it cannot be shown to be worse.
		game = TreeGame(SHALLOW_TREE, dict.fromkeys(SHALLOW_SCORES, 1.0))

		statistics = make_search(game, alpha_beta=True).search(
			game.initial_state()
		).statistics

		assert statistics.nodes_visited == 6
		assert statistics.leaf_nodes == 3
		assert statistics.cutoffs == 1
		assert game.made == ["a", "a1", "a2", "b", "b1"]

	def test_alpha_beta_cutoffs_in_a_deeper_tree(
		self,
		deep_game: TreeGame,
	) -> None:
		# Cutoffs: after A1 = 5, the first leaf of A2 (6) exceeds the
		# minimizer's bound of 5; after A = 5, B1 = 2 refutes B.
		statistics = make_search(
			deep_game,
			depth=3,
			alpha_beta=True,
		).search(deep_game.initial_state()).statistics

		assert statistics.nodes_visited == 11
		assert statistics.leaf_nodes == 5
		assert statistics.cutoffs == 2

	def test_plain_search_of_the_deeper_tree_visits_every_node(
		self,
		deep_game: TreeGame,
	) -> None:
		statistics = make_search(deep_game, depth=3).search(
			deep_game.initial_state()
		).statistics

		assert statistics.nodes_visited == 15
		assert statistics.leaf_nodes == 8
		assert statistics.terminal_nodes == 8
		assert statistics.max_ply_reached == 3

	def test_depth_limit_leaves_are_not_terminal(
		self,
		game: TreeGame,
	) -> None:
		statistics = make_search(game, depth=1).search(
			game.initial_state()
		).statistics

		assert statistics.nodes_visited == 3
		assert statistics.leaf_nodes == 2
		assert statistics.terminal_nodes == 0
		assert statistics.max_ply_reached == 1

	def test_leaf_nodes_include_every_terminal_node(
		self,
		deep_game: TreeGame,
	) -> None:
		statistics = make_search(deep_game, depth=3).search(
			deep_game.initial_state()
		).statistics

		assert statistics.leaf_nodes >= statistics.terminal_nodes

	def test_completed_depth_is_the_configured_depth(
		self,
		game: TreeGame,
	) -> None:
		statistics = make_search(game, depth=10).search(
			game.initial_state()
		).statistics

		assert statistics.completed_depth == 10

	def test_max_ply_reached_follows_the_deepest_position(
		self,
		deep_game: TreeGame,
	) -> None:
		statistics = make_search(deep_game, depth=3).search(
			deep_game.initial_state()
		).statistics

		assert statistics.max_ply_reached == 3

	@pytest.mark.parametrize("tree, scores", [
		pytest.param(SHALLOW_TREE, SHALLOW_SCORES, id="search"),
		pytest.param({}, {"root": 1.0}, id="terminal-root"),
		pytest.param({}, {}, id="root-without-moves"),
	])
	def test_elapsed_time_is_measured_with_perf_counter(
		self,
		tree: dict,
		scores: dict,
	) -> None:
		game = TreeGame(tree, scores)

		with patch(
			"tree_forge.algorithms.minimax.perf_counter",
			side_effect=[10.0, 12.5],
		):
			result = make_search(game).search(game.initial_state())

		assert result.statistics.elapsed_time_seconds == 2.5

	def test_real_elapsed_time_is_non_negative(self, game: TreeGame) -> None:
		result = make_search(game).search(game.initial_state())

		assert result.statistics.elapsed_time_seconds >= 0.0


class TestMinimaxStatisticsAvailability:
	"""A statistic that was not measured is None, and zero only if measured."""

	CACHE_STATISTICS = [
		"transposition_table_hits",
		"transposition_table_stores",
		"transposition_table_size",
	]

	DEPENDENCY_STATISTICS = [
		"transposition_table_invalidations",
		"transposition_table_refreshes",
	]

	@pytest.fixture
	def single_line(self) -> TreeGame:
		# Nothing can be pruned: every position has a single reply.
		return TreeGame({"root": ("a",), "a": ("a1",)}, {"a1": 1.0})

	def test_a_plain_search_does_not_measure_cutoffs(
		self,
		game: TreeGame,
	) -> None:
		statistics = make_search(game).search(game.initial_state()).statistics

		assert statistics.cutoffs is None

	def test_alpha_beta_that_prunes_nothing_reports_zero_cutoffs(
		self,
		single_line: TreeGame,
	) -> None:
		statistics = make_search(single_line, depth=3, alpha_beta=True).search(
			single_line.initial_state()
		).statistics

		assert statistics.cutoffs == 0
		assert statistics.cutoffs is not None

	@pytest.mark.parametrize("name", CACHE_STATISTICS + DEPENDENCY_STATISTICS)
	def test_cache_statistics_are_not_measured_without_a_table(
		self,
		game: TreeGame,
		name: str,
	) -> None:
		statistics = make_search(game).search(game.initial_state()).statistics

		assert getattr(statistics, name) is None

	@pytest.mark.parametrize("name", CACHE_STATISTICS)
	def test_cache_statistics_are_measured_with_a_table(
		self,
		game: TreeGame,
		name: str,
	) -> None:
		search = make_search(game, canonicalizer=TreeCanonicalizer())

		statistics = search.search(game.initial_state()).statistics

		assert getattr(statistics, name) is not None

	@pytest.mark.parametrize("name", DEPENDENCY_STATISTICS)
	def test_dependency_statistics_need_dependency_tracking(
		self,
		game: TreeGame,
		name: str,
	) -> None:
		search = make_search(game, canonicalizer=TreeCanonicalizer())

		statistics = search.search(game.initial_state()).statistics

		assert getattr(statistics, name) is None

	@pytest.mark.parametrize("name", DEPENDENCY_STATISTICS)
	def test_dependency_statistics_are_measured_with_tracking(
		self,
		game: TreeGame,
		name: str,
	) -> None:
		search = make_search(
			game,
			canonicalizer=TreeCanonicalizer(),
			transposition_table=TranspositionTable(dependency_tracking=True),
		)

		statistics = search.search(game.initial_state()).statistics

		# Nothing was invalidated or refreshed, which is a measurement.
		assert getattr(statistics, name) == 0

	@pytest.mark.parametrize("tracking", [False, True])
	def test_table_size_is_the_number_of_entries_when_the_search_returns(
		self,
		game: TreeGame,
		tracking: bool,
	) -> None:
		table = TranspositionTable(dependency_tracking=tracking)
		search = make_search(
			game,
			canonicalizer=TreeCanonicalizer(),
			transposition_table=table,
		)

		statistics = search.search(game.initial_state()).statistics

		# The root and its six descendants.
		assert statistics.transposition_table_size == 7
		assert statistics.transposition_table_size == len(table)

	def test_table_size_includes_entries_from_earlier_searches(self) -> None:
		game = TreeGame(DISCOVERY_TREE, DISCOVERY_SCORES)
		table = TranspositionTable()
		search = Minimax(
			game,
			limits=SearchLimits(max_depth=None),
			alpha_beta=True,
			canonicalizer=TreeCanonicalizer(),
			transposition_table=table,
		)

		first = search.search(TreeState("root", 1)).statistics
		second = search.search(TreeState("B", 2)).statistics

		# Pruning leaves "a2" unsearched: the first search stores p1, P,
		# a1, A, and the root. The second adds "B" and "a2".
		assert first.transposition_table_size == 5
		assert second.transposition_table_size == 7

	def test_table_size_does_not_change_when_nothing_new_is_stored(
		self,
		game: TreeGame,
	) -> None:
		search = make_search(game, canonicalizer=TreeCanonicalizer())

		first = search.search(game.initial_state()).statistics
		second = search.search(game.initial_state()).statistics

		assert second.transposition_table_size == first.transposition_table_size

	def test_table_size_counts_positions_not_writes(self, game: TreeGame) -> None:
		search = make_search(game, canonicalizer=TreeCanonicalizer())

		statistics = search.search(game.initial_state()).statistics

		# Every position is written once here, but rewriting a position
		# would not make the table any bigger.
		assert statistics.transposition_table_size <= (
			statistics.transposition_table_stores
		)

	def test_table_size_is_the_total_not_only_what_this_search_added(
		self,
		game: TreeGame,
	) -> None:
		table = TranspositionTable()
		table.store(cached("unrelated", 1, 0.0, TranspositionBound.EXACT))
		search = make_search(
			game,
			canonicalizer=TreeCanonicalizer(),
			transposition_table=table,
		)

		statistics = search.search(game.initial_state()).statistics

		assert statistics.transposition_table_size == 8

	def test_a_first_search_measures_zero_hits(self, game: TreeGame) -> None:
		search = make_search(game, canonicalizer=TreeCanonicalizer())

		statistics = search.search(game.initial_state()).statistics

		assert statistics.transposition_table_hits == 0
		assert statistics.transposition_table_hits is not None

	def test_completed_depth_is_not_applicable_without_a_depth_limit(
		self,
		game: TreeGame,
	) -> None:
		statistics = make_search(game, depth=None).search(
			game.initial_state()
		).statistics

		assert statistics.completed_depth is None

	def test_iterations_are_not_applicable(self, game: TreeGame) -> None:
		statistics = make_search(game).search(game.initial_state()).statistics

		assert statistics.iterations_completed is None

	@pytest.mark.parametrize("name", [
		"nodes_visited",
		"leaf_nodes",
		"terminal_nodes",
		"max_ply_reached",
		"elapsed_time_seconds",
	])
	def test_basic_statistics_are_always_measured(
		self,
		game: TreeGame,
		name: str,
	) -> None:
		statistics = make_search(game).search(game.initial_state()).statistics

		assert getattr(statistics, name) is not None

	def test_zero_terminal_nodes_is_a_measurement(self, game: TreeGame) -> None:
		statistics = make_search(game, depth=1).search(
			game.initial_state()
		).statistics

		assert statistics.terminal_nodes == 0

	def test_statistics_of_different_configurations_merge_without_zeros(
		self,
		game: TreeGame,
	) -> None:
		plain = make_search(game).search(game.initial_state()).statistics
		pruned = make_search(game, alpha_beta=True).search(
			TreeGame(SHALLOW_TREE, SHALLOW_SCORES).initial_state()
		).statistics

		combined = plain.merge(pruned)

		assert combined.nodes_visited == 13
		assert combined.cutoffs == 1
		assert combined.transposition_table_hits is None


class TestMinimaxAlphaBeta:
	@pytest.mark.parametrize("depth", [1, 2, 3, 4, None])
	@pytest.mark.parametrize("root_player", [1, 2])
	def test_matches_plain_search_on_the_deep_tree(
		self,
		depth,
		root_player: int,
	) -> None:
		evaluator_values = {"A": 2.0, "B": -1.0, "A1": 4.0, "A2": -3.0, "B1": 0.0, "B2": 7.0}
		results = []

		for alpha_beta in (False, True):
			game = TreeGame(DEEP_TREE, DEEP_SCORES)
			search = make_search(
				game,
				depth=depth,
				alpha_beta=alpha_beta,
				evaluator=TableEvaluator(evaluator_values),
			)

			results.append(search.search(TreeState(current_player=root_player)))

		plain, pruned = results

		assert pruned.score == plain.score
		assert pruned.best_move == plain.best_move

	@pytest.mark.parametrize("tree, scores", [
		pytest.param(SHALLOW_TREE, SHALLOW_SCORES, id="shallow"),
		pytest.param(DEEP_TREE, DEEP_SCORES, id="deep"),
		pytest.param(
			TRANSPOSITION_TREE,
			TRANSPOSITION_SCORES,
			id="transposition",
		),
		pytest.param(
			SHALLOW_TREE,
			dict.fromkeys(SHALLOW_SCORES, 1.0),
			id="all-equal",
		),
	])
	def test_matches_plain_search_on_each_tree(
		self,
		tree: dict,
		scores: dict,
	) -> None:
		plain_game = TreeGame(tree, scores)
		pruned_game = TreeGame(tree, scores)

		plain = make_search(plain_game, depth=None).search(
			plain_game.initial_state()
		)
		pruned = make_search(pruned_game, depth=None, alpha_beta=True).search(
			pruned_game.initial_state()
		)

		assert pruned.score == plain.score
		assert pruned.best_move == plain.best_move

	def test_never_visits_more_nodes_than_plain_search(
		self,
		deep_game: TreeGame,
	) -> None:
		plain = make_search(deep_game, depth=3).search(
			deep_game.initial_state()
		)
		pruned = make_search(deep_game, depth=3, alpha_beta=True).search(
			deep_game.initial_state()
		)

		assert pruned.statistics.nodes_visited < plain.statistics.nodes_visited
		assert plain.statistics.cutoffs is None
		assert pruned.statistics.cutoffs > 0


class TestMinimaxStateHandling:
	def test_state_is_restored_after_search(self, deep_game: TreeGame) -> None:
		state = deep_game.initial_state()

		make_search(deep_game, depth=3).search(state)

		assert state == TreeState()

	def test_every_move_is_unmade(self, deep_game: TreeGame) -> None:
		make_search(deep_game, depth=3, alpha_beta=True).search(
			deep_game.initial_state()
		)

		assert len(deep_game.made) == len(deep_game.unmade)
		assert sorted(deep_game.made) == sorted(deep_game.unmade)

	def test_state_is_restored_when_evaluation_fails(
		self,
		game: TreeGame,
	) -> None:
		state = game.initial_state()
		search = make_search(game, depth=1, evaluator=RaisingEvaluator())

		with pytest.raises(RuntimeError, match="evaluation failed"):
			search.search(state)

		assert state == TreeState()
		assert len(game.made) == len(game.unmade)

	def test_search_does_not_require_a_fresh_state_object(
		self,
		game: TreeGame,
	) -> None:
		state = game.initial_state()
		search = make_search(game)

		first = search.search(state)
		second = search.search(state)

		assert first.best_move == second.best_move

	def test_exploration_follows_legal_move_order_by_default(
		self,
		game: TreeGame,
	) -> None:
		make_search(game).search(game.initial_state())

		assert game.made == ["a", "a1", "a2", "b", "b1", "b2"]


class TestMinimaxMoveOrdering:
	def test_ordering_is_applied_at_the_root(self, game: TreeGame) -> None:
		ordering = ReversingMoveOrdering()

		make_search(game, move_ordering=ordering).search(game.initial_state())

		assert ordering.calls[0] == ("root", 1)

	def test_ordering_is_applied_at_interior_nodes_for_their_player(
		self,
		game: TreeGame,
	) -> None:
		ordering = ReversingMoveOrdering()

		make_search(game, move_ordering=ordering).search(game.initial_state())

		# Reversal makes the root visit "b" before "a".
		assert ordering.calls == [("root", 1), ("b", 2), ("a", 2)]

	def test_ordering_is_not_applied_to_terminal_positions(
		self,
		game: TreeGame,
	) -> None:
		ordering = ReversingMoveOrdering()

		make_search(game, move_ordering=ordering).search(game.initial_state())

		assert all(node in {"root", "a", "b"} for node, _ in ordering.calls)

	def test_ordering_determines_exploration_order(
		self,
		game: TreeGame,
	) -> None:
		make_search(game, move_ordering=ReversingMoveOrdering()).search(
			game.initial_state()
		)

		assert game.made == ["b", "b2", "b1", "a", "a2", "a1"]

	def test_ordering_determines_which_tied_move_is_chosen(self) -> None:
		game = TreeGame(
			SHALLOW_TREE,
			{"a1": 1.0, "a2": 1.0, "b1": 1.0, "b2": 1.0},
		)

		result = make_search(game, move_ordering=ReversingMoveOrdering()).search(
			game.initial_state()
		)

		assert result.best_move == TreeMove("b")

	@pytest.mark.parametrize("alpha_beta", [False, True])
	def test_ordering_does_not_change_the_score(
		self,
		deep_game: TreeGame,
		alpha_beta: bool,
	) -> None:
		plain = make_search(deep_game, depth=3, alpha_beta=alpha_beta).search(
			deep_game.initial_state()
		)
		ordered = make_search(
			deep_game,
			depth=3,
			alpha_beta=alpha_beta,
			move_ordering=ReversingMoveOrdering(),
		).search(deep_game.initial_state())

		assert ordered.score == plain.score
		assert ordered.best_move == plain.best_move

	def test_accepts_orderings_that_return_tuples(self, game: TreeGame) -> None:
		result = make_search(game, move_ordering=TupleMoveOrdering()).search(
			game.initial_state()
		)

		assert result.best_move == TreeMove("a")

	@pytest.mark.parametrize("mode", ["add", "omit", "duplicate", "substitute"])
	def test_rejects_orderings_that_change_the_legal_moves(
		self,
		game: TreeGame,
		mode: str,
	) -> None:
		search = make_search(game, move_ordering=InvalidMoveOrdering(mode))

		with pytest.raises(
			ValueError,
			match="MoveOrdering must not add or omit legal moves",
		):
			search.search(game.initial_state())

	def test_state_is_untouched_when_ordering_is_rejected(
		self,
		game: TreeGame,
	) -> None:
		state = game.initial_state()
		search = make_search(game, move_ordering=InvalidMoveOrdering("add"))

		with pytest.raises(ValueError):
			search.search(state)

		assert state == TreeState()


def seeded_search(game: TreeGame, cached_move: str, **kwargs) -> Minimax:
	"""A search whose table already remembers a best move for the root."""
	search = Minimax(
		game,
		limits=SearchLimits(max_depth=None),
		canonicalizer=TreeCanonicalizer(),
		**kwargs,
	)

	search.transposition_table.store(
		TranspositionEntry(
			key=("root", 1),
			depth=None,
			best_move=TreeMove(cached_move),
		)
	)

	return search


EQUAL_SCORES = dict.fromkeys(SHALLOW_SCORES, 1.0)

THREE_EQUAL_MOVES = {
	"tree": {
		"root": ("a", "b", "c"),
		"a": ("a1",),
		"b": ("b1",),
		"c": ("c1",),
	},
	"scores": {"a1": 1.0, "b1": 1.0, "c1": 1.0},
}


class TestMinimaxRootTies:
	"""Equally good root moves are decided by the move ordering alone."""

	@pytest.mark.parametrize("alpha_beta", [False, True])
	def test_a_cached_move_does_not_decide_between_equal_moves(
		self,
		alpha_beta: bool,
	) -> None:
		game = TreeGame(SHALLOW_TREE, EQUAL_SCORES)
		search = seeded_search(game, "b", alpha_beta=alpha_beta)

		result = search.search(game.initial_state())

		# "b" is cached, but "a" is first in the legal moves.
		assert result.best_move == TreeMove("a")

	@pytest.mark.parametrize("alpha_beta", [False, True])
	def test_the_move_ordering_decides_between_equal_moves(
		self,
		alpha_beta: bool,
	) -> None:
		game = TreeGame(SHALLOW_TREE, EQUAL_SCORES)
		search = seeded_search(
			game,
			"a",
			alpha_beta=alpha_beta,
			move_ordering=ReversingMoveOrdering(),
		)

		result = search.search(game.initial_state())

		assert result.best_move == TreeMove("b")

	@pytest.mark.parametrize("alpha_beta", [False, True])
	def test_the_choice_does_not_depend_on_what_was_searched_before(
		self,
		alpha_beta: bool,
	) -> None:
		game = TreeGame(SHALLOW_TREE, EQUAL_SCORES)
		table = TranspositionTable()

		def search(**kwargs) -> Minimax:
			return Minimax(
				game,
				limits=SearchLimits(max_depth=None),
				alpha_beta=alpha_beta,
				canonicalizer=TreeCanonicalizer(),
				transposition_table=table,
				**kwargs,
			)

		# An earlier search with the opposite preference fills the table.
		earlier = search(move_ordering=ReversingMoveOrdering()).search(
			game.initial_state()
		)
		warm = search().search(game.initial_state())
		fresh = make_search(
			TreeGame(SHALLOW_TREE, EQUAL_SCORES),
			depth=None,
			alpha_beta=alpha_beta,
			canonicalizer=TreeCanonicalizer(),
		).search(TreeState())

		assert earlier.best_move == TreeMove("b")
		assert warm.best_move == fresh.best_move == TreeMove("a")

	def test_a_cached_move_is_still_searched_first(self) -> None:
		game = TreeGame(SHALLOW_TREE, EQUAL_SCORES)
		search = seeded_search(game, "b")

		search.search(game.initial_state())

		# Searching it first is what makes pruning effective.
		assert game.made[0] == "b"

	@pytest.mark.parametrize("alpha_beta", [False, True])
	def test_a_better_move_beats_both_the_ordering_and_the_cache(
		self,
		alpha_beta: bool,
	) -> None:
		game = TreeGame(SHALLOW_TREE, SHALLOW_SCORES)
		search = seeded_search(
			game,
			"b",
			alpha_beta=alpha_beta,
			move_ordering=ReversingMoveOrdering(),
		)

		result = search.search(game.initial_state())

		# "a" is worth 3 and "b" only 2.
		assert result.best_move == TreeMove("a")
		assert result.score == 3.0

	@pytest.mark.parametrize("alpha_beta", [False, True])
	def test_an_earlier_move_that_only_looks_tied_does_not_win(
		self,
		alpha_beta: bool,
	) -> None:
		# "a" ranks first and its first reply is worth 1, the same as "b",
		# but its second reply is worth 0, so "a" is worth only 0.
		game = TreeGame(
			SHALLOW_TREE,
			{"a1": 1.0, "a2": 0.0, "b1": 1.0, "b2": 1.0},
		)
		search = seeded_search(game, "b", alpha_beta=alpha_beta)

		result = search.search(game.initial_state())

		assert result.best_move == TreeMove("b")
		assert result.score == 1.0

	@pytest.mark.parametrize("alpha_beta", [False, True])
	def test_an_earlier_move_that_really_is_tied_wins(
		self,
		alpha_beta: bool,
	) -> None:
		game = TreeGame(
			SHALLOW_TREE,
			{"a1": 1.0, "a2": 1.0, "b1": 1.0, "b2": 1.0},
		)
		search = seeded_search(game, "b", alpha_beta=alpha_beta)

		result = search.search(game.initial_state())

		assert result.best_move == TreeMove("a")

	@pytest.mark.parametrize("cached_move", ["a", "b", "c"])
	@pytest.mark.parametrize("alpha_beta", [False, True])
	def test_the_best_ranked_of_several_equal_moves_wins(
		self,
		cached_move: str,
		alpha_beta: bool,
	) -> None:
		game = TreeGame(
			THREE_EQUAL_MOVES["tree"],
			THREE_EQUAL_MOVES["scores"],
		)
		search = seeded_search(game, cached_move, alpha_beta=alpha_beta)

		assert search.search(game.initial_state()).best_move == TreeMove("a")

	@pytest.mark.parametrize("cached_move", ["a", "b", "c"])
	def test_the_ordering_ranks_all_equal_moves(self, cached_move: str) -> None:
		game = TreeGame(
			THREE_EQUAL_MOVES["tree"],
			THREE_EQUAL_MOVES["scores"],
		)
		search = seeded_search(
			game,
			cached_move,
			alpha_beta=True,
			move_ordering=ReversingMoveOrdering(),
		)

		assert search.search(game.initial_state()).best_move == TreeMove("c")

	def test_the_score_is_the_same_whichever_move_is_chosen(self) -> None:
		game = TreeGame(SHALLOW_TREE, SHALLOW_SCORES)

		scores = {
			seeded_search(
				game,
				cached,
				alpha_beta=alpha_beta,
			).search(game.initial_state()).score
			for cached in ("a", "b")
			for alpha_beta in (False, True)
		}

		assert scores == {3.0}


class TestMinimaxTranspositionTable:
	@staticmethod
	def make_cached_search(game: TreeGame, **kwargs) -> Minimax:
		return make_search(game, canonicalizer=TreeCanonicalizer(), **kwargs)

	def test_results_match_a_search_without_a_table(
		self,
		deep_game: TreeGame,
	) -> None:
		plain = make_search(deep_game, depth=3).search(
			deep_game.initial_state()
		)
		cached = self.make_cached_search(deep_game, depth=3).search(
			deep_game.initial_state()
		)

		assert cached.score == plain.score
		assert cached.best_move == plain.best_move

	def test_every_searched_position_is_stored_including_the_root(
		self,
		game: TreeGame,
	) -> None:
		search = self.make_cached_search(game)

		statistics = search.search(game.initial_state()).statistics
		table = search.transposition_table

		assert len(table) == 7
		assert statistics.transposition_table_stores == 7
		assert statistics.transposition_table_hits == 0

		# The root is worth 3 to the player moving, and is proven.
		root = table.get(("root", 1))

		assert root.score == 3.0
		assert root.solved is True
		assert root.depth == 2

	def test_transposed_position_is_searched_once(
		self,
		transposition_game: TreeGame,
	) -> None:
		plain = make_search(transposition_game, depth=4).search(
			transposition_game.initial_state()
		)
		cached_game = TreeGame(TRANSPOSITION_TREE, TRANSPOSITION_SCORES)
		cached = self.make_cached_search(cached_game, depth=4).search(
			cached_game.initial_state()
		)

		assert plain.statistics.nodes_visited == 9
		assert cached.statistics.nodes_visited == 7
		assert cached.statistics.transposition_table_hits == 1
		assert cached.score == plain.score == 4.0
		assert cached.best_move == plain.best_move

	def test_second_search_reuses_solved_entries(self, game: TreeGame) -> None:
		search = self.make_cached_search(game)

		first = search.search(game.initial_state())
		second = search.search(game.initial_state())

		assert second.best_move == first.best_move
		assert second.score == first.score
		# Only the root is searched again; both replies are cached.
		assert second.statistics.nodes_visited == 3
		assert second.statistics.transposition_table_hits == 3
		assert second.statistics.transposition_table_stores == 0

	def test_entries_are_stored_from_the_perspective_of_the_player_to_move(
		self,
		game: TreeGame,
	) -> None:
		search = self.make_cached_search(game)

		search.search(game.initial_state())
		table = search.transposition_table

		# "a" is played by the opponent, so its value for the root player
		# (3) is stored as -3. "a1" is played by the root player.
		assert table.get(("a", 2)) == cached(
			key=("a", 2),
			depth=1,
			score=-3.0,
			bound=TranspositionBound.EXACT,
			best_move=TreeMove("a1"),
			solved=True,
		)
		assert table.get(("a1", 1)) == cached(
			key=("a1", 1),
			depth=0,
			score=3.0,
			bound=TranspositionBound.EXACT,
			best_move=None,
			solved=True,
		)

	def test_pruned_continuations_are_not_played_when_dependencies_are_not_tracked(
		self,
		game: TreeGame,
	) -> None:
		search = self.make_cached_search(game, alpha_beta=True)

		search.search(game.initial_state())

		# "b2" is pruned and nothing needs its key.
		assert game.made == ["a", "a1", "a2", "b", "b1"]

	def test_pruned_continuations_are_played_to_register_dependencies(
		self,
	) -> None:
		game = TreeGame(SHALLOW_TREE, SHALLOW_SCORES)
		search = Minimax(
			game,
			limits=SearchLimits(max_depth=2),
			alpha_beta=True,
			canonicalizer=TreeCanonicalizer(),
			transposition_table=TranspositionTable(dependency_tracking=True),
		)

		search.search(game.initial_state())

		# Playing "b2" yields its key without searching it.
		assert game.made.count("b2") == 1
		assert game.made.index("b2") > game.made.index("b1")

	@pytest.mark.parametrize("tracking", [False, True])
	def test_zero_scores_are_stored_without_a_sign(self, tracking: bool) -> None:
		game = TreeGame(SHALLOW_TREE, SHALLOW_SCORES)
		table = TranspositionTable(dependency_tracking=tracking)
		search = Minimax(
			game,
			limits=SearchLimits(max_depth=1),
			canonicalizer=TreeCanonicalizer(),
			transposition_table=table,
		)

		# Nothing is evaluated, so every score is zero, and positions the
		# opponent moves in store the negation of zero.
		search.search(game.initial_state())

		entry = table.get(("a", 2))

		assert (entry.lower, entry.upper) == (0.0, 0.0)
		assert repr(entry.lower) == "0.0"
		assert repr(entry.upper) == "0.0"

	def test_first_of_several_equal_moves_is_stored_at_a_minimizing_node(
		self,
	) -> None:
		game = TreeGame(SHALLOW_TREE, dict.fromkeys(SHALLOW_SCORES, 1.0))
		search = self.make_cached_search(game)

		search.search(game.initial_state())

		assert search.transposition_table.get(("a", 2)).best_move == (
			TreeMove("a1")
		)

	def test_first_of_several_equal_moves_is_stored_at_a_maximizing_node(
		self,
	) -> None:
		game = TreeGame(DEEP_TREE, dict.fromkeys(DEEP_SCORES, 1.0))
		search = self.make_cached_search(game, depth=3)

		search.search(game.initial_state())

		assert search.transposition_table.get(("A1", 1)).best_move == (
			TreeMove("A1a")
		)

	def test_depth_limited_positions_are_stored_as_unsolved(
		self,
		game: TreeGame,
	) -> None:
		search = self.make_cached_search(game, depth=1)

		search.search(game.initial_state())

		entry = search.transposition_table.get(("a", 2))

		assert entry.solved is False
		assert entry.depth == 0
		assert entry.is_exact is True

	def test_parent_of_a_dead_end_is_not_marked_as_solved(self) -> None:
		# "dead" is neither terminal nor expandable, so its score is only
		# an estimate. "ok" leads to a terminal position, but that cannot
		# make "p" solved while one of its children is not.
		game = TreeGame(
			{"root": ("p",), "p": ("dead", "ok"), "ok": ("ok1",)},
			{"ok1": 5.0},
		)
		search = self.make_cached_search(game, depth=4)

		search.search(game.initial_state())
		table = search.transposition_table

		assert table.get(("dead", 1)).solved is False
		assert table.get(("ok", 1)).solved is True
		assert table.get(("p", 2)).solved is False

	def test_best_moves_are_stored_in_canonical_coordinates(
		self,
		game: TreeGame,
	) -> None:
		search = make_search(game, canonicalizer=PrefixingCanonicalizer())

		search.search(game.initial_state())

		entry = search.transposition_table.get(("a", 2))

		assert entry.best_move == TreeMove("canonical:a1")

	def test_pruned_positions_are_stored_as_unsolved_bounds(
		self,
		game: TreeGame,
	) -> None:
		search = self.make_cached_search(game, alpha_beta=True)

		search.search(game.initial_state())
		table = search.transposition_table

		# "a" was searched completely.
		assert table.get(("a", 2)).solved is True
		assert table.get(("a", 2)).is_exact is True

		# "b" is worth at most 2 to the root player, which is below the
		# alpha of 3. That upper bound is stored as a lower bound of -2
		# from the opponent's perspective, and nothing is known about the
		# other side of the interval.
		pruned = table.get(("b", 2))

		assert pruned.solved is False
		assert pruned.lower == -2.0
		assert pruned.upper is None
		assert pruned.score is None
		assert pruned.best_move == TreeMove("b1")

	def test_score_equal_to_beta_is_stored_as_a_lower_bound(self) -> None:
		# Without terminal scores the third ply is evaluated. "A1" is
		# worth 5, so at "A2" beta is 5. Its first leaf is also worth 5,
		# which cuts the search off with a score equal to beta.
		game = TreeGame(DEEP_TREE, {})
		evaluator = TableEvaluator(
			{"A1a": 3.0, "A1b": 5.0, "A2a": 5.0, "A2b": 9.0}
		)
		search = self.make_cached_search(
			game,
			depth=3,
			alpha_beta=True,
			evaluator=evaluator,
		)

		search.search(game.initial_state())

		entry = search.transposition_table.get(("A2", 1))

		assert entry.lower == 5.0
		assert entry.upper is None
		assert entry.solved is False
		assert "A2b" not in [node for node, _ in evaluator.calls]

	def test_parent_of_a_node_cut_off_by_the_cache_is_not_solved(
		self,
		deep_game: TreeGame,
	) -> None:
		search = self.make_cached_search(deep_game, depth=3, alpha_beta=True)

		# At "A2", beta is 5 after "A1". A cached lower bound of 7 for the
		# maximizing player ends the search of "A2" without exploring it.
		search.transposition_table.store(
			cached(
				key=("A2", 1),
				depth=1,
				score=7.0,
				bound=TranspositionBound.LOWER,
			)
		)

		result = search.search(deep_game.initial_state())
		parent = search.transposition_table.get(("A", 2))

		assert "A2a" not in deep_game.made
		assert result.statistics.cutoffs >= 1
		assert parent.solved is False

	def test_cached_best_move_is_searched_first(self, game: TreeGame) -> None:
		search = make_search(game, canonicalizer=PrefixingCanonicalizer())

		# Too shallow to reuse the score, but its best move is a hint.
		search.transposition_table.store(
			cached(
				key=("a", 2),
				depth=0,
				score=0.0,
				bound=TranspositionBound.EXACT,
				best_move=TreeMove("canonical:a2"),
			)
		)

		search.search(game.initial_state())

		assert game.made == ["a", "a2", "a1", "b", "b1", "b2"]

	def test_cached_best_move_takes_precedence_over_move_ordering(
		self,
		game: TreeGame,
	) -> None:
		search = make_search(
			game,
			canonicalizer=TreeCanonicalizer(),
			move_ordering=ReversingMoveOrdering(),
		)

		search.transposition_table.store(
			cached(
				key=("a", 2),
				depth=0,
				score=0.0,
				bound=TranspositionBound.EXACT,
				best_move=TreeMove("a1"),
			)
		)

		search.search(game.initial_state())

		assert game.made == ["b", "b2", "b1", "a", "a1", "a2"]

	def test_illegal_cached_best_move_is_ignored(self, game: TreeGame) -> None:
		search = self.make_cached_search(game)

		search.transposition_table.store(
			cached(
				key=("a", 2),
				depth=0,
				score=0.0,
				bound=TranspositionBound.EXACT,
				best_move=TreeMove("nowhere"),
			)
		)

		result = search.search(game.initial_state())

		assert game.made == ["a", "a1", "a2", "b", "b1", "b2"]
		assert result.score == 3.0

	def test_too_shallow_entry_score_is_not_reused(
		self,
		game: TreeGame,
	) -> None:
		search = self.make_cached_search(game)

		search.transposition_table.store(
			cached(
				key=("a", 2),
				depth=0,
				score=-999.0,
				bound=TranspositionBound.EXACT,
			)
		)

		result = search.search(game.initial_state())

		assert result.score == 3.0
		assert result.statistics.transposition_table_hits == 1

	def test_invalidated_entry_score_is_not_reused(
		self,
		game: TreeGame,
	) -> None:
		search = self.make_cached_search(game)

		search.transposition_table.store(
			cached(
				key=("a", 2),
				depth=5,
				score=-999.0,
				bound=TranspositionBound.EXACT,
				score_valid=False,
			)
		)

		result = search.search(game.initial_state())

		assert result.score == 3.0

	def test_solved_entry_is_reused_at_any_depth_from_the_root_perspective(
		self,
		game: TreeGame,
	) -> None:
		search = self.make_cached_search(game)

		# Stored from the opponent's perspective: worth 42 to the root.
		search.transposition_table.store(
			cached(
				key=("a", 2),
				depth=0,
				score=-42.0,
				bound=TranspositionBound.EXACT,
				solved=True,
			)
		)

		result = search.search(game.initial_state())

		assert result.score == 42.0
		assert result.best_move == TreeMove("a")
		assert game.made == ["a", "b", "b1", "b2"]

	def test_cached_score_for_the_root_player_is_not_negated(
		self,
		deep_game: TreeGame,
	) -> None:
		search = self.make_cached_search(deep_game, depth=3)

		# "A1" is played by the root player: A = min(100, 9) = 9.
		search.transposition_table.store(
			cached(
				key=("A1", 1),
				depth=0,
				score=100.0,
				bound=TranspositionBound.EXACT,
				solved=True,
			)
		)

		result = search.search(deep_game.initial_state())

		assert result.best_move == TreeMove("A")
		assert result.score == 9.0

	def test_bounds_are_ignored_without_alpha_beta(
		self,
		game: TreeGame,
	) -> None:
		search = self.make_cached_search(game)

		search.transposition_table.store(
			cached(
				key=("b", 2),
				depth=1,
				score=-1000.0,
				bound=TranspositionBound.LOWER,
			)
		)

		result = search.search(game.initial_state())

		assert result.best_move == TreeMove("a")
		assert result.score == 3.0
		assert game.made == ["a", "a1", "a2", "b", "b1", "b2"]

	def test_cached_bound_can_cut_off_a_search_with_alpha_beta(
		self,
		game: TreeGame,
	) -> None:
		search = self.make_cached_search(game, alpha_beta=True)

		# From the opponent's view this is a lower bound of -2, i.e. the
		# root player can get at most 2 from "b". That is below the 3 already
		# guaranteed by "a", so "b" is refuted without searching it.
		search.transposition_table.store(
			cached(
				key=("b", 2),
				depth=1,
				score=-2.0,
				bound=TranspositionBound.LOWER,
			)
		)

		result = search.search(game.initial_state())

		assert result.best_move == TreeMove("a")
		assert result.score == 3.0
		assert game.made == ["a", "a1", "a2", "b"]
		assert result.statistics.cutoffs == 1
		assert result.statistics.nodes_visited == 5

	def test_cached_bound_equal_to_alpha_cuts_off_the_search(
		self,
		game: TreeGame,
	) -> None:
		search = self.make_cached_search(game, alpha_beta=True)

		# At most 3 for the root player, exactly what "a" already secured.
		search.transposition_table.store(
			cached(
				key=("b", 2),
				depth=1,
				score=-3.0,
				bound=TranspositionBound.LOWER,
			)
		)

		result = search.search(game.initial_state())

		assert result.best_move == TreeMove("a")
		assert result.score == 3.0
		assert game.made == ["a", "a1", "a2", "b"]
		assert result.statistics.cutoffs == 1

	@pytest.mark.parametrize("alpha_beta", [False, True])
	def test_table_does_not_change_results_across_depths(
		self,
		deep_game: TreeGame,
		alpha_beta: bool,
	) -> None:
		search = self.make_cached_search(
			deep_game,
			depth=1,
			alpha_beta=alpha_beta,
		)

		for depth in (1, 2, 3):
			search.limits = SearchLimits(max_depth=depth)
			reference = make_search(
				TreeGame(DEEP_TREE, DEEP_SCORES),
				depth=depth,
			)

			cached_result = search.search(deep_game.initial_state())
			reference_result = reference.search(deep_game.initial_state())

			assert cached_result.score == reference_result.score
			assert cached_result.best_move == reference_result.best_move


# A pruned continuation whose value is only discovered by a later search.
#
# From "root", the root player first finds "P" worth 3. In "A" the reply "a1"
# is worth only 1, which refutes "A" at once, so "a2" is never searched and
# "A" is known to be worth at most 1. Starting from "B" reveals "a2" (worth 7),
# which proves that "A" is worth exactly min(1, 7) = 1 after all.
DISCOVERY_TREE = {
	"root": ("P", "A"),
	"P": ("p1",),
	"A": ("a1", "a2"),
	"B": ("a2",),
	"G": ("root",),
}

DISCOVERY_SCORES = {"p1": 3.0, "a1": 1.0, "a2": 7.0}

ROOT = ("root", 1)
OPPONENT_A = ("A", 2)


def discovery_search(**kwargs) -> tuple[TreeGame, TranspositionTable, Minimax]:
	game = TreeGame(DISCOVERY_TREE, DISCOVERY_SCORES)
	table = TranspositionTable(dependency_tracking=True)
	search = Minimax(
		game,
		limits=SearchLimits(max_depth=None),
		alpha_beta=True,
		canonicalizer=TreeCanonicalizer(),
		transposition_table=table,
		**kwargs,
	)

	return game, table, search


def snapshot(table: TranspositionTable) -> dict:
	"""Everything the table knows, ignoring move-ordering hints."""
	return {
		key: (entry.lower, entry.upper, entry.solved, entry.score_valid)
		for key, entry in table._entries.items()
	}


class TestMinimaxInvalidations:
	"""Invalidations only happen for dependencies without a known relation.

	Minimax registers every dependency together with its relation, so the
	positions it affects are recomputed (refreshed) instead of invalidated.
	Invalidations therefore stay at zero unless the table also holds
	dependencies that were registered without a relation.
	"""

	def test_minimax_refreshes_instead_of_invalidating(self) -> None:
		_, _, search = discovery_search()
		search.search(TreeState("root", 1))

		statistics = search.search(TreeState("B", 2)).statistics

		assert statistics.transposition_table_refreshes > 0
		assert statistics.transposition_table_invalidations == 0

	def test_a_dependency_without_a_relation_is_counted_as_an_invalidation(
		self,
		game: TreeGame,
	) -> None:
		table = TranspositionTable(dependency_tracking=True)
		stale_key = ("legacy", 9)

		# Registered like this, nothing says how "a1" contributes to it.
		table.store(cached(stale_key, 3, 1.0, TranspositionBound.EXACT))
		table.replace_dependencies(stale_key, [("a1", 1)])

		search = make_search(
			game,
			canonicalizer=TreeCanonicalizer(),
			transposition_table=table,
		)

		statistics = search.search(game.initial_state()).statistics

		assert statistics.transposition_table_invalidations == 1
		assert statistics.transposition_table_refreshes == 0
		assert table.get(stale_key).score_valid is False

	def test_an_invalidated_entry_is_counted_once(self, game: TreeGame) -> None:
		table = TranspositionTable(dependency_tracking=True)
		stale_key = ("legacy", 9)
		table.store(cached(stale_key, 3, 1.0, TranspositionBound.EXACT))
		table.replace_dependencies(stale_key, [("a1", 1), ("a2", 1)])

		search = make_search(
			game,
			canonicalizer=TreeCanonicalizer(),
			transposition_table=table,
		)

		# Both "a1" and "a2" are new, but the entry is already stale after
		# the first of them.
		statistics = search.search(game.initial_state()).statistics

		assert statistics.transposition_table_invalidations == 1


class TestMinimaxDependencyTracking:
	def test_a_pruned_position_is_only_bounded_at_first(self) -> None:
		_, table, search = discovery_search()

		search.search(TreeState("root", 1))

		pruned = table.get(OPPONENT_A)

		# "A" is worth at most 1 to the root player. Seen by the opponent
		# who moves there that is a lower bound of -1, and the upper
		# bound is unknown.
		assert (pruned.lower, pruned.upper) == (-1.0, None)
		assert pruned.solved is False
		assert table.get(("a2", 1)) is None

	def test_later_information_improves_a_position_searched_earlier(
		self,
	) -> None:
		_, table, search = discovery_search()
		search.search(TreeState("root", 1))

		result = search.search(TreeState("B", 2))

		improved = table.get(OPPONENT_A)

		assert (improved.lower, improved.upper) == (-1.0, -1.0)
		assert improved.solved is True
		assert result.statistics.transposition_table_refreshes == 2
		assert result.statistics.transposition_table_invalidations == 0

	def test_the_improvement_reaches_the_positions_above_it(self) -> None:
		_, table, search = discovery_search()
		search.search(TreeState("G", 2))

		assert table.get(ROOT).solved is False
		assert table.get(("G", 2)).solved is False

		result = search.search(TreeState("B", 2))

		# "A", then "root", then "G" were improved, one level at a time.
		assert result.statistics.transposition_table_refreshes == 3
		assert table.get(OPPONENT_A).solved is True
		assert table.get(ROOT).solved is True
		assert table.get(ROOT).score == 3.0
		assert table.get(("G", 2)).solved is True
		assert table.get(("G", 2)).score == -3.0

	def test_the_position_is_improved_without_being_searched_again(self) -> None:
		_, table, search = discovery_search()
		search.search(TreeState("root", 1))

		result = search.search(TreeState("B", 2))

		# Only "B" and its reply were searched. "A" was not revisited,
		# yet it is now proven.
		assert result.statistics.nodes_visited == 2
		assert table.get(OPPONENT_A).solved is True

	def test_without_dependency_tracking_the_position_stays_bounded(
		self,
	) -> None:
		game = TreeGame(DISCOVERY_TREE, DISCOVERY_SCORES)
		table = TranspositionTable(dependency_tracking=False)
		search = Minimax(
			game,
			limits=SearchLimits(max_depth=None),
			alpha_beta=True,
			canonicalizer=TreeCanonicalizer(),
			transposition_table=table,
		)
		search.search(TreeState("root", 1))

		search.search(TreeState("B", 2))

		assert table.get(OPPONENT_A).solved is False
		assert table.get(OPPONENT_A).upper is None

	def test_the_final_value_is_the_true_value_after_the_update(self) -> None:
		game, _, search = discovery_search()
		search.search(TreeState("root", 1))
		search.search(TreeState("B", 2))
		reference = Minimax(
			TreeGame(DISCOVERY_TREE, DISCOVERY_SCORES),
			limits=SearchLimits(max_depth=None),
		)

		assert search.search(TreeState("root", 1)).score == (
			reference.search(TreeState("root", 1)).score
		)

	def test_repeating_a_search_does_not_update_anything(self) -> None:
		_, _, search = discovery_search()
		search.search(TreeState("root", 1))
		search.search(TreeState("B", 2))

		statistics = search.search(TreeState("root", 1)).statistics

		assert statistics.transposition_table_refreshes == 0
		assert statistics.transposition_table_invalidations == 0
		assert statistics.transposition_table_stores == 0

	def test_an_unrelated_search_does_not_update_anything(self) -> None:
		tree = {**DISCOVERY_TREE, "X": ("x1",)}
		game = TreeGame(tree, {**DISCOVERY_SCORES, "x1": 2.0})
		table = TranspositionTable(dependency_tracking=True)
		search = Minimax(
			game,
			limits=SearchLimits(max_depth=None),
			alpha_beta=True,
			canonicalizer=TreeCanonicalizer(),
			transposition_table=table,
		)
		search.search(TreeState("root", 1))
		before = snapshot(table)

		statistics = search.search(TreeState("X", 2)).statistics

		after = snapshot(table)

		assert statistics.transposition_table_refreshes == 0

		for key, information in before.items():
			assert after[key] == information

	def test_a_deeper_search_improves_the_positions_above(self) -> None:
		game = TreeGame(DEEP_TREE, DEEP_SCORES)
		table = TranspositionTable(dependency_tracking=True)
		search = Minimax(
			game,
			limits=SearchLimits(max_depth=2),
			canonicalizer=TreeCanonicalizer(),
			transposition_table=table,
		)

		search.search(game.initial_state())

		# Without terminal scores in sight, "A" is worth 0 at depth 1.
		assert table.get(("A", 2)).score == 0.0
		assert table.get(("A", 2)).solved is False

		search.limits = SearchLimits(max_depth=3)
		result = search.search(game.initial_state())

		# Searching "A1" and "A2" completely proves that "A" is worth
		# min(max(3, 5), max(6, 9)) = 5, which is -5 for the opponent.
		assert result.score == 5.0
		assert table.get(("A", 2)).score == -5.0
		assert table.get(("A", 2)).solved is True
		assert table.get(("A", 2)).depth == 2
		assert result.statistics.transposition_table_refreshes > 0
		assert result.statistics.transposition_table_invalidations == 0

	def test_a_shallow_search_keeps_the_dependencies_of_a_deeper_result(
		self,
	) -> None:
		# "other" reaches "b" too, so "b" can be met again as a leaf.
		game = TreeGame({**SHALLOW_TREE, "other": ("b",)}, SHALLOW_SCORES)
		table = TranspositionTable(dependency_tracking=True)

		def make(depth: int) -> Minimax:
			return Minimax(
				game,
				limits=SearchLimits(max_depth=depth),
				alpha_beta=True,
				canonicalizer=TreeCanonicalizer(),
				transposition_table=table,
			)

		# "b2" is pruned, so "b" is only bounded: worth at most 2.
		make(2).search(TreeState())

		assert table.get(("b", 2)).solved is False
		assert table.get(("b2", 1)) is None

		# Meeting "b" again as a leaf of a shallower search is rejected in
		# favor of the deeper result, which must keep its dependencies.
		result = make(1).search(TreeState("other", 1))

		assert result.statistics.leaf_nodes == 1
		assert table.get(("b", 2)).depth == 1

		# Discovering "b2" proves "b", through the preserved dependency.
		table.store(cached(("b2", 1), 0, 9.0, TranspositionBound.EXACT, solved=True))

		assert table.get(("b", 2)).solved is True
		assert table.get(("b", 2)).score == -2.0

	def test_a_node_that_becomes_a_leaf_forgets_its_old_dependencies(
		self,
		deep_game: TreeGame,
	) -> None:
		table = TranspositionTable(dependency_tracking=True)
		search = Minimax(
			deep_game,
			limits=SearchLimits(max_depth=2),
			canonicalizer=TreeCanonicalizer(),
			transposition_table=table,
		)
		leaf_key = ("A1", 1)
		earlier_child_key = ("elsewhere", 1)

		table.replace_dependencies(leaf_key, {earlier_child_key: 1})

		# At this depth "A1" is evaluated rather than expanded.
		search.search(deep_game.initial_state())

		outcome = table.store(
			cached(earlier_child_key, 1, 0.0, TranspositionBound.EXACT)
		)

		assert outcome.invalidated == 0
		assert table.get(leaf_key).score_valid is True


class TestMinimaxSearchOrder:
	STARTS = [
		TreeState("G", 2),
		TreeState("root", 1),
		TreeState("B", 2),
		TreeState("A", 2),
	]

	@staticmethod
	def reference_scores() -> dict:
		reference = Minimax(
			TreeGame(DISCOVERY_TREE, DISCOVERY_SCORES),
			limits=SearchLimits(max_depth=None),
		)

		return {
			(state.node, state.current_player): reference.search(
				TreeState(state.node, state.current_player)
			).score
			for state in TestMinimaxSearchOrder.STARTS
		}

	def run(self, order, **kwargs) -> tuple[dict, dict]:
		game = TreeGame(DISCOVERY_TREE, DISCOVERY_SCORES)
		table = TranspositionTable(dependency_tracking=True)
		search = Minimax(
			game,
			limits=SearchLimits(max_depth=None),
			canonicalizer=TreeCanonicalizer(),
			transposition_table=table,
			**kwargs,
		)

		scores = {}

		for state in order:
			key = (state.node, state.current_player)
			scores[key] = search.search(
				TreeState(state.node, state.current_player)
			).score

		return snapshot(table), scores

	@pytest.mark.parametrize("alpha_beta", [False, True])
	def test_every_search_order_leaves_the_same_knowledge(
		self,
		alpha_beta: bool,
	) -> None:
		results = [
			self.run(order, alpha_beta=alpha_beta)
			for order in permutations(self.STARTS)
		]

		assert len(results) == 24

		first_snapshot = results[0][0]

		assert all(snapshot_ == first_snapshot for snapshot_, _ in results)

	@pytest.mark.parametrize("alpha_beta", [False, True])
	def test_every_search_returns_the_true_value_in_any_order(
		self,
		alpha_beta: bool,
	) -> None:
		expected = self.reference_scores()

		for order in permutations(self.STARTS):
			_, scores = self.run(order, alpha_beta=alpha_beta)

			assert scores == expected

	def test_the_final_knowledge_is_proven_for_every_position(self) -> None:
		snapshot_, _ = self.run(self.STARTS, alpha_beta=True)

		for key in (("A", 2), ("root", 1), ("G", 2)):
			assert snapshot_[key][2] is True
