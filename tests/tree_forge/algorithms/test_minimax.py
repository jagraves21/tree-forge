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
		assert statistics.cutoffs == 0

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

		# The root is expanded directly by search() and is not counted.
		assert statistics.nodes_visited == 6
		assert statistics.leaf_nodes == 4
		assert statistics.terminal_nodes == 4
		assert statistics.cutoffs == 0
		assert statistics.max_ply_reached == 2
		assert statistics.completed_depth == 2

	def test_unused_cache_counters_stay_zero(self, game: TreeGame) -> None:
		statistics = make_search(game).search(game.initial_state()).statistics

		assert statistics.transposition_table_hits == 0
		assert statistics.transposition_table_stores == 0
		assert statistics.transposition_table_invalidations == 0

	def test_alpha_beta_prunes_the_second_reply_of_a_refuted_move(
		self,
		game: TreeGame,
	) -> None:
		# After "a" scores 3, "b1" scores 2 and refutes "b", so "b2" is
		# never searched.
		statistics = make_search(game, alpha_beta=True).search(
			game.initial_state()
		).statistics

		assert statistics.nodes_visited == 5
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

		assert statistics.nodes_visited == 5
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

		assert statistics.nodes_visited == 10
		assert statistics.leaf_nodes == 5
		assert statistics.cutoffs == 2

	def test_plain_search_of_the_deeper_tree_visits_every_node(
		self,
		deep_game: TreeGame,
	) -> None:
		statistics = make_search(deep_game, depth=3).search(
			deep_game.initial_state()
		).statistics

		assert statistics.nodes_visited == 14
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

		assert statistics.nodes_visited == 2
		assert statistics.leaf_nodes == 2
		assert statistics.terminal_nodes == 0
		assert statistics.max_ply_reached == 1

	def test_completed_depth_is_the_configured_depth(
		self,
		game: TreeGame,
	) -> None:
		statistics = make_search(game, depth=10).search(
			game.initial_state()
		).statistics

		assert statistics.completed_depth == 10

	def test_completed_depth_is_unset_without_a_depth_limit(
		self,
		game: TreeGame,
	) -> None:
		statistics = make_search(game, depth=None).search(
			game.initial_state()
		).statistics

		assert statistics.completed_depth == 0

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
		assert plain.statistics.cutoffs == 0
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

	def test_every_non_root_position_is_stored(self, game: TreeGame) -> None:
		search = self.make_cached_search(game)

		statistics = search.search(game.initial_state()).statistics

		assert len(search.transposition_table) == 6
		assert statistics.transposition_table_stores == 6
		assert statistics.transposition_table_hits == 0
		assert search.transposition_table.get(("root", 1)) is None

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

		assert plain.statistics.nodes_visited == 8
		assert cached.statistics.nodes_visited == 6
		assert cached.statistics.transposition_table_hits == 1
		assert cached.score == plain.score == 4.0
		assert cached.best_move == plain.best_move

	def test_second_search_reuses_solved_entries(self, game: TreeGame) -> None:
		search = self.make_cached_search(game)

		first = search.search(game.initial_state())
		second = search.search(game.initial_state())

		assert second.best_move == first.best_move
		assert second.score == first.score
		assert second.statistics.nodes_visited == 2
		assert second.statistics.transposition_table_hits == 2
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
		assert table.get(("a", 2)) == TranspositionEntry(
			key=("a", 2),
			depth=1,
			score=-3.0,
			bound=TranspositionBound.EXACT,
			best_move=TreeMove("a1"),
			solved=True,
		)
		assert table.get(("a1", 1)) == TranspositionEntry(
			key=("a1", 1),
			depth=0,
			score=3.0,
			bound=TranspositionBound.EXACT,
			best_move=None,
			solved=True,
		)

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
		assert entry.bound is TranspositionBound.EXACT

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
		assert table.get(("a", 2)).bound is TranspositionBound.EXACT

		# "b" is worth at most 2 to the root player, which is below the
		# alpha of 3. That upper bound is stored as a lower bound of -2
		# from the opponent's perspective.
		pruned = table.get(("b", 2))

		assert pruned.solved is False
		assert pruned.bound is TranspositionBound.LOWER
		assert pruned.score == -2.0
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

		assert entry.score == 5.0
		assert entry.bound is TranspositionBound.LOWER
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
			TranspositionEntry(
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
			TranspositionEntry(
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
			TranspositionEntry(
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
			TranspositionEntry(
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
			TranspositionEntry(
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
			TranspositionEntry(
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
			TranspositionEntry(
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
			TranspositionEntry(
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
			TranspositionEntry(
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
			TranspositionEntry(
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
		assert result.statistics.nodes_visited == 4

	def test_cached_bound_equal_to_alpha_cuts_off_the_search(
		self,
		game: TreeGame,
	) -> None:
		search = self.make_cached_search(game, alpha_beta=True)

		# At most 3 for the root player, exactly what "a" already secured.
		search.transposition_table.store(
			TranspositionEntry(
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


class TestMinimaxDependencyInvalidation:
	@staticmethod
	def search_twice(table: TranspositionTable, game: TreeGame):
		search = Minimax(
			game,
			limits=SearchLimits(max_depth=2),
			canonicalizer=TreeCanonicalizer(),
			transposition_table=table,
		)

		search.search(game.initial_state())

		search.limits = SearchLimits(max_depth=3)

		return search.search(game.initial_state())

	def test_deeper_search_invalidates_stale_ancestors(
		self,
		deep_game: TreeGame,
	) -> None:
		table = TranspositionTable(dependency_tracking=True)

		result = self.search_twice(table, deep_game)

		# Improving "A1" and "B1" invalidates their unsolved parents once.
		assert result.statistics.transposition_table_invalidations == 2
		assert result.score == 5.0
		assert result.best_move == TreeMove("A")

	def test_ancestors_are_refreshed_by_the_deeper_search(
		self,
		deep_game: TreeGame,
	) -> None:
		table = TranspositionTable(dependency_tracking=True)

		self.search_twice(table, deep_game)

		entry = table.get(("A", 2))

		assert entry.score_valid is True
		assert entry.solved is True
		assert entry.depth == 2

	def test_no_invalidations_without_dependency_tracking(
		self,
		deep_game: TreeGame,
	) -> None:
		table = TranspositionTable(dependency_tracking=False)

		result = self.search_twice(table, deep_game)

		assert result.statistics.transposition_table_invalidations == 0
		assert result.score == 5.0

	def test_cached_bound_keeps_previously_tracked_dependencies(
		self,
		deep_game: TreeGame,
	) -> None:
		table = TranspositionTable(dependency_tracking=True)
		search = Minimax(
			deep_game,
			limits=SearchLimits(max_depth=2),
			alpha_beta=True,
			canonicalizer=TreeCanonicalizer(),
			transposition_table=table,
		)
		parent_key = ("A", 2)
		earlier_child_key = ("elsewhere", 1)

		table.store(
			TranspositionEntry(
				key=earlier_child_key,
				depth=1,
				score=0.0,
				bound=TranspositionBound.EXACT,
			)
		)
		table.replace_dependencies(parent_key, {earlier_child_key})

		# Seen from the root this is an upper bound of 100 for "A". It
		# tightens the window without causing a cutoff, so the search
		# of "A" goes ahead and must add to the existing dependencies.
		table.store(
			TranspositionEntry(
				key=parent_key,
				depth=1,
				score=-100.0,
				bound=TranspositionBound.LOWER,
			)
		)

		search.search(deep_game.initial_state())

		improved = TranspositionEntry(
			key=earlier_child_key,
			depth=2,
			score=0.0,
			bound=TranspositionBound.EXACT,
		)

		assert table.get(parent_key).score_valid is True
		assert table.store(improved) == 1
		assert table.get(parent_key).score_valid is False

	def test_dependencies_are_replaced_when_no_cached_bound_is_used(
		self,
		deep_game: TreeGame,
	) -> None:
		table = TranspositionTable(dependency_tracking=True)
		search = Minimax(
			deep_game,
			limits=SearchLimits(max_depth=2),
			alpha_beta=True,
			canonicalizer=TreeCanonicalizer(),
			transposition_table=table,
		)
		parent_key = ("A", 2)
		earlier_child_key = ("elsewhere", 1)

		table.store(
			TranspositionEntry(
				key=earlier_child_key,
				depth=1,
				score=0.0,
				bound=TranspositionBound.EXACT,
			)
		)
		table.replace_dependencies(parent_key, {earlier_child_key})

		search.search(deep_game.initial_state())

		improved = TranspositionEntry(
			key=earlier_child_key,
			depth=2,
			score=0.0,
			bound=TranspositionBound.EXACT,
		)

		# "A" was searched from scratch, so it no longer depends on the
		# child it was previously registered with.
		assert table.get(parent_key).score_valid is True
		assert table.store(improved) == 0
		assert table.get(parent_key).score_valid is True

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

		table.store(
			TranspositionEntry(
				key=earlier_child_key,
				depth=1,
				score=0.0,
				bound=TranspositionBound.EXACT,
			)
		)
		table.replace_dependencies(leaf_key, {earlier_child_key})

		# At this depth "A1" is evaluated rather than expanded.
		search.search(deep_game.initial_state())

		improved = TranspositionEntry(
			key=earlier_child_key,
			depth=2,
			score=0.0,
			bound=TranspositionBound.EXACT,
		)

		assert table.store(improved) == 0
		assert table.get(leaf_key).score_valid is True
