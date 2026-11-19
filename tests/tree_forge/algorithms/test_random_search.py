from unittest.mock import patch

import pytest

from tree_forge.algorithms.random_search import RandomSearch
from tree_forge.core.models import (
	SearchResult,
	SearchStatistics,
	SearchTerminationReason,
)

from .conftest import (
	SHALLOW_SCORES,
	SHALLOW_TREE,
	TreeGame,
	TreeMove,
	TreeState,
)


class TestRandomSearchInit:
	def test_stores_game(self, game: TreeGame) -> None:
		assert RandomSearch(game).game is game


class TestRandomSearchSearch:
	@pytest.fixture
	def search(self, game: TreeGame) -> RandomSearch:
		return RandomSearch(game)

	def test_returns_search_result(
		self,
		search: RandomSearch,
		game: TreeGame,
	) -> None:
		result = search.search(game.initial_state())

		assert isinstance(result, SearchResult)

	def test_best_move_is_a_legal_move(
		self,
		search: RandomSearch,
		game: TreeGame,
	) -> None:
		state = game.initial_state()

		for _ in range(25):
			assert search.search(state).best_move in game.legal_moves(state)

	def test_every_legal_move_can_be_selected(
		self,
		search: RandomSearch,
		game: TreeGame,
	) -> None:
		state = game.initial_state()

		selected = {search.search(state).best_move for _ in range(200)}

		assert selected == set(game.legal_moves(state))

	def test_selects_the_move_chosen_by_random_choice(
		self,
		search: RandomSearch,
		game: TreeGame,
	) -> None:
		state = game.initial_state()
		chosen = TreeMove("b")

		with patch(
			"tree_forge.algorithms.random_search.choice",
			return_value=chosen,
		) as mock_choice:
			result = search.search(state)

		assert result.best_move is chosen
		mock_choice.assert_called_once_with(game.legal_moves(state))

	def test_single_legal_move_is_always_selected(self) -> None:
		game = TreeGame({"root": ("only",)}, {"only": 1.0})

		result = RandomSearch(game).search(game.initial_state())

		assert result.best_move == TreeMove("only")

	def test_has_no_score_because_positions_are_not_evaluated(
		self,
		search: RandomSearch,
		game: TreeGame,
	) -> None:
		assert search.search(game.initial_state()).score is None

	def test_reports_a_completed_search(
		self,
		search: RandomSearch,
		game: TreeGame,
	) -> None:
		result = search.search(game.initial_state())

		assert result.completed is True
		assert result.termination_reason is SearchTerminationReason.COMPLETED

	def test_statistics_record_a_single_visited_node(
		self,
		search: RandomSearch,
		game: TreeGame,
	) -> None:
		statistics = search.search(game.initial_state()).statistics

		assert statistics == SearchStatistics(nodes_visited=1)

	@pytest.mark.parametrize("name", [
		"leaf_nodes",
		"terminal_nodes",
		"cutoffs",
		"transposition_table_hits",
		"transposition_table_stores",
		"transposition_table_invalidations",
		"transposition_table_refreshes",
		"iterations_completed",
		"elapsed_time_seconds",
		"completed_depth",
		"max_ply_reached",
	])
	def test_statistics_that_are_not_measured_are_none(
		self,
		search: RandomSearch,
		game: TreeGame,
		name: str,
	) -> None:
		statistics = search.search(game.initial_state()).statistics

		assert getattr(statistics, name) is None

	def test_each_search_gets_fresh_statistics(
		self,
		search: RandomSearch,
		game: TreeGame,
	) -> None:
		state = game.initial_state()

		first = search.search(state)
		second = search.search(state)

		assert first.statistics is not second.statistics
		assert second.statistics.nodes_visited == 1

	def test_does_not_modify_the_state(
		self,
		search: RandomSearch,
		game: TreeGame,
	) -> None:
		state = game.initial_state()

		search.search(state)

		assert state == TreeState()
		assert game.made == []
		assert game.unmade == []

	def test_searches_from_the_supplied_position(
		self,
		search: RandomSearch,
	) -> None:
		state = TreeState(node="a", current_player=2)

		for _ in range(25):
			assert search.search(state).best_move in {
				TreeMove("a1"),
				TreeMove("a2"),
			}


class TestRandomSearchWithoutLegalMoves:
	@pytest.mark.parametrize("tree, scores", [
		pytest.param({}, {}, id="dead-end"),
		pytest.param({}, {"root": 5.0}, id="terminal-position"),
	])
	def test_returns_no_move(self, tree: dict, scores: dict) -> None:
		game = TreeGame(tree, scores)

		result = RandomSearch(game).search(game.initial_state())

		assert result.best_move is None
		assert result.score is None
		assert result.completed is True
		assert result.termination_reason is SearchTerminationReason.COMPLETED

	def test_still_records_the_inspected_position(self) -> None:
		game = TreeGame({}, {})

		result = RandomSearch(game).search(game.initial_state())

		assert result.statistics.nodes_visited == 1

	def test_does_not_consult_random_choice(self) -> None:
		game = TreeGame(SHALLOW_TREE, SHALLOW_SCORES)
		state = TreeState(node="a1")

		with patch("tree_forge.algorithms.random_search.choice") as mock_choice:
			RandomSearch(game).search(state)

		mock_choice.assert_not_called()
