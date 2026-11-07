from dataclasses import FrozenInstanceError
from threading import Event

import pytest

from tree_forge.models import (
	SearchAlgorithm,
	SearchConfig,
	SearchLimits,
	SearchResult,
	SearchStatistics,
	SearchTerminationReason,
)


class TestSearchAlgorithm:
	def test_available_algorithms(self):
		assert SearchAlgorithm.MINIMAX
		assert SearchAlgorithm.NEGAMAX
		assert SearchAlgorithm.BEAM

	def test_algorithms_are_distinct(self):
		assert SearchAlgorithm.MINIMAX != SearchAlgorithm.NEGAMAX
		assert SearchAlgorithm.MINIMAX != SearchAlgorithm.BEAM
		assert SearchAlgorithm.NEGAMAX != SearchAlgorithm.BEAM


class TestSearchTerminationReason:
	def test_available_termination_reasons(self):
		assert SearchTerminationReason.COMPLETED
		assert SearchTerminationReason.TIME_LIMIT
		assert SearchTerminationReason.NODE_LIMIT
		assert SearchTerminationReason.CANCELLED


class TestSearchLimits:
	def test_default_values(self):
		limits = SearchLimits()

		assert limits.max_depth == 4
		assert limits.time_limit_seconds is None
		assert limits.node_limit is None
		assert limits.cancellation_event is None

	def test_custom_values(self):
		event = Event()

		limits = SearchLimits(
			max_depth=8,
			time_limit_seconds=5.0,
			node_limit=1000,
			cancellation_event=event,
		)

		assert limits.max_depth == 8
		assert limits.time_limit_seconds == 5.0
		assert limits.node_limit == 1000
		assert limits.cancellation_event is event

	def test_max_depth_can_be_unlimited(self):
		limits = SearchLimits(max_depth=None)

		assert limits.max_depth is None

	def test_max_depth_must_be_positive(self):
		with pytest.raises(ValueError):
			SearchLimits(max_depth=0)

		with pytest.raises(ValueError):
			SearchLimits(max_depth=-1)

	def test_time_limit_must_be_positive(self):
		with pytest.raises(ValueError):
			SearchLimits(time_limit_seconds=0)

		with pytest.raises(ValueError):
			SearchLimits(time_limit_seconds=-1.0)

	def test_node_limit_must_be_positive(self):
		with pytest.raises(ValueError):
			SearchLimits(node_limit=0)

		with pytest.raises(ValueError):
			SearchLimits(node_limit=-1)

	def test_search_limits_are_immutable(self):
		limits = SearchLimits()

		with pytest.raises(FrozenInstanceError):
			limits.max_depth = 10

	def test_cancellation_event_does_not_affect_equality(self):
		first = SearchLimits(cancellation_event=Event())
		second = SearchLimits(cancellation_event=Event())

		assert first == second

	def test_cancellation_event_is_excluded_from_repr(self):
		limits = SearchLimits(cancellation_event=Event())

		assert "cancellation_event" not in repr(limits)


class TestSearchConfig:
	def test_default_values(self):
		config = SearchConfig()

		assert config.algorithm == SearchAlgorithm.NEGAMAX
		assert config.limits == SearchLimits()
		assert config.alpha_beta is True
		assert config.iterative_deepening is True
		assert config.beam_width == 8
		assert config.use_transposition_table is True
		assert config.use_canonicalization is False

	def test_custom_values(self):
		limits = SearchLimits(max_depth=10)

		config = SearchConfig(
			algorithm=SearchAlgorithm.MINIMAX,
			limits=limits,
			alpha_beta=False,
			iterative_deepening=False,
			beam_width=4,
			use_transposition_table=False,
			use_canonicalization=True,
		)

		assert config.algorithm == SearchAlgorithm.MINIMAX
		assert config.limits is limits
		assert config.alpha_beta is False
		assert config.iterative_deepening is False
		assert config.beam_width == 4
		assert config.use_transposition_table is False
		assert config.use_canonicalization is True

	def test_beam_width_must_be_positive(self):
		with pytest.raises(ValueError):
			SearchConfig(beam_width=0)

		with pytest.raises(ValueError):
			SearchConfig(beam_width=-1)

	def test_each_config_gets_independent_default_limits(self):
		first = SearchConfig()
		second = SearchConfig()

		assert first.limits is not second.limits


class TestSearchStatistics:
	def test_default_values(self):
		statistics = SearchStatistics()

		assert statistics.nodes_visited == 0
		assert statistics.leaf_nodes == 0
		assert statistics.terminal_nodes == 0
		assert statistics.cutoffs == 0
		assert statistics.transposition_table_hits == 0
		assert statistics.transposition_table_stores == 0
		assert statistics.elapsed_time_seconds == 0.0
		assert statistics.completed_depth == 0

	def test_statistics_can_be_updated(self):
		statistics = SearchStatistics()

		statistics.nodes_visited = 100
		statistics.leaf_nodes = 40
		statistics.terminal_nodes = 5
		statistics.cutoffs = 20
		statistics.transposition_table_hits = 10
		statistics.transposition_table_stores = 15
		statistics.elapsed_time_seconds = 0.25
		statistics.completed_depth = 4

		assert statistics.nodes_visited == 100
		assert statistics.leaf_nodes == 40
		assert statistics.terminal_nodes == 5
		assert statistics.cutoffs == 20
		assert statistics.transposition_table_hits == 10
		assert statistics.transposition_table_stores == 15
		assert statistics.elapsed_time_seconds == 0.25
		assert statistics.completed_depth == 4


class TestSearchResult:
	def test_default_values(self):
		result = SearchResult[int](
			best_move=5,
			score=10.0,
		)

		assert result.best_move == 5
		assert result.score == 10.0
		assert isinstance(result.statistics, SearchStatistics)
		assert result.completed is True
		assert result.termination_reason == SearchTerminationReason.COMPLETED

	def test_custom_values(self):
		statistics = SearchStatistics(completed_depth=6)

		result = SearchResult[int](
			best_move=3,
			score=-5.0,
			statistics=statistics,
			completed=False,
			termination_reason=SearchTerminationReason.TIME_LIMIT,
		)

		assert result.best_move == 3
		assert result.score == -5.0
		assert result.statistics is statistics
		assert result.completed is False
		assert result.termination_reason == SearchTerminationReason.TIME_LIMIT

	def test_best_move_can_be_none(self):
		result = SearchResult[int](
			best_move=None,
			score=0.0,
		)

		assert result.best_move is None

	def test_completed_depth_property(self):
		statistics = SearchStatistics(completed_depth=5)

		result = SearchResult[int](
			best_move=2,
			score=1.0,
			statistics=statistics,
		)

		assert result.completed_depth == 5

	def test_statistics_are_independent_by_default(self):
		first = SearchResult[int](best_move=1, score=1.0)
		second = SearchResult[int](best_move=2, score=2.0)

		assert first.statistics is not second.statistics

	def test_result_can_represent_interrupted_search(self):
		result = SearchResult[int](
			best_move=1,
			score=5.0,
			completed=False,
			termination_reason=SearchTerminationReason.CANCELLED,
		)

		assert result.completed is False
		assert result.termination_reason == SearchTerminationReason.CANCELLED

