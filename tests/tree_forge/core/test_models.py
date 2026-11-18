from dataclasses import FrozenInstanceError
from threading import Event

import pytest

from tree_forge.core.models import (
	CanonicalPosition,
	SearchAlgorithm,
	SearchConfig,
	SearchLimits,
	SearchResult,
	SearchStatistics,
	SearchTerminationReason,
)


def parse_report(text: str) -> dict[str, str]:
	"""Parse "Label: value" report lines into a dictionary."""
	report: dict[str, str] = {}

	for line in text.splitlines():
		label, separator, value = line.partition(":")

		if separator:
			report[label.strip()] = value.strip()

	return report


class TestCanonicalPosition:
	@staticmethod
	def make_position(key="key") -> CanonicalPosition[int]:
		return CanonicalPosition[int](
			key=key,
			to_canonical_move=lambda move: move + 1,
			from_canonical_move=lambda move: move - 1,
		)

	def test_stores_key(self) -> None:
		assert self.make_position("canonical-key").key == "canonical-key"

	def test_applies_to_canonical_move_transformation(self) -> None:
		assert self.make_position().to_canonical_move(5) == 6

	def test_applies_from_canonical_move_transformation(self) -> None:
		assert self.make_position().from_canonical_move(6) == 5

	def test_transformations_round_trip_moves(self) -> None:
		position = self.make_position()

		for move in (-3, 0, 7):
			canonical_move = position.to_canonical_move(move)

			assert position.from_canonical_move(canonical_move) == move

	@pytest.mark.parametrize("field_name", [
		"key",
		"to_canonical_move",
		"from_canonical_move",
	])
	def test_is_immutable(self, field_name: str) -> None:
		position = self.make_position()

		with pytest.raises(FrozenInstanceError):
			setattr(position, field_name, None)

	def test_positions_with_same_key_and_transformations_are_equal(
		self,
	) -> None:
		def identity(move: int) -> int:
			return move

		first = CanonicalPosition[int]("key", identity, identity)
		second = CanonicalPosition[int]("key", identity, identity)

		assert first == second
		assert hash(first) == hash(second)

	def test_positions_with_different_keys_are_not_equal(self) -> None:
		def identity(move: int) -> int:
			return move

		first = CanonicalPosition[int]("first", identity, identity)
		second = CanonicalPosition[int]("second", identity, identity)

		assert first != second


class TestSearchAlgorithm:
	def test_defines_expected_algorithms(self) -> None:
		assert {algorithm.name for algorithm in SearchAlgorithm} == {
			"MINIMAX",
			"NEGAMAX",
			"BEAM",
		}

	def test_algorithms_have_distinct_values(self) -> None:
		values = [algorithm.value for algorithm in SearchAlgorithm]

		assert len(values) == len(set(values))


class TestSearchTerminationReason:
	def test_defines_expected_reasons(self) -> None:
		assert {reason.name for reason in SearchTerminationReason} == {
			"COMPLETED",
			"TIME_LIMIT",
			"NODE_LIMIT",
			"CANCELLED",
		}

	def test_reasons_have_distinct_values(self) -> None:
		values = [reason.value for reason in SearchTerminationReason]

		assert len(values) == len(set(values))


class TestSearchLimits:
	def test_default_values(self) -> None:
		limits = SearchLimits()

		assert limits.max_depth == 4
		assert limits.time_limit_seconds is None
		assert limits.node_limit is None
		assert limits.cancellation_event is None

	def test_custom_values(self) -> None:
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

	def test_every_limit_can_be_disabled(self) -> None:
		limits = SearchLimits(
			max_depth=None,
			time_limit_seconds=None,
			node_limit=None,
		)

		assert limits.max_depth is None
		assert limits.time_limit_seconds is None
		assert limits.node_limit is None

	@pytest.mark.parametrize("keyword, value", [
		("max_depth", 1),
		("time_limit_seconds", 0.001),
		("node_limit", 1),
	])
	def test_accepts_smallest_valid_value(
		self,
		keyword: str,
		value: float,
	) -> None:
		limits = SearchLimits(**{keyword: value})

		assert getattr(limits, keyword) == value

	@pytest.mark.parametrize("keyword, value, message", [
		("max_depth", 0, "max_depth must be at least 1 or None"),
		("max_depth", -1, "max_depth must be at least 1 or None"),
		(
			"time_limit_seconds",
			0,
			"time_limit_seconds must be greater than 0 or None",
		),
		(
			"time_limit_seconds",
			-1.5,
			"time_limit_seconds must be greater than 0 or None",
		),
		("node_limit", 0, "node_limit must be at least 1 or None"),
		("node_limit", -10, "node_limit must be at least 1 or None"),
	])
	def test_rejects_invalid_values(
		self,
		keyword: str,
		value: float,
		message: str,
	) -> None:
		with pytest.raises(ValueError, match=message):
			SearchLimits(**{keyword: value})

	def test_is_immutable(self) -> None:
		limits = SearchLimits()

		with pytest.raises(FrozenInstanceError):
			limits.max_depth = 10

	def test_cancellation_event_does_not_affect_equality(self) -> None:
		first = SearchLimits(cancellation_event=Event())
		second = SearchLimits(cancellation_event=Event())

		assert first == second
		assert hash(first) == hash(second)

	def test_other_fields_affect_equality(self) -> None:
		assert SearchLimits(max_depth=3) != SearchLimits(max_depth=4)
		assert SearchLimits(node_limit=5) != SearchLimits(node_limit=6)

	def test_repr_omits_cancellation_event(self) -> None:
		limits = SearchLimits(max_depth=6, cancellation_event=Event())

		assert "cancellation_event" not in repr(limits)
		assert "max_depth=6" in repr(limits)


class TestSearchConfig:
	def test_default_values(self) -> None:
		config = SearchConfig()

		assert config.algorithm is SearchAlgorithm.NEGAMAX
		assert config.limits == SearchLimits()
		assert config.alpha_beta is True
		assert config.iterative_deepening is True
		assert config.beam_width == 8
		assert config.use_transposition_table is True
		assert config.use_canonicalization is False

	def test_custom_values(self) -> None:
		limits = SearchLimits(max_depth=10)

		config = SearchConfig(
			algorithm=SearchAlgorithm.BEAM,
			limits=limits,
			alpha_beta=False,
			iterative_deepening=False,
			beam_width=4,
			use_transposition_table=False,
			use_canonicalization=True,
		)

		assert config.algorithm is SearchAlgorithm.BEAM
		assert config.limits is limits
		assert config.alpha_beta is False
		assert config.iterative_deepening is False
		assert config.beam_width == 4
		assert config.use_transposition_table is False
		assert config.use_canonicalization is True

	def test_default_limits_are_created_per_instance(self) -> None:
		first = SearchConfig()
		second = SearchConfig()

		assert first.limits == second.limits
		assert first.limits is not second.limits

	def test_accepts_smallest_beam_width(self) -> None:
		assert SearchConfig(beam_width=1).beam_width == 1

	@pytest.mark.parametrize("beam_width", [0, -1, -100])
	def test_rejects_beam_width_below_one(self, beam_width: int) -> None:
		with pytest.raises(ValueError, match="beam_width must be at least 1"):
			SearchConfig(beam_width=beam_width)

	def test_is_immutable(self) -> None:
		config = SearchConfig()

		with pytest.raises(FrozenInstanceError):
			config.beam_width = 2


class TestSearchStatistics:
	def test_default_values(self) -> None:
		statistics = SearchStatistics()

		assert statistics.nodes_visited == 0
		assert statistics.leaf_nodes == 0
		assert statistics.terminal_nodes == 0
		assert statistics.cutoffs == 0
		assert statistics.transposition_table_hits == 0
		assert statistics.transposition_table_stores == 0
		assert statistics.transposition_table_invalidations == 0
		assert statistics.iterations_completed == 0
		assert statistics.elapsed_time_seconds == 0.0
		assert statistics.completed_depth == 0
		assert statistics.max_ply_reached == 0

	def test_accepts_initial_values(self) -> None:
		statistics = SearchStatistics(nodes_visited=12, cutoffs=3)

		assert statistics.nodes_visited == 12
		assert statistics.cutoffs == 3
		assert statistics.leaf_nodes == 0

	def test_counters_can_be_updated(self) -> None:
		statistics = SearchStatistics()

		statistics.nodes_visited += 5
		statistics.nodes_visited += 2
		statistics.transposition_table_invalidations += 1
		statistics.max_ply_reached = 7

		assert statistics.nodes_visited == 7
		assert statistics.transposition_table_invalidations == 1
		assert statistics.max_ply_reached == 7

	def test_instances_do_not_share_state(self) -> None:
		first = SearchStatistics()
		second = SearchStatistics()

		first.nodes_visited = 10

		assert second.nodes_visited == 0

	def test_instances_with_equal_values_are_equal(self) -> None:
		assert SearchStatistics(nodes_visited=3) == SearchStatistics(
			nodes_visited=3
		)
		assert SearchStatistics(nodes_visited=3) != SearchStatistics(
			nodes_visited=4
		)


class TestSearchStatisticsString:
	@pytest.fixture
	def populated(self) -> SearchStatistics:
		return SearchStatistics(
			nodes_visited=1_234_567,
			leaf_nodes=890,
			terminal_nodes=12,
			cutoffs=3_456,
			transposition_table_hits=78,
			transposition_table_stores=9_000,
			transposition_table_invalidations=4,
			iterations_completed=5,
			elapsed_time_seconds=0.25,
			completed_depth=6,
			max_ply_reached=8,
		)

	def test_is_framed_by_title_and_rules(
		self,
		populated: SearchStatistics,
	) -> None:
		lines = str(populated).splitlines()

		assert lines[0] == "SEARCH STATISTICS"
		assert lines[1] == "=" * 35
		assert lines[-1] == "=" * 35

	def test_reports_every_statistic_once(
		self,
		populated: SearchStatistics,
	) -> None:
		lines = str(populated).splitlines()

		# Title, two rules, and eleven statistics.
		assert len(lines) == 14
		assert set(parse_report(str(populated))) == {
			"Nodes visited",
			"Leaf nodes",
			"Terminal nodes",
			"Alpha-beta cutoffs",
			"Transposition hits",
			"Transposition stores",
			"Transposition invalidations",
			"Search time",
			"Completed depth",
			"Iterations completed",
			"Maximum ply reached",
		}

	def test_reports_values(self, populated: SearchStatistics) -> None:
		report = parse_report(str(populated))

		assert report["Leaf nodes"] == "890"
		assert report["Terminal nodes"] == "12"
		assert report["Transposition invalidations"] == "4"
		assert report["Completed depth"] == "6"
		assert report["Iterations completed"] == "5"
		assert report["Maximum ply reached"] == "8"

	def test_groups_large_counters_with_thousands_separators(
		self,
		populated: SearchStatistics,
	) -> None:
		report = parse_report(str(populated))

		assert report["Nodes visited"] == "1,234,567"
		assert report["Alpha-beta cutoffs"] == "3,456"
		assert report["Transposition stores"] == "9,000"

	def test_formats_elapsed_time_with_six_decimals(
		self,
		populated: SearchStatistics,
	) -> None:
		assert parse_report(str(populated))["Search time"] == "0.250000s"

	def test_default_statistics_report_zeros(self) -> None:
		report = parse_report(str(SearchStatistics()))

		assert report["Nodes visited"] == "0"
		assert report["Search time"] == "0.000000s"


class TestSearchResult:
	def test_default_values(self) -> None:
		result = SearchResult[int](best_move=5, score=10.0)

		assert result.best_move == 5
		assert result.score == 10.0
		assert result.statistics == SearchStatistics()
		assert result.completed is True
		assert result.termination_reason is SearchTerminationReason.COMPLETED

	def test_custom_values(self) -> None:
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
		assert result.termination_reason is SearchTerminationReason.TIME_LIMIT

	def test_best_move_can_be_none(self) -> None:
		result = SearchResult[int](best_move=None, score=0.0)

		assert result.best_move is None

	def test_default_statistics_are_created_per_instance(self) -> None:
		first = SearchResult[int](best_move=1, score=1.0)
		second = SearchResult[int](best_move=2, score=2.0)

		assert first.statistics is not second.statistics

	def test_can_represent_interrupted_search(self) -> None:
		result = SearchResult[int](
			best_move=1,
			score=5.0,
			completed=False,
			termination_reason=SearchTerminationReason.CANCELLED,
		)

		assert result.completed is False
		assert result.termination_reason is SearchTerminationReason.CANCELLED

	def test_completed_depth_reads_from_statistics(self) -> None:
		result = SearchResult[int](
			best_move=2,
			score=1.0,
			statistics=SearchStatistics(completed_depth=5),
		)

		assert result.completed_depth == 5

	def test_completed_depth_tracks_later_statistics_changes(self) -> None:
		result = SearchResult[int](best_move=2, score=1.0)

		result.statistics.completed_depth = 9

		assert result.completed_depth == 9


class TestSearchResultString:
	def test_starts_with_title_and_rule(self) -> None:
		lines = str(SearchResult[int](best_move=1, score=1.0)).splitlines()

		assert lines[0] == "SEARCH RESULT"
		assert lines[1] == "=" * 35

	def test_reports_best_move_score_and_completion(self) -> None:
		report = parse_report(
			str(SearchResult[str](best_move="e2e4", score=10.0))
		)

		assert report["Best move"] == "e2e4"
		assert report["Score"] == "10.00"
		assert report["Completed"] == "True"

	def test_reports_missing_best_move(self) -> None:
		report = parse_report(
			str(SearchResult[int](best_move=None, score=0.0))
		)

		assert report["Best move"] == "None"

	@pytest.mark.parametrize("score, expected", [
		(1.0, "1.00"),
		(-3.14159, "-3.14"),
		(2.5, "2.50"),
		(1_000_000.0, "1000000.00"),
	])
	def test_formats_score_with_two_decimals(
		self,
		score: float,
		expected: str,
	) -> None:
		report = parse_report(str(SearchResult[int](best_move=1, score=score)))

		assert report["Score"] == expected

	def test_reports_incomplete_search(self) -> None:
		result = SearchResult[int](
			best_move=1,
			score=0.0,
			completed=False,
			termination_reason=SearchTerminationReason.NODE_LIMIT,
		)

		report = parse_report(str(result))

		assert report["Completed"] == "False"
		assert "Termination reason" in report

	def test_embeds_search_statistics(self) -> None:
		statistics = SearchStatistics(nodes_visited=1_500, cutoffs=7)

		result = SearchResult[int](
			best_move=1,
			score=1.0,
			statistics=statistics,
		)

		text = str(result)

		assert text.endswith(str(statistics))
		assert parse_report(text)["Nodes visited"] == "1,500"
