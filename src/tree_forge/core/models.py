from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import Event
from typing import Generic, TypeVar


MoveT = TypeVar("MoveT")


@dataclass(frozen=True, slots=True)
class CanonicalPosition(Generic[MoveT]):
	"""Canonical representation of a game position.

	The key identifies positions that are equivalent under the
	game's supported symmetries.

	The move transformations convert moves between the original
	position's coordinates and the canonical position's coordinates:

	- to_canonical_move converts an original-position move to
	  canonical coordinates.
	- from_canonical_move converts a canonical move back to
	  original-position coordinates.

	When no symmetries are supported, both transformations are
	identity transformations.
	"""

	key: Hashable

	to_canonical_move: Callable[[MoveT], MoveT]
	from_canonical_move: Callable[[MoveT], MoveT]


class Outcome(Enum):
	"""Outcome of a terminal game state for a particular player.

	An outcome is always relative to the player it was requested for:
	when the opponent has won, the outcome for the player is LOSS.
	"""

	WIN = auto()
	LOSS = auto()
	DRAW = auto()


class SearchAlgorithm(Enum):
	"""Available search algorithms."""

	MINIMAX = auto()
	NEGAMAX = auto()
	BEAM = auto()


class SearchTerminationReason(Enum):
	"""Reason a search operation terminated."""

	COMPLETED = auto()
	TIME_LIMIT = auto()
	NODE_LIMIT = auto()
	CANCELLED = auto()


@dataclass(frozen=True, slots=True)
class SearchLimits:
	"""Limits controlling search execution.

	Any limit set to None is disabled.

	At least one limit should be configured to prevent an
	unbounded search.
	"""

	max_depth: int | None = 4
	time_limit_seconds: float | None = None
	node_limit: int | None = None
	cancellation_event: Event | None = field(
		default=None,
		repr=False,
		compare=False,
	)

	def __post_init__(self) -> None:
		if self.max_depth is not None and self.max_depth < 1:
			raise ValueError("max_depth must be at least 1 or None")

		if (
			self.time_limit_seconds is not None
			and self.time_limit_seconds <= 0
		):
			raise ValueError(
				"time_limit_seconds must be greater than 0 or None"
			)

		if self.node_limit is not None and self.node_limit < 1:
			raise ValueError("node_limit must be at least 1 or None")


@dataclass(frozen=True, slots=True)
class SearchConfig:
	"""Configuration shared by search algorithms."""

	algorithm: SearchAlgorithm = SearchAlgorithm.NEGAMAX
	limits: SearchLimits = field(default_factory=SearchLimits)

	alpha_beta: bool = True
	iterative_deepening: bool = True

	beam_width: int = 8

	use_transposition_table: bool = True
	use_canonicalization: bool = False

	def __post_init__(self) -> None:
		if self.beam_width < 1:
			raise ValueError("beam_width must be at least 1")


@dataclass(slots=True)
class SearchStatistics:
	"""Statistics collected during a search operation.

	Every statistic is either a measured value or None. None means the
	statistic was not measured: it is disabled, unavailable, or not
	applicable to the search that produced these statistics (for
	example, cutoffs when alpha-beta pruning is off). A value of 0 only
	ever means that the statistic was measured and its result was zero.

	nodes_visited, leaf_nodes and terminal_nodes count positions,
	including the root. cutoffs counts alpha-beta cutoffs.

	transposition_table_hits counts lookups that found an entry for
	the position. transposition_table_stores counts writes that
	changed the table; rejected writes are not counted.
	transposition_table_invalidations and
	transposition_table_refreshes count entries of previously
	searched positions that were marked stale, or recomputed in
	place, because information about their continuations changed.
	They are only measured when the table tracks dependencies.
	transposition_table_size is the number of entries in the table
	when the search returned, including entries stored by earlier
	searches that use the same table.

	completed_depth is the depth of a completed fixed-depth search
	and is None when the search had no depth limit.
	iterations_completed counts iterative-deepening passes and is
	None for algorithms that do not iterate. max_ply_reached is the
	deepest ply visited, even if the search was interrupted.

	Use merge() to aggregate the statistics of several searches.
	"""

	nodes_visited: int | None = None
	leaf_nodes: int | None = None
	terminal_nodes: int | None = None

	cutoffs: int | None = None

	transposition_table_hits: int | None = None
	transposition_table_stores: int | None = None
	transposition_table_invalidations: int | None = None
	transposition_table_refreshes: int | None = None
	transposition_table_size: int | None = None

	iterations_completed: int | None = None

	elapsed_time_seconds: float | None = None
	completed_depth: int | None = None
	max_ply_reached: int | None = None

	_SUMMED = (
		"nodes_visited",
		"leaf_nodes",
		"terminal_nodes",
		"cutoffs",
		"transposition_table_hits",
		"transposition_table_stores",
		"transposition_table_invalidations",
		"transposition_table_refreshes",
		"iterations_completed",
		"elapsed_time_seconds",
	)

	# Snapshots and maxima, which are not added up.
	_LARGEST = (
		"transposition_table_size",
		"completed_depth",
		"max_ply_reached",
	)

	def merge(self, other: SearchStatistics) -> SearchStatistics:
		"""Return the combined statistics of two searches.

		Counters and elapsed time are added. completed_depth,
		max_ply_reached and transposition_table_size keep the larger
		value, since they are not amounts that accumulate. A statistic that was
		not measured by one of the searches is taken from the other;
		it is only None when neither search measured it, so the
		result never reports a measurement that did not happen.
		"""
		combined = SearchStatistics()

		for name in self._SUMMED:
			setattr(
				combined,
				name,
				self._combine(
					getattr(self, name),
					getattr(other, name),
					lambda left, right: left + right,
				),
			)

		for name in self._LARGEST:
			setattr(
				combined,
				name,
				self._combine(getattr(self, name), getattr(other, name), max),
			)

		return combined

	@staticmethod
	def _combine(left, right, operation):
		if left is None:
			return right

		if right is None:
			return left

		return operation(left, right)

	@staticmethod
	def _format(value: int | float | None, template: str = "{:,}") -> str:
		return "n/a" if value is None else template.format(value)

	def __str__(self) -> str:
		rows = [
			("Nodes visited:", self._format(self.nodes_visited)),
			("Leaf nodes:", self._format(self.leaf_nodes)),
			("Terminal nodes:", self._format(self.terminal_nodes)),
			("Alpha-beta cutoffs:", self._format(self.cutoffs)),
			(
				"Transposition hits:",
				self._format(self.transposition_table_hits),
			),
			(
				"Transposition stores:",
				self._format(self.transposition_table_stores),
			),
			(
				"Transposition invalidations:",
				self._format(self.transposition_table_invalidations),
			),
			(
				"Transposition refreshes:",
				self._format(self.transposition_table_refreshes),
			),
			(
				"Transposition table size:",
				self._format(self.transposition_table_size),
			),
			(
				"Search time:",
				self._format(self.elapsed_time_seconds, "{:.6f}s"),
			),
			("Completed depth:", self._format(self.completed_depth, "{}")),
			(
				"Iterations completed:",
				self._format(self.iterations_completed, "{}"),
			),
			(
				"Maximum ply reached:",
				self._format(self.max_ply_reached, "{}"),
			),
		]

		return "\n".join([
			"SEARCH STATISTICS",
			"=" * 35,
			*(f"{label:<25} {value}" for label, value in rows),
			"=" * 35,
		])


@dataclass(slots=True)
class SearchResult(Generic[MoveT]):
	"""Result returned by a search operation.

	If a search is interrupted, best_move and score should represent
	the last successfully completed search iteration, when available.

	score is None when the algorithm does not evaluate positions.
	"""

	best_move: MoveT | None
	score: float | None

	statistics: SearchStatistics = field(
		default_factory=SearchStatistics
	)

	completed: bool = True

	termination_reason: SearchTerminationReason = (
		SearchTerminationReason.COMPLETED
	)

	@property
	def completed_depth(self) -> int | None:
		"""Return the completed search depth, or None if not applicable."""
		return self.statistics.completed_depth

	def __str__(self) -> str:
		score = "n/a" if self.score is None else f"{self.score:.2f}"

		return "\n".join([
			"SEARCH RESULT",
			"=" * 35,
			f"{'Best move:':<25} {self.best_move}",
			f"{'Score:':<25} {score}",
			f"{'Completed:':<25} {self.completed}",
			f"{'Termination reason:':<25} {self.termination_reason.name}",
			"",
			str(self.statistics),
		])

