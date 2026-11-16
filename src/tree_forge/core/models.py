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

	Counters are cumulative across iterative-deepening passes.

	completed_depth represents the deepest fully completed
	iteration. max_ply_reached represents the deepest recursive
	ply visited, even if that iteration was interrupted.
	"""

	nodes_visited: int = 0
	leaf_nodes: int = 0
	terminal_nodes: int = 0

	cutoffs: int = 0

	transposition_table_hits: int = 0
	transposition_table_stores: int = 0
	transposition_table_invalidations: int = 0

	iterations_completed: int = 0

	elapsed_time_seconds: float = 0.0
	completed_depth: int = 0
	max_ply_reached: int = 0

	def __str__(self) -> str:
		return "\n".join([
			"SEARCH STATISTICS",
			"=" * 35,
			f"{'Nodes visited:':<25} {self.nodes_visited:,}",
			f"{'Leaf nodes:':<25} {self.leaf_nodes:,}",
			f"{'Terminal nodes:':<25} {self.terminal_nodes:,}",
			f"{'Alpha-beta cutoffs:':<25} {self.cutoffs:,}",
			f"{'Transposition hits:':<25} {self.transposition_table_hits:,}",
			f"{'Transposition stores:':<25} {self.transposition_table_stores:,}",
			f"{'Transposition invalidations:':<25} {self.transposition_table_invalidations:,}",
			f"{'Search time:':<25} {self.elapsed_time_seconds:.6f}s",
			f"{'Completed depth:':<25} {self.completed_depth}",
			f"{'Iterations completed:':<25} {self.iterations_completed}",
			f"{'Maximum ply reached:':<25} {self.max_ply_reached}",
			"=" * 35,
		])


@dataclass(slots=True)
class SearchResult(Generic[MoveT]):
	"""Result returned by a search operation.

	If a search is interrupted, best_move and score should represent
	the last successfully completed search iteration, when available.
	"""

	best_move: MoveT | None
	score: float

	statistics: SearchStatistics = field(
		default_factory=SearchStatistics
	)

	completed: bool = True

	termination_reason: SearchTerminationReason = (
		SearchTerminationReason.COMPLETED
	)

	@property
	def completed_depth(self) -> int:
		"""Return the deepest fully completed search depth."""
		return self.statistics.completed_depth

	def __str__(self) -> str:
		return "\n".join([
			"SEARCH RESULT",
			"=" * 35,
			f"{'Best move:':<25} {self.best_move}",
			f"{'Score:':<25} {self.score:.2f}",
			f"{'Completed:':<25} {self.completed}",
			f"{'Termination reason:':<25} {self.termination_reason.value}",
			"",
			str(self.statistics),
		])

