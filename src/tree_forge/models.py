
from dataclasses import dataclass
from typing import Generic, TypeVar


MoveT = TypeVar("MoveT")


@dataclass(frozen=True, slots=True)
class SearchConfig:
	"""Configuration shared by search algorithms."""

	max_depth: int = 1
	time_limit_seconds: float | None = None
	node_limit: int | None = None

	def __post_init__(self) -> None:
		if self.max_depth < 1:
			raise ValueError("max_depth must be at least 1")

		if self.time_limit_seconds is not None:
			if self.time_limit_seconds <= 0:
				raise ValueError("time_limit_seconds must be positive")

		if self.node_limit is not None:
			if self.node_limit < 1:
				raise ValueError("node_limit must be at least 1")


@dataclass(slots=True)
class SearchStats:
	"""Statistics collected during a search."""

	nodes: int = 0
	cutoffs: int = 0
	transposition_hits: int = 0
	elapsed_seconds: float = 0.0
	completed_depth: int = 0


@dataclass(slots=True)
class SearchResult(Generic[MoveT]):
	"""Result returned by a search algorithm."""

	best_move: MoveT | None
	score: float
	stats: SearchStats


