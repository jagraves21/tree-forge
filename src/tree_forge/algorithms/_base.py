from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from time import perf_counter
from typing import Generic, NamedTuple, TypeVar

from ..caching import TranspositionTable
from ..core import (
	Canonicalizer,
	Evaluator,
	Game,
	MoveOrdering,
	SearchLimits,
	SearchResult,
	SearchStatistics,
	SearchTerminationReason,
)


StateT = TypeVar("StateT")
MoveT = TypeVar("MoveT")
PlayerT = TypeVar("PlayerT")
UndoT = TypeVar("UndoT")


def negated(score: float) -> float:
	"""Return the opposite score, without producing a negative zero."""
	return -score + 0.0


class _Node(NamedTuple):
	"""Outcome of searching one position."""

	score: float
	solved: bool
	best_move: object | None


class _Counters:
	"""Mutable counters collected while a search runs."""

	__slots__ = (
		"nodes",
		"leaves",
		"terminals",
		"cutoffs",
		"hits",
		"stores",
		"invalidations",
		"refreshes",
		"max_ply",
	)

	def __init__(self) -> None:
		self.nodes = 0
		self.leaves = 0
		self.terminals = 0
		self.cutoffs = 0
		self.hits = 0
		self.stores = 0
		self.invalidations = 0
		self.refreshes = 0
		self.max_ply = 0


class _TreeSearch(ABC, Generic[StateT, MoveT, PlayerT, UndoT]):
	"""Skeleton shared by the tree search algorithms.

	The base class owns everything around the recursion: the
	configuration, the validation of the move ordering, the counters and
	the construction of the result and its statistics. A concrete
	algorithm only implements _search_root(), which searches the root
	position to a given depth and returns the node it found.

	Scores are from the perspective of the player to move at the root.
	"""

	def __init__(
		self,
		game: Game[StateT, MoveT, PlayerT, UndoT],
		evaluator: Evaluator[StateT, PlayerT] | None = None,
		limits: SearchLimits | None = None,
		alpha_beta: bool = False,
		canonicalizer: Canonicalizer[StateT, MoveT] | None = None,
		transposition_table: TranspositionTable[MoveT] | None = None,
		move_ordering: MoveOrdering[StateT, MoveT, PlayerT] | None = None,
	) -> None:
		if transposition_table is not None and canonicalizer is None:
			raise ValueError(
				"A canonicalizer is required when using a transposition table"
			)

		self.game = game
		self.evaluator = evaluator if evaluator is not None else Evaluator()
		self.limits = limits or SearchLimits()
		self.alpha_beta = alpha_beta
		self.canonicalizer = canonicalizer

		if canonicalizer is not None:
			self.transposition_table = (
				transposition_table
				if transposition_table is not None
				else TranspositionTable()
			)
		else:
			self.transposition_table = None

		self.move_ordering = move_ordering

	def search(self, state: StateT) -> SearchResult[MoveT]:
		"""Find the best move for the current player."""
		start_time = perf_counter()
		counters = _Counters()

		root_depth = self.limits.max_depth
		result = self._search_root(state, root_depth, counters)

		return SearchResult(
			best_move=result.best_move,
			score=result.score,
			statistics=self._statistics(counters, root_depth, start_time),
			termination_reason=SearchTerminationReason.COMPLETED,
		)

	@abstractmethod
	def _search_root(
		self,
		state: StateT,
		root_depth: int | None,
		counters: _Counters,
	) -> _Node:
		"""Search the root position to the given depth.

		Implementations update the counters while they search and
		return the root's score, whether it is solved, and its best
		move.
		"""
		raise NotImplementedError

	def _statistics(
		self,
		counters: _Counters,
		completed_depth: int | None,
		start_time: float,
	) -> SearchStatistics:
		"""Convert the counters of a search into its statistics.

		Statistics that were not measured, such as cutoffs without
		alpha-beta pruning or table statistics without a table, are
		None.
		"""
		table = self.transposition_table
		tracking = table is not None and table.dependency_tracking

		return SearchStatistics(
			nodes_visited=counters.nodes,
			leaf_nodes=counters.leaves,
			terminal_nodes=counters.terminals,
			cutoffs=counters.cutoffs if self.alpha_beta else None,
			transposition_table_hits=(
				counters.hits if table is not None else None
			),
			transposition_table_stores=(
				counters.stores if table is not None else None
			),
			transposition_table_invalidations=(
				counters.invalidations if tracking else None
			),
			transposition_table_refreshes=(
				counters.refreshes if tracking else None
			),
			transposition_table_size=(
				len(table) if table is not None else None
			),
			iterations_completed=None,
			elapsed_time_seconds=perf_counter() - start_time,
			completed_depth=completed_depth,
			max_ply_reached=counters.max_ply,
		)

	def _order_moves(
		self,
		state: StateT,
		moves: Sequence[MoveT],
		player: PlayerT,
	) -> list[MoveT]:
		"""Return the legal moves in the move ordering's preferred order.

		The ordering may only rearrange the moves it was given.
		"""
		ordered = list(moves)

		if self.move_ordering is None:
			return ordered

		ordered = list(
			self.move_ordering.order_moves(
				state,
				ordered,
				player,
			)
		)

		# Validate that ordering only rearranged the original moves.
		remaining = list(moves)

		if len(ordered) != len(remaining):
			raise ValueError(
				"MoveOrdering must not add or omit legal moves"
			)

		for move in ordered:
			try:
				index = remaining.index(move)
			except ValueError as error:
				raise ValueError(
					"MoveOrdering must not add or omit legal moves"
				) from error

			remaining.pop(index)

		return ordered
