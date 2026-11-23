from __future__ import annotations

from collections.abc import Hashable, Sequence
from math import inf, nextafter
from time import perf_counter
from typing import Generic, NamedTuple, TypeVar

from ..caching import (
	TranspositionEntry,
	TranspositionTable,
)
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


class Minimax(Generic[StateT, MoveT, PlayerT, UndoT]):
	"""Minimax search with optional alpha-beta pruning and caching.

	Scores are from the perspective of the player to move at the
	root. Terminal positions are scored by the game, and positions at
	the depth limit by the evaluator (or 0 without one).

	With a canonicalizer, every searched position, including the
	root, is cached in a transposition table as an interval of
	possible values. Cached information about a position is reused
	when it was gathered with at least the requested depth, or when
	it is proven (solved). The table must belong to a single game and
	evaluator, and the evaluator is assumed to be zero-sum, so that
	scores can be stored from the perspective of the player to move.

	When the table tracks dependencies, every position registers all
	of its legal continuations as dependencies, searched or not.
	Information discovered by later searches therefore improves the
	cached values of positions that were searched earlier, as
	described in TranspositionTable.

	Of several root moves that are equally good, the one that the move
	ordering ranks first is returned, whatever order they were searched
	in and whatever earlier searches cached. A move ordering can therefore
	express preferences that scores cannot, such as stopping an opponent's
	threat when every move loses at the same time. It never makes the
	search return a move with a worse score.

	Statistics that are not measured, such as cutoffs without
	alpha-beta pruning or table statistics without a table, are None.
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
		self.evaluator = evaluator
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

		table = self.transposition_table
		canonicalizer = self.canonicalizer
		tracking = table is not None and table.dependency_tracking

		root_player = self.game.current_player(state)
		root_depth = self.limits.max_depth

		def evaluate_position() -> float:
			if self.evaluator is None:
				return 0.0

			return self.evaluator.evaluate(state, root_player)

		def to_root_interval(
			entry: TranspositionEntry[MoveT],
			maximizing: bool,
		) -> tuple[float | None, float | None]:
			"""Convert a stored interval to the root player's perspective."""
			if maximizing:
				return entry.lower, entry.upper

			# Negating a score swaps its bounds.
			return (
				None if entry.upper is None else negated(entry.upper),
				None if entry.lower is None else negated(entry.lower),
			)

		def record(outcome) -> None:
			"""Count the stale and recomputed entries reported by the table."""
			counters.invalidations += outcome.invalidated or 0
			counters.refreshes += outcome.refreshed or 0

		def store_entry(
			position,
			depth: int | None,
			lower: float | None,
			upper: float | None,
			best_move: MoveT | None,
			maximizing: bool,
			solved: bool,
		) -> bool:
			"""Store what is known about a position, from the root's view.

			Returns whether the table changed.
			"""
			if table is None or position is None:
				return False

			# Store scores from the perspective of the player
			# whose turn it is at this position.
			if not maximizing:
				lower, upper = (
					None if upper is None else negated(upper),
					None if lower is None else negated(lower),
				)

			outcome = table.store(
				TranspositionEntry(
					key=position.key,
					depth=depth,
					lower=lower,
					upper=upper,
					best_move=(
						position.to_canonical_move(best_move)
						if best_move is not None
						else None
					),
					solved=solved,
				)
			)

			if outcome.stored:
				counters.stores += 1

			record(outcome)

			return outcome.stored

		def finish_leaf(
			position,
			depth: int | None,
			score: float,
			maximizing: bool,
			solved: bool,
		) -> None:
			"""Record a position that is not expanded."""
			stored = store_entry(
				position,
				depth,
				score,
				score,
				None,
				maximizing,
				solved,
			)

			# The position may have been expanded by an earlier search. Its
			# dependencies only go away if this entry replaced that result;
			# a deeper result that was kept still depends on its children.
			if stored and table is not None and position is not None:
				record(table.replace_dependencies(position.key, {}))

		def continuation(maximizing: bool):
			"""Describe the current position as a continuation of its parent.

			Returns its canonical position and relation: +1 if its score
			has the same perspective as the parent's, -1 if the
			perspective is opposite.
			"""
			assert canonicalizer is not None

			child_maximizing = (
				self.game.current_player(state) == root_player
			)

			return (
				canonicalizer.canonicalize(state),
				1 if child_maximizing == maximizing else -1,
			)

		def order_moves(
			position_state: StateT,
			moves: Sequence[MoveT],
			player: PlayerT,
		) -> list[MoveT]:
			ordered = list(moves)

			if self.move_ordering is None:
				return ordered

			ordered = list(
				self.move_ordering.order_moves(
					position_state,
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

		def minimax(
			depth: int | None,
			ply: int,
			alpha: float,
			beta: float,
			position=None,
		) -> _Node:
			"""Search a position.

			The score is from the root player's perspective. It is exact
			if it lies strictly between alpha and beta, and otherwise a
			bound. It is solved if it is proven independently of the
			depth limit.
			"""
			counters.nodes += 1
			counters.max_ply = max(counters.max_ply, ply)

			player = self.game.current_player(state)
			maximizing = player == root_player

			cached_move = None

			# Look up the current position. Its canonical position is
			# reused if the caller already computed it.
			if table is not None:
				assert canonicalizer is not None

				if position is None:
					position = canonicalizer.canonicalize(state)

				entry = table.get(position.key)

				if entry is not None:
					counters.hits += 1

					if entry.best_move is not None:
						cached_move = position.from_canonical_move(
							entry.best_move
						)

					# The root is always searched, because a move is required.
					if ply > 0 and entry.is_usable_for_depth(depth):
						lower, upper = to_root_interval(entry, maximizing)

						if lower is not None and lower == upper:
							return _Node(lower, entry.solved, None)

						if self.alpha_beta:
							if lower is not None and lower >= beta:
								counters.cutoffs += 1

								return _Node(lower, False, None)

							if upper is not None and upper <= alpha:
								counters.cutoffs += 1

								return _Node(upper, False, None)

							if lower is not None:
								alpha = max(alpha, lower)

							if upper is not None:
								beta = min(beta, upper)

			# Bounds are classified against the window actually searched.
			window_alpha = alpha
			window_beta = beta

			# Handle terminal positions.
			if self.game.is_terminal(state):
				counters.leaves += 1
				counters.terminals += 1

				score = self.game.terminal_score(state, root_player)

				finish_leaf(position, depth, score, maximizing, True)

				return _Node(score, True, None)

			# Handle depth-limited positions.
			if depth == 0:
				counters.leaves += 1
				score = evaluate_position()

				finish_leaf(position, depth, score, maximizing, False)

				return _Node(score, False, None)

			moves = order_moves(
				state,
				self.game.legal_moves(state),
				player,
			)

			# Handle positions without legal moves.
			if not moves:
				counters.leaves += 1
				score = evaluate_position()

				finish_leaf(position, depth, score, maximizing, False)

				return _Node(score, False, None)

			# Every move remembers its place in the move ordering, which
			# decides between equally good moves at the root.
			ranked = list(enumerate(moves))

			# Search the cached best move first when it is legal.
			if cached_move in moves:
				ranked.insert(0, ranked.pop(moves.index(cached_move)))

			best_score = -inf if maximizing else inf
			best_move = None
			all_children_solved = True
			cutoff_occurred = False
			continuations: dict[Hashable, int] = {}
			searched = 0

			next_depth = depth - 1 if depth is not None else None

			best_rank = 0

			for rank, move in ranked:
				undo_data = self.game.make_move(state, move)
				searched += 1

				try:
					child_position = None

					if tracking:
						child_position, relation = continuation(maximizing)
						continuations[child_position.key] = relation

					# At the root, a move that ranks above the best move so far
					# wins a tie, so it is searched for a score that equals the
					# best score instead of one that beats it.
					child_alpha = alpha

					if (
						ply == 0
						and self.alpha_beta
						and best_move is not None
						and rank < best_rank
					):
						child_alpha = nextafter(alpha, -inf)

					score, child_solved, _ = minimax(
						next_depth,
						ply + 1,
						child_alpha,
						beta,
						child_position,
					)
				finally:
					self.game.unmake_move(
						state,
						move,
						undo_data,
					)

				if not child_solved:
					all_children_solved = False

				# At the root, equally good moves are decided by the move
				# ordering, no matter in which order they were searched. A
				# move found later is only preferred if it ranks higher.
				if (
					best_move is None
					or (maximizing and score > best_score)
					or (not maximizing and score < best_score)
					or (ply == 0 and score == best_score and rank < best_rank)
				):
					best_move = move
					best_score = score
					best_rank = rank

				if maximizing:
					if self.alpha_beta:
						alpha = max(alpha, best_score)
				else:
					if self.alpha_beta:
						beta = min(beta, best_score)

				if self.alpha_beta and alpha >= beta:
					counters.cutoffs += 1
					cutoff_occurred = True
					break

			# The result also depends on the continuations that were not
			# searched: information about them may arrive from later
			# searches, so they are dependencies as well.
			if tracking and position is not None:
				for _, move in ranked[searched:]:
					undo_data = self.game.make_move(state, move)

					try:
						child_position, relation = continuation(maximizing)
						continuations[child_position.key] = relation
					finally:
						self.game.unmake_move(
							state,
							move,
							undo_data,
						)

				continuations.pop(position.key, None)

			# A node is solved only when every legal continuation
			# was searched and every child was independently solved.
			solved = all_children_solved and not cutoff_occurred

			# Fully solved children establish an exact result regardless
			# of the alpha-beta window used to reach this node.
			if solved:
				lower, upper = best_score, best_score
			elif best_score <= window_alpha:
				lower, upper = None, best_score
			elif best_score >= window_beta:
				lower, upper = best_score, None
			else:
				lower, upper = best_score, best_score

			stored = store_entry(
				position,
				depth,
				lower,
				upper,
				best_move,
				maximizing,
				solved,
			)

			# Dependencies describe the entry that is kept: if a deeper
			# result for this position was kept instead, its dependencies
			# stay as they are.
			if stored and tracking and position is not None:
				assert table is not None
				record(table.replace_dependencies(position.key, continuations))

			return _Node(best_score, solved, best_move)

		result = minimax(root_depth, 0, -inf, inf)

		statistics = SearchStatistics(
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
			completed_depth=root_depth,
			max_ply_reached=counters.max_ply,
		)

		return SearchResult(
			best_move=result.best_move,
			score=result.score,
			statistics=statistics,
			termination_reason=SearchTerminationReason.COMPLETED,
		)
