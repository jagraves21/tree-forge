from __future__ import annotations

from collections.abc import Sequence
from math import inf
from time import perf_counter
from typing import Generic, TypeVar

from ..caching import (
	TranspositionBound,
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


class Minimax(Generic[StateT, MoveT, PlayerT, UndoT]):
	"""Minimax search with optional alpha-beta pruning and caching."""

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
		statistics = SearchStatistics()

		root_player = self.game.current_player(state)

		def evaluate_position() -> float:
			if self.evaluator is None:
				return 0.0

			return self.evaluator.evaluate(state, root_player)

		def update_dependencies(
			canonical_position,
			child_keys: set,
			preserve_existing: bool = False,
		) -> None:
			if (
				self.transposition_table is None
				or canonical_position is None
			):
				return

			if preserve_existing:
				for child_key in child_keys:
					self.transposition_table.register_dependency(
						canonical_position.key,
						child_key,
					)
			else:
				self.transposition_table.replace_dependencies(
					canonical_position.key,
					child_keys,
				)

		def store_entry(
			canonical_position,
			depth: int | None,
			score: float,
			bound: TranspositionBound,
			best_move: MoveT | None,
			maximizing: bool,
			solved: bool,
		) -> None:
			if (
				self.transposition_table is None
				or canonical_position is None
			):
				return

			# Store scores from the perspective of the player
			# whose turn it is at this position.
			stored_score = score if maximizing else -score

			stored_bound = bound

			# Negating a score reverses its bound.
			if not maximizing:
				if bound == TranspositionBound.LOWER:
					stored_bound = TranspositionBound.UPPER
				elif bound == TranspositionBound.UPPER:
					stored_bound = TranspositionBound.LOWER

			stored_move = (
				canonical_position.to_canonical_move(best_move)
				if best_move is not None
				else None
			)

			invalidated = self.transposition_table.store(
				TranspositionEntry(
					key=canonical_position.key,
					depth=depth,
					score=stored_score,
					bound=stored_bound,
					best_move=stored_move,
					solved=solved,
				)
			)

			statistics.transposition_table_invalidations += invalidated
			statistics.transposition_table_stores += 1

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
		) -> tuple[float, bool]:
			"""Return the score and whether it is horizon-independent."""
			statistics.nodes_visited += 1
			statistics.max_ply_reached = max(
				statistics.max_ply_reached,
				ply,
			)

			player = self.game.current_player(state)
			maximizing = player == root_player

			# Preserve the original search window for bound classification.
			original_alpha = alpha
			original_beta = beta

			canonical_position = None
			cached_move = None
			used_cached_bound = False

			# Look up the current position.
			if self.transposition_table is not None:
				assert self.canonicalizer is not None

				canonical_position = self.canonicalizer.canonicalize(
					state
				)

				entry = self.transposition_table.get(
					canonical_position.key
				)

				if entry is not None:
					statistics.transposition_table_hits += 1

					if entry.best_move is not None:
						cached_move = (
							canonical_position.from_canonical_move(
								entry.best_move
							)
						)

					if entry.is_usable_for_depth(depth):
						score = (
							entry.score
							if maximizing
							else -entry.score
						)

						bound = entry.bound

						# Negating the score also reverses its bound.
						if not maximizing:
							if bound == TranspositionBound.LOWER:
								bound = TranspositionBound.UPPER
							elif bound == TranspositionBound.UPPER:
								bound = TranspositionBound.LOWER

						if bound == TranspositionBound.EXACT:
							return score, entry.solved

						if self.alpha_beta:
							if bound == TranspositionBound.LOWER:
								alpha = max(alpha, score)
							elif bound == TranspositionBound.UPPER:
								beta = min(beta, score)

							if alpha >= beta:
								statistics.cutoffs += 1
								return score, False

							used_cached_bound = True

			# Handle terminal positions.
			if self.game.is_terminal(state):
				statistics.leaf_nodes += 1
				statistics.terminal_nodes += 1

				score = self.game.terminal_score(
					state,
					root_player,
				)

				update_dependencies(
					canonical_position,
					set(),
				)

				store_entry(
					canonical_position,
					depth,
					score,
					TranspositionBound.EXACT,
					None,
					maximizing,
					True,
				)

				return score, True

			# Handle depth-limited positions.
			if depth == 0:
				statistics.leaf_nodes += 1
				score = evaluate_position()

				update_dependencies(
					canonical_position,
					set(),
				)

				store_entry(
					canonical_position,
					depth,
					score,
					TranspositionBound.EXACT,
					None,
					maximizing,
					False,
				)

				return score, False

			moves = order_moves(
				state,
				self.game.legal_moves(state),
				player,
			)

			# Handle positions without legal moves.
			if not moves:
				statistics.leaf_nodes += 1
				score = evaluate_position()

				update_dependencies(
					canonical_position,
					set(),
				)

				store_entry(
					canonical_position,
					depth,
					score,
					TranspositionBound.EXACT,
					None,
					maximizing,
					False,
				)

				return score, False

			# Search the cached best move first when it is legal.
			if cached_move in moves:
				moves.remove(cached_move)
				moves.insert(0, cached_move)

			best_score = -inf if maximizing else inf
			best_move = None
			all_children_solved = True
			cutoff_occurred = False
			searched_child_keys = set()

			next_depth = (
				depth - 1
				if depth is not None
				else None
			)

			for move in moves:
				undo_data = self.game.make_move(state, move)

				try:
					child_canonical_position = None

					if self.transposition_table is not None:
						assert self.canonicalizer is not None

						child_canonical_position = (
							self.canonicalizer.canonicalize(state)
						)

					score, child_solved = minimax(
						next_depth,
						ply + 1,
						alpha,
						beta,
					)

					if child_canonical_position is not None:
						searched_child_keys.add(
							child_canonical_position.key
						)
				finally:
					self.game.unmake_move(
						state,
						move,
						undo_data,
					)

				if not child_solved:
					all_children_solved = False

				if (
					best_move is None
					or (maximizing and score > best_score)
					or (not maximizing and score < best_score)
				):
					best_move = move
					best_score = score

				if maximizing:
					if self.alpha_beta:
						alpha = max(alpha, best_score)
				else:
					if self.alpha_beta:
						beta = min(beta, best_score)

				if self.alpha_beta and alpha >= beta:
					statistics.cutoffs += 1
					cutoff_occurred = True
					break

			# A node is solved only when every legal continuation
			# was searched and every child was independently solved.
			solved = all_children_solved and not cutoff_occurred

			# Fully solved children establish an exact result regardless
			# of the alpha-beta window used to reach this node.
			bound = TranspositionBound.EXACT

			if not solved:
				if best_score <= original_alpha:
					bound = TranspositionBound.UPPER
				elif best_score >= original_beta:
					bound = TranspositionBound.LOWER

			# If a cached bound contributed to this search, retain
			# its existing dependencies and add the newly searched
			# children. Otherwise, replace the old dependency set.
			update_dependencies(
				canonical_position,
				searched_child_keys,
				preserve_existing=used_cached_bound,
			)

			store_entry(
				canonical_position,
				depth,
				best_score,
				bound,
				best_move,
				maximizing,
				solved,
			)

			return best_score, solved

		# Handle terminal root positions.
		if self.game.is_terminal(state):
			statistics.nodes_visited = 1
			statistics.leaf_nodes = 1
			statistics.terminal_nodes = 1

			score = self.game.terminal_score(state, root_player)

			statistics.elapsed_time_seconds = (
				perf_counter() - start_time
			)

			return SearchResult(
				best_move=None,
				score=score,
				statistics=statistics,
				termination_reason=SearchTerminationReason.COMPLETED,
			)

		moves = order_moves(
			state,
			self.game.legal_moves(state),
			root_player,
		)

		# Handle positions without legal moves.
		if not moves:
			statistics.nodes_visited = 1
			statistics.leaf_nodes = 1

			score = evaluate_position()

			statistics.elapsed_time_seconds = (
				perf_counter() - start_time
			)

			return SearchResult(
				best_move=None,
				score=score,
				statistics=statistics,
				termination_reason=SearchTerminationReason.COMPLETED,
			)

		best_move: MoveT | None = None
		best_score = -inf
		alpha = -inf
		beta = inf

		depth = self.limits.max_depth
		child_depth = depth - 1 if depth is not None else None

		for move in moves:
			undo_data = self.game.make_move(state, move)

			try:
				score, _ = minimax(
					child_depth,
					1,
					alpha,
					beta,
				)
			finally:
				self.game.unmake_move(
					state,
					move,
					undo_data,
				)

			if best_move is None or score > best_score:
				best_move = move
				best_score = score

			if self.alpha_beta:
				alpha = max(alpha, best_score)

		# The fixed-depth search completed successfully.
		if depth is not None:
			statistics.completed_depth = depth

		statistics.elapsed_time_seconds = (
			perf_counter() - start_time
		)

		return SearchResult(
			best_move=best_move,
			score=best_score,
			statistics=statistics,
			termination_reason=SearchTerminationReason.COMPLETED,
		)

