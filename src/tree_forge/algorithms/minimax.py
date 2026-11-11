from __future__ import annotations

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

	def search(self, state: StateT) -> SearchResult[MoveT]:
		"""Find the best move for the current player."""
		start_time = perf_counter()
		statistics = SearchStatistics()

		root_player = self.game.current_player(state)

		def evaluate_position() -> float:
			if self.evaluator is None:
				return 0.0

			return self.evaluator.evaluate(state, root_player)

		def store_entry(
			canonical_position,
			depth: int | None,
			score: float,
			bound: TranspositionBound,
			best_move: MoveT | None,
			maximizing: bool,
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

			self.transposition_table.store(
				TranspositionEntry(
					key=canonical_position.key,
					depth=depth,
					score=stored_score,
					bound=stored_bound,
					best_move=stored_move,
				)
			)

			statistics.transposition_table_stores += 1

		def minimax(
			depth: int | None,
			ply: int,
			alpha: float,
			beta: float,
		) -> float:
			statistics.nodes_visited += 1
			statistics.completed_depth = max(
				statistics.completed_depth,
				ply,
			)

			player = self.game.current_player(state)
			maximizing = player == root_player

			# Preserve the original search window for bound classification.
			original_alpha = alpha
			original_beta = beta

			canonical_position = None
			cached_move = None

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

					# Reuse scores only when the cached search
					# reached at least the requested depth.
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
							return score

						if self.alpha_beta:
							if bound == TranspositionBound.LOWER:
								alpha = max(alpha, score)
							elif bound == TranspositionBound.UPPER:
								beta = min(beta, score)

							if self.alpha_beta and alpha >= beta:
								statistics.cutoffs += 1
								return score

			# Handle terminal positions.
			if self.game.is_terminal(state):
				statistics.leaf_nodes += 1
				statistics.terminal_nodes += 1

				score = self.game.terminal_score(
					state,
					root_player,
				)

				store_entry(
					canonical_position,
					depth,
					score,
					TranspositionBound.EXACT,
					None,
					maximizing,
				)

				return score

			# Handle depth-limited positions.
			if depth == 0:
				statistics.leaf_nodes += 1
				score = evaluate_position()

				store_entry(
					canonical_position,
					depth,
					score,
					TranspositionBound.EXACT,
					None,
					maximizing,
				)

				return score

			moves = list(self.game.legal_moves(state))

			# Handle positions without legal moves.
			if not moves:
				statistics.leaf_nodes += 1
				score = evaluate_position()

				store_entry(
					canonical_position,
					depth,
					score,
					TranspositionBound.EXACT,
					None,
					maximizing,
				)

				return score

			# Search the cached best move first when it is legal.
			if cached_move in moves:
				moves.remove(cached_move)
				moves.insert(0, cached_move)

			best_score = -inf if maximizing else inf
			best_move = None

			next_depth = (
				depth - 1
				if depth is not None
				else None
			)

			for move in moves:
				undo_data = self.game.make_move(state, move)

				try:
					score = minimax(
						next_depth,
						ply + 1,
						alpha,
						beta,
					)
				finally:
					self.game.unmake_move(
						state,
						move,
						undo_data,
					)

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
					break

			# Classify the result against the original search window.
			bound = TranspositionBound.EXACT

			if best_score <= original_alpha:
				bound = TranspositionBound.UPPER
			elif best_score >= original_beta:
				bound = TranspositionBound.LOWER

			store_entry(
				canonical_position,
				depth,
				best_score,
				bound,
				best_move,
				maximizing,
			)

			return best_score

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

		moves = self.game.legal_moves(state)

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
				score = minimax(
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

		statistics.elapsed_time_seconds = (
			perf_counter() - start_time
		)

		return SearchResult(
			best_move=best_move,
			score=best_score,
			statistics=statistics,
			termination_reason=SearchTerminationReason.COMPLETED,
		)

