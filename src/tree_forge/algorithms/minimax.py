from __future__ import annotations

from math import inf
from time import perf_counter
from typing import Generic, TypeVar

from ..core import (
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
	"""Exhaustive Minimax search for a two-player zero-sum game."""

	def __init__(
		self,
		game: Game[StateT, MoveT, PlayerT, UndoT],
		evaluator: Evaluator[StateT, PlayerT] | None = None,
		limits: SearchLimits | None = None,
	) -> None:
		self.game = game
		self.evaluator = evaluator
		self.limits = limits or SearchLimits()

	def search(self, state: StateT) -> SearchResult[MoveT]:
		"""Find the best move for the current player."""
		start_time = perf_counter()
		statistics = SearchStatistics()

		root_player = self.game.current_player(state)

		def evaluate_position() -> float:
			if self.evaluator is None:
				return 0.0

			return self.evaluator.evaluate(state, root_player)

		def minimax(
			depth: int | None,
			ply: int,
		) -> float:
			statistics.nodes_visited += 1
			statistics.completed_depth = max(
				statistics.completed_depth,
				ply,
			)

			if self.game.is_terminal(state):
				statistics.leaf_nodes += 1
				statistics.terminal_nodes += 1

				return self.game.terminal_score(state, root_player)

			if depth == 0:
				statistics.leaf_nodes += 1
				return evaluate_position()

			moves = self.game.legal_moves(state)

			if not moves:
				statistics.leaf_nodes += 1
				return evaluate_position()

			maximizing = (
				self.game.current_player(state) == root_player
			)

			best_score = -inf if maximizing else inf

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
					)
				finally:
					self.game.unmake_move(
						state,
						move,
						undo_data,
					)

				if maximizing:
					best_score = max(best_score, score)
				else:
					best_score = min(best_score, score)

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

		depth = self.limits.max_depth
		child_depth = depth - 1 if depth is not None else None

		for move in moves:
			undo_data = self.game.make_move(state, move)

			try:
				score = minimax(
					child_depth,
					1,
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

		statistics.elapsed_time_seconds = (
			perf_counter() - start_time
		)

		return SearchResult(
			best_move=best_move,
			score=best_score,
			statistics=statistics,
			termination_reason=SearchTerminationReason.COMPLETED,
		)

