import random
from collections.abc import Sequence

from tree_forge.core import MoveOrdering

from .game import GameState, Move, Player


class TicTacToeMoveOrdering(
	MoveOrdering[GameState, Move, Player]
):
	"""Orders Tic-Tac-Toe moves by positional preference.

	Priority:
		1. Center
		2. Corners
		3. Edges

	Move order within each priority group is preserved.
	"""

	PRIORITIES = {
		(1, 1): 0,
		(0, 0): 1,
		(0, 2): 1,
		(2, 0): 1,
		(2, 2): 1,
		(0, 1): 2,
		(1, 0): 2,
		(1, 2): 2,
		(2, 1): 2,
	}

	def order_moves(
		self,
		state: GameState,
		moves: Sequence[Move],
		player: Player,
	) -> Sequence[Move]:
		return sorted(
			moves,
			key=lambda move: self.PRIORITIES[
				(move.row, move.column)
			],
		)


class RandomizedTicTacToeMoveOrdering(
	TicTacToeMoveOrdering
):
	"""Orders moves by positional preference, randomly shuffling
	moves within each priority group.

	Priority:
		1. Center
		2. Corners
		3. Edges
	"""

	def order_moves(
		self,
		state: GameState,
		moves: Sequence[Move],
		player: Player,
	) -> Sequence[Move]:
		groups: dict[int, list[Move]] = {}

		for move in moves:
			priority = self.PRIORITIES[
				(move.row, move.column)
			]
			groups.setdefault(priority, []).append(move)

		ordered_moves = []

		for priority in sorted(groups):
			group = groups[priority]
			random.shuffle(group)
			ordered_moves.extend(group)

		return ordered_moves


class RandomTicTacToeMoveOrdering(
	MoveOrdering[GameState, Move, Player]
):
	"""Orders Tic-Tac-Toe moves randomly, ignoring positional
	preferences and priority groups.
	"""

	def order_moves(
		self,
		state: GameState,
		moves: Sequence[Move],
		player: Player,
	) -> Sequence[Move]:
		ordered_moves = list(moves)
		random.shuffle(ordered_moves)
		return ordered_moves

