from __future__ import annotations

from random import choice
from typing import Generic, TypeVar

from tree_forge.core.interfaces import Game
from tree_forge.core.models import (
	SearchResult,
	SearchStatistics,
	SearchTerminationReason,
)


StateT = TypeVar("StateT")
MoveT = TypeVar("MoveT")
PlayerT = TypeVar("PlayerT")
UndoT = TypeVar("UndoT")


class RandomSearch(Generic[StateT, MoveT, PlayerT, UndoT]):
	"""Select a random legal move.

	Positions are not evaluated, so the result has no score, and the
	only statistic measured is the position inspected: the root.
	"""

	def __init__(
		self,
		game: Game[StateT, MoveT, PlayerT, UndoT],
	) -> None:
		self.game = game

	def search(self, state: StateT) -> SearchResult[MoveT]:
		moves = self.game.legal_moves(state)

		statistics = SearchStatistics(nodes_visited=1)

		if not moves:
			return SearchResult(
				best_move=None,
				score=None,
				statistics=statistics,
				termination_reason=SearchTerminationReason.COMPLETED,
			)

		return SearchResult(
			best_move=choice(moves),
			score=None,
			statistics=statistics,
			termination_reason=SearchTerminationReason.COMPLETED,
		)

