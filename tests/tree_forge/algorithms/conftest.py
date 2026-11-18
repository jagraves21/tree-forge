"""Test doubles shared by the search algorithm tests.

The doubles describe tiny games as explicit trees so that every expected
score, node count, and cutoff in the tests can be derived by hand.
"""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import pytest

from tree_forge.core.interfaces import (
	Canonicalizer,
	Evaluator,
	Game,
	MoveOrdering,
)
from tree_forge.core.models import CanonicalPosition


# Two plies. Player 1 moves at the root and player 2 replies, so the
# minimax value of "a" is min(3, 5) = 3, "b" is min(2, 9) = 2, and the
# root value is max(3, 2) = 3.
SHALLOW_TREE = {
	"root": ("a", "b"),
	"a": ("a1", "a2"),
	"b": ("b1", "b2"),
}

SHALLOW_SCORES = {
	"a1": 3.0,
	"a2": 5.0,
	"b1": 2.0,
	"b2": 9.0,
}

# Three plies: player 1 (max), player 2 (min), player 1 (max).
#
#   root = max(A, B)
#   A    = min(A1, A2) = min(max(3, 5), max(6, 9)) = min(5, 9) = 5
#   B    = min(B1, B2) = min(max(1, 2), max(0, -1)) = min(2, 0) = 0
#
# so the root value is 5 and the best move is "A".
DEEP_TREE = {
	"root": ("A", "B"),
	"A": ("A1", "A2"),
	"B": ("B1", "B2"),
	"A1": ("A1a", "A1b"),
	"A2": ("A2a", "A2b"),
	"B1": ("B1a", "B1b"),
	"B2": ("B2a", "B2b"),
}

DEEP_SCORES = {
	"A1a": 3.0,
	"A1b": 5.0,
	"A2a": 6.0,
	"A2b": 9.0,
	"B1a": 1.0,
	"B1b": 2.0,
	"B2a": 0.0,
	"B2b": -1.0,
}

# "c" can be reached through both "a" and "b" with the same player to
# move, which makes it a transposition.
TRANSPOSITION_TREE = {
	"root": ("a", "b"),
	"a": ("c",),
	"b": ("c",),
	"c": ("c1", "c2"),
}

TRANSPOSITION_SCORES = {
	"c1": 1.0,
	"c2": 4.0,
}


@dataclass
class TreeState:
	node: str = "root"
	current_player: int = 1


@dataclass(frozen=True)
class TreeMove:
	destination: str


@dataclass(frozen=True)
class TreeUndo:
	node: str
	current_player: int


class TreeGame(Game[TreeState, TreeMove, int, TreeUndo]):
	"""Game played on an explicit graph of named positions.

	Players alternate between 1 and 2. A position listed in scores is
	terminal and its score is from player 1's perspective. A position
	that is neither terminal nor listed in the tree has no legal moves.

	Every make_move() and unmake_move() call is logged.
	"""

	def __init__(
		self,
		tree: Mapping[str, Sequence[str]],
		scores: Mapping[str, float],
	) -> None:
		self.tree = dict(tree)
		self.scores = dict(scores)
		self.made: list[str] = []
		self.unmade: list[str] = []

	def initial_state(self) -> TreeState:
		return TreeState()

	def current_player(self, state: TreeState) -> int:
		return state.current_player

	def legal_moves(self, state: TreeState) -> tuple[TreeMove, ...]:
		return tuple(
			TreeMove(destination)
			for destination in self.tree.get(state.node, ())
		)

	def make_move(self, state: TreeState, move: TreeMove) -> TreeUndo:
		if move not in self.legal_moves(state):
			raise ValueError(f"Illegal move: {move}")

		undo_data = TreeUndo(state.node, state.current_player)

		state.node = move.destination
		state.current_player = 3 - state.current_player
		self.made.append(move.destination)

		return undo_data

	def unmake_move(
		self,
		state: TreeState,
		move: TreeMove,
		undo_data: TreeUndo,
	) -> None:
		state.node = undo_data.node
		state.current_player = undo_data.current_player
		self.unmade.append(move.destination)

	def is_terminal(self, state: TreeState) -> bool:
		return state.node in self.scores

	def terminal_score(self, state: TreeState, player: int) -> float:
		if not self.is_terminal(state):
			raise ValueError("Position is not terminal")

		score = self.scores[state.node]

		return score if player == 1 else -score


class TableEvaluator(Evaluator[TreeState, int]):
	"""Evaluate positions by node name and record every call."""

	def __init__(
		self,
		values: Mapping[str, float] | None = None,
		default: float = 0.0,
	) -> None:
		self.values = dict(values or {})
		self.default = default
		self.calls: list[tuple[str, int]] = []

	def evaluate(self, state: TreeState, player: int) -> float:
		self.calls.append((state.node, player))

		return self.values.get(state.node, self.default)


class RaisingEvaluator(Evaluator[TreeState, int]):
	"""Evaluator that always fails."""

	def evaluate(self, state: TreeState, player: int) -> float:
		raise RuntimeError("evaluation failed")


class TreeCanonicalizer(Canonicalizer[TreeState, TreeMove]):
	"""Identify a position by its node and the player to move."""

	def position_key(self, state: TreeState) -> Hashable:
		return state.node, state.current_player


class PrefixingCanonicalizer(TreeCanonicalizer):
	"""Canonicalizer whose move transformation is easy to observe."""

	PREFIX = "canonical:"

	def canonicalize(self, state: TreeState) -> CanonicalPosition[TreeMove]:
		return CanonicalPosition(
			key=self.position_key(state),
			to_canonical_move=lambda move: TreeMove(
				self.PREFIX + move.destination
			),
			from_canonical_move=lambda move: TreeMove(
				move.destination.removeprefix(self.PREFIX)
			),
		)


class ReversingMoveOrdering(MoveOrdering[TreeState, TreeMove, int]):
	"""Reverse the move order and record the (node, player) of each call."""

	def __init__(self) -> None:
		self.calls: list[tuple[str, int]] = []

	def order_moves(
		self,
		state: TreeState,
		moves: Sequence[TreeMove],
		player: int,
	) -> Sequence[TreeMove]:
		self.calls.append((state.node, player))

		return list(reversed(moves))


class TupleMoveOrdering(MoveOrdering[TreeState, TreeMove, int]):
	"""Return moves as an immutable tuple."""

	def order_moves(
		self,
		state: TreeState,
		moves: Sequence[TreeMove],
		player: int,
	) -> Sequence[TreeMove]:
		return tuple(moves)


class InvalidMoveOrdering(MoveOrdering[TreeState, TreeMove, int]):
	"""Return an ordering that breaks the MoveOrdering contract."""

	def __init__(self, mode: str) -> None:
		self.mode = mode

	def order_moves(
		self,
		state: TreeState,
		moves: Sequence[TreeMove],
		player: int,
	) -> Sequence[TreeMove]:
		if self.mode == "add":
			return [*moves, TreeMove("invented")]

		if self.mode == "omit":
			return list(moves[:-1])

		if self.mode == "duplicate":
			return [moves[0]] * len(moves)

		if self.mode == "substitute":
			return [TreeMove("invented"), *moves[1:]]

		raise ValueError(f"Unknown mode: {self.mode}")


@pytest.fixture
def game() -> TreeGame:
	return TreeGame(SHALLOW_TREE, SHALLOW_SCORES)


@pytest.fixture
def deep_game() -> TreeGame:
	return TreeGame(DEEP_TREE, DEEP_SCORES)


@pytest.fixture
def transposition_game() -> TreeGame:
	return TreeGame(TRANSPOSITION_TREE, TRANSPOSITION_SCORES)
