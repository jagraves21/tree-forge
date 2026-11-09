from dataclasses import dataclass

import pytest

from tree_forge.core.interfaces import Game


@dataclass
class SimpleState:
	node: str = "root"
	current_player: int = 1


@dataclass(frozen=True)
class SimpleMove:
	destination: str


@dataclass(frozen=True)
class SimpleUndo:
	node: str
	current_player: int


class SimpleGame(Game[SimpleState, SimpleMove, int, SimpleUndo]):
	"""Small deterministic game tree for testing search algorithms."""

	TREE = {
		"root": ("a", "b"),
		"a": ("a1", "a2"),
		"b": ("b1", "b2"),
	}

	SCORES = {
		"a1": 3.0,
		"a2": 5.0,
		"b1": 2.0,
		"b2": 9.0,
	}

	def initial_state(self) -> SimpleState:
		return SimpleState()

	def current_player(self, state: SimpleState) -> int:
		return state.current_player

	def legal_moves(self, state: SimpleState) -> tuple[SimpleMove, ...]:
		return tuple(
			SimpleMove(destination)
			for destination in self.TREE.get(state.node, ())
		)

	def make_move(
		self,
		state: SimpleState,
		move: SimpleMove,
	) -> SimpleUndo:
		if move not in self.legal_moves(state):
			raise ValueError("Illegal move")

		undo = SimpleUndo(
			node=state.node,
			current_player=state.current_player,
		)

		state.node = move.destination
		state.current_player = 3 - state.current_player

		return undo

	def unmake_move(
		self,
		state: SimpleState,
		move: SimpleMove,
		undo_data: SimpleUndo,
	) -> None:
		state.node = undo_data.node
		state.current_player = undo_data.current_player

	def is_terminal(self, state: SimpleState) -> bool:
		return state.node in self.SCORES

	def terminal_score(
		self,
		state: SimpleState,
		player: int,
	) -> float:
		if not self.is_terminal(state):
			raise ValueError("Position is not terminal")

		score = self.SCORES[state.node]

		return score if player == 1 else -score

	def position_hash(self, state: SimpleState) -> int:
		return hash((state.node, state.current_player))


@pytest.fixture
def game() -> SimpleGame:
	return SimpleGame()

