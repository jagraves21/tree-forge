import pytest

from tree_forge.interfaces import (
	Evaluator,
	Game,
	MoveOrdering,
)


class DummyGame(Game[list[int], int, str, int]):
	"""Minimal concrete game implementation for interface tests."""

	def initial_state(self) -> list[int]:
		return []

	def current_player(self, state: list[int]) -> str:
		return "X"

	def legal_moves(self, state: list[int]) -> list[int]:
		return [0, 1, 2]

	def make_move(self, state: list[int], move: int) -> int:
		undo_data = len(state)
		state.append(move)
		return undo_data

	def unmake_move(
		self,
		state: list[int],
		move: int,
		undo_data: int,
	) -> None:
		del state[undo_data:]

	def is_terminal(self, state: list[int]) -> bool:
		return len(state) >= 3

	def terminal_score(self, state: list[int], player: str) -> float:
		return 1.0 if player == "X" else -1.0

	def position_hash(self, state: list[int]) -> int:
		return hash((tuple(state), self.current_player(state)))


class DummyEvaluator(Evaluator[list[int], str]):
	def evaluate(self, state: list[int], player: str) -> float:
		return float(len(state))


class DummyMoveOrdering(MoveOrdering[list[int], int, str]):
	def order_moves(
		self,
		state: list[int],
		moves: list[int],
		player: str,
	) -> list[int]:
		return sorted(moves, reverse=True)


class IncompleteGame(Game[list[int], int, str, int]):
	"""Intentionally incomplete implementation."""


class IncompleteEvaluator(Evaluator[list[int], str]):
	"""Intentionally incomplete implementation."""


class IncompleteMoveOrdering(MoveOrdering[list[int], int, str]):
	"""Intentionally incomplete implementation."""


class TestGameInterface:
	def test_game_is_abstract(self):
		with pytest.raises(TypeError):
			Game()

	def test_incomplete_game_cannot_be_instantiated(self):
		with pytest.raises(TypeError):
			IncompleteGame()

	def test_concrete_game_can_be_instantiated(self):
		game = DummyGame()

		assert isinstance(game, Game)

	def test_initial_state(self):
		game = DummyGame()

		assert game.initial_state() == []

	def test_current_player(self):
		game = DummyGame()

		assert game.current_player([]) == "X"

	def test_legal_moves(self):
		game = DummyGame()

		assert game.legal_moves([]) == [0, 1, 2]

	def test_make_and_unmake_move(self):
		game = DummyGame()
		state = [1, 2]

		original_state = state.copy()

		undo_data = game.make_move(state, 3)

		assert state == [1, 2, 3]

		game.unmake_move(state, 3, undo_data)

		assert state == original_state

	def test_is_terminal(self):
		game = DummyGame()

		assert not game.is_terminal([1, 2])
		assert game.is_terminal([1, 2, 3])

	def test_terminal_score(self):
		game = DummyGame()
		state = [1, 2, 3]

		assert game.terminal_score(state, "X") == 1.0
		assert game.terminal_score(state, "O") == -1.0

	def test_position_hash(self):
		game = DummyGame()

		assert game.position_hash([1, 2]) == hash(((1, 2), "X"))
		assert game.position_hash([1, 2]) == game.position_hash([1, 2])

	def test_position_hash_changes_with_position(self):
		game = DummyGame()

		assert game.position_hash([1, 2]) != game.position_hash([1, 3])


class TestEvaluatorInterface:
	def test_evaluator_is_abstract(self):
		with pytest.raises(TypeError):
			Evaluator()

	def test_incomplete_evaluator_cannot_be_instantiated(self):
		with pytest.raises(TypeError):
			IncompleteEvaluator()

	def test_concrete_evaluator_can_be_instantiated(self):
		evaluator = DummyEvaluator()

		assert isinstance(evaluator, Evaluator)

	def test_evaluate(self):
		evaluator = DummyEvaluator()

		assert evaluator.evaluate([1, 2, 3], "X") == 3.0


class TestMoveOrderingInterface:
	def test_move_ordering_is_abstract(self):
		with pytest.raises(TypeError):
			MoveOrdering()

	def test_incomplete_move_ordering_cannot_be_instantiated(self):
		with pytest.raises(TypeError):
			IncompleteMoveOrdering()

	def test_concrete_move_ordering_can_be_instantiated(self):
		ordering = DummyMoveOrdering()

		assert isinstance(ordering, MoveOrdering)

	def test_order_moves(self):
		ordering = DummyMoveOrdering()

		moves = [1, 3, 2]

		assert ordering.order_moves([], moves, "X") == [3, 2, 1]

