import pytest

from tic_tac_toe import (
	GameState,
	Player,
	TicTacToeEvaluator,
)


@pytest.fixture
def evaluator() -> TicTacToeEvaluator:
	return TicTacToeEvaluator()


class TestTicTacToeEvaluator:
	def test_empty_board_scores_zero(self, evaluator):
		state = GameState()

		assert evaluator.evaluate(state, Player.X) == 0.0
		assert evaluator.evaluate(state, Player.O) == 0.0

	def test_two_in_a_row_scores_higher_than_one(
		self,
		evaluator,
	):
		state = GameState()
		state.board[0][0] = Player.X
		state.board[0][1] = Player.X

		score = evaluator.evaluate(state, Player.X)

		assert score > 0

	def test_opponent_opportunities_reduce_score(
		self,
		evaluator,
	):
		state = GameState()
		state.board[0][0] = Player.X
		state.board[0][1] = Player.X
		state.board[2][0] = Player.O
		state.board[2][1] = Player.O

		score = evaluator.evaluate(state, Player.X)

		assert score == 0.0

	def test_center_control_adds_bonus(self, evaluator):
		state = GameState()
		center_state = GameState()

		center_state.board[1][1] = Player.X

		assert (
			evaluator.evaluate(center_state, Player.X)
			> evaluator.evaluate(state, Player.X)
		)

	def test_corner_control_adds_bonus(self, evaluator):
		state = GameState()
		corner_state = GameState()

		corner_state.board[0][0] = Player.X

		assert (
			evaluator.evaluate(corner_state, Player.X)
			> evaluator.evaluate(state, Player.X)
		)

	def test_swapping_players_reverses_score(self, evaluator):
		state = GameState()
		state.board[0][0] = Player.X
		state.board[0][1] = Player.X
		state.board[1][1] = Player.O
		state.board[2][2] = Player.O

		x_score = evaluator.evaluate(state, Player.X)
		o_score = evaluator.evaluate(state, Player.O)

		assert x_score == -o_score

	def test_evaluating_from_either_perspective_is_symmetric(
		self,
		evaluator,
	):
		state = GameState()
		state.board[0][0] = Player.X
		state.board[1][1] = Player.O
		state.board[2][2] = Player.X

		assert (
			evaluator.evaluate(state, Player.X)
			== -evaluator.evaluate(state, Player.O)
		)

