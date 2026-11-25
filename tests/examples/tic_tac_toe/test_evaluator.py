import pytest

from math import inf

from tic_tac_toe import BOARD_SIZE, GameState, Player, TicTacToeEvaluator, TicTacToeGame
from tree_forge import Evaluator, Outcome

from .test_canonicalizer import ALL_CELLS, reachable_states


def nonterminal_value(
	evaluator: TicTacToeEvaluator,
	state: GameState,
	player: Player,
) -> float:
	"""Return the evaluator's value of a position that is not terminal."""
	return evaluator.evaluate(state, player, None, 0, 1)


def mirror_columns(state: GameState) -> GameState:
	return GameState(
		board=[row[::-1] for row in state.board],
		current_player=state.current_player,
	)


def rotate_quarter_turn(state: GameState) -> GameState:
	board = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]

	for row, column in ALL_CELLS:
		board[column][BOARD_SIZE - 1 - row] = state.board[row][column]

	return GameState(board=board, current_player=state.current_player)


@pytest.fixture
def evaluator() -> TicTacToeEvaluator:
	return TicTacToeEvaluator()


class TestTerminalValues:
	def test_is_an_evaluator(self, evaluator: TicTacToeEvaluator) -> None:
		assert isinstance(evaluator, Evaluator)

	@pytest.mark.parametrize("outcome, expected", [
		pytest.param(Outcome.WIN, inf, id="win"),
		pytest.param(Outcome.LOSS, -inf, id="loss"),
		pytest.param(Outcome.DRAW, 0.0, id="draw"),
	])
	def test_results_keep_the_default_values(
		self,
		evaluator: TicTacToeEvaluator,
		outcome: Outcome,
		expected: float,
	) -> None:
		state = GameState()

		assert evaluator.evaluate(state, Player.X, outcome, 0, 1) == expected

	def test_a_win_outranks_every_heuristic_score(
		self,
		evaluator: TicTacToeEvaluator,
	) -> None:
		state = GameState()

		assert evaluator.evaluate(state, Player.X, Outcome.WIN, 0, 1) > (
			nonterminal_value(evaluator, state, Player.X)
		)

	def test_nonterminal_positions_use_the_heuristic(
		self,
		evaluator: TicTacToeEvaluator,
	) -> None:
		state = GameState()
		state.board[1][1] = Player.X

		assert evaluator.evaluate_non_terminal(state, Player.X, 0, 1) == (
			nonterminal_value(evaluator, state, Player.X)
		)
		assert nonterminal_value(evaluator, state, Player.X) > 0.0


class TestEvaluatorConstants:
	def test_defines_every_winning_line_once(self) -> None:
		lines = TicTacToeEvaluator.WINNING_LINES

		assert len(lines) == 8
		assert len({frozenset(line) for line in lines}) == 8
		assert all(len(line) == BOARD_SIZE for line in lines)

	def test_every_cell_is_covered_by_the_expected_number_of_lines(
		self,
	) -> None:
		counts = {cell: 0 for cell in ALL_CELLS}

		for line in TicTacToeEvaluator.WINNING_LINES:
			for cell in line:
				counts[cell] += 1

		assert counts[(1, 1)] == 4
		assert all(counts[cell] == 3 for cell in [(0, 0), (0, 2), (2, 0), (2, 2)])
		assert all(counts[cell] == 2 for cell in [(0, 1), (1, 0), (1, 2), (2, 1)])


class TestEvaluate:
	def test_empty_board_is_worth_zero_to_both_players(
		self,
		evaluator: TicTacToeEvaluator,
	) -> None:
		state = GameState()

		assert nonterminal_value(evaluator, state, Player.X) == 0.0
		assert nonterminal_value(evaluator, state, Player.O) == 0.0

	@pytest.mark.parametrize("rows, expected", [
		# One open line scores 1; the center adds 3. The center lies on
		# four lines: 4 + 3.
		pytest.param(("...", ".X.", "..."), 7.0, id="center"),
		# A corner lies on three lines: 3 + 1 for the corner.
		pytest.param(("X..", "...", "..."), 4.0, id="corner"),
		# An edge lies on two lines.
		pytest.param((".X.", "...", "..."), 2.0, id="edge"),
		# Top row with two pieces scores 10; the two columns and the
		# diagonal through (0, 0) score 1 each; the corner adds 1.
		pytest.param(("XX.", "...", "..."), 14.0, id="two-in-a-row"),
		# Row 0 is blocked. Column 0 and the diagonal favor X (+2), column 1
		# favors O (-1), and the corner adds 1.
		pytest.param(("XO.", "...", "..."), 2.0, id="blocked-row"),
		# Mirror-image threats cancel out.
		pytest.param(("X.O", "...", "..."), 0.0, id="balanced"),
	])
	def test_scores_known_positions_for_x(
		self,
		evaluator: TicTacToeEvaluator,
		make_state,
		rows,
		expected: float,
	) -> None:
		assert nonterminal_value(evaluator, make_state(*rows), Player.X) == expected

	def test_opponent_pieces_count_against_the_player(
		self,
		evaluator: TicTacToeEvaluator,
		make_state,
	) -> None:
		state = make_state("...", ".O.", "...")

		assert nonterminal_value(evaluator, state, Player.X) == -7.0
		assert nonterminal_value(evaluator, state, Player.O) == 7.0

	def test_line_with_both_players_scores_nothing(
		self,
		evaluator: TicTacToeEvaluator,
		make_state,
	) -> None:
		blocked = make_state("XO.", "...", "...")
		unblocked = make_state("X..", "...", "...")

		# Blocking row 0 removes +1 for X and the O's own lines count
		# against X, so the blocked position is strictly worse for X.
		assert nonterminal_value(evaluator, blocked, Player.X) < nonterminal_value(
			evaluator,
			unblocked,
			Player.X,
		)

	def test_ignores_the_player_to_move(
		self,
		evaluator: TicTacToeEvaluator,
		make_state,
	) -> None:
		x_to_move = make_state("XX.", ".O.", "...", current_player=Player.X)
		o_to_move = make_state("XX.", ".O.", "...", current_player=Player.O)

		assert nonterminal_value(evaluator, x_to_move, Player.X) == nonterminal_value(
			evaluator,
			o_to_move,
			Player.X,
		)

	def test_does_not_modify_the_state(
		self,
		evaluator: TicTacToeEvaluator,
		make_state,
	) -> None:
		state = make_state("XX.", ".O.", "...")
		before = make_state("XX.", ".O.", "...")

		nonterminal_value(evaluator, state, Player.X)

		assert state == before


class TestEvaluateOverAllPositions:
	@pytest.fixture(scope="class")
	@classmethod
	def open_states(self) -> list[GameState]:
		game = TicTacToeGame()

		return [
			state
			for state in reachable_states()
			if not game.is_terminal(state)
		]

	def test_scores_are_zero_sum_between_players(
		self,
		evaluator: TicTacToeEvaluator,
		open_states: list[GameState],
	) -> None:
		for state in open_states:
			assert nonterminal_value(evaluator, state, Player.X) == -nonterminal_value(
				evaluator,
				state,
				Player.O,
			)

	def test_scores_do_not_depend_on_board_orientation(
		self,
		evaluator: TicTacToeEvaluator,
		open_states: list[GameState],
	) -> None:
		for state in open_states:
			score = nonterminal_value(evaluator, state, Player.X)

			assert nonterminal_value(evaluator, mirror_columns(state), Player.X) == score
			assert nonterminal_value(
				evaluator,
				rotate_quarter_turn(state),
				Player.X,
			) == score
