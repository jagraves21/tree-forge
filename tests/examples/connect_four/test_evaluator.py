import random

import pytest

from connect_four import (
	BOARD_COLUMNS,
	BOARD_ROWS,
	ConnectFourEvaluator,
	ConnectFourGame,
	GameState,
	Move,
	Player,
)


def mirror_state(state: GameState) -> GameState:
	return GameState(
		board=[row[::-1] for row in state.board],
		current_player=state.current_player,
		move_count=state.move_count,
	)


@pytest.fixture
def evaluator() -> ConnectFourEvaluator:
	return ConnectFourEvaluator()


class TestEvaluatorConstants:
	def test_window_scores_grow_with_the_number_of_pieces(self) -> None:
		scores = ConnectFourEvaluator.LINE_SCORES

		assert scores == {1: 1.0, 2: 10.0, 3: 100.0}

	def test_center_column_bonus(self) -> None:
		assert ConnectFourEvaluator.CENTER_SCORE == 3.0


class TestEvaluate:
	def test_empty_board_is_worth_zero(
		self,
		evaluator: ConnectFourEvaluator,
		game: ConnectFourGame,
	) -> None:
		state = game.initial_state()

		assert evaluator.evaluate(state, Player.RED) == 0.0
		assert evaluator.evaluate(state, Player.YELLOW) == 0.0

	@pytest.mark.parametrize("rows, expected", [
		# Bottom center lies in 7 windows (4 horizontal, 1 vertical and
		# 1 on each diagonal) worth 1 each, plus the center bonus of 3.
		pytest.param(
			(*("." * 7,) * 5, "...R..."),
			10.0,
			id="bottom-center",
		),
		# Top center is in the same number of windows by symmetry.
		pytest.param(
			("...R...", *("." * 7,) * 5),
			10.0,
			id="top-center",
		),
		# Bottom-left corner lies in 3 windows: 1 horizontal, 1 vertical,
		# and 1 up-right diagonal. It is not in the center column.
		pytest.param(
			(*("." * 7,) * 5, "R......"),
			3.0,
			id="bottom-left-corner",
		),
		# Two neighbors: window 0-3 holds both (10), window 1-4 holds one
		# (1); verticals add 2 and the two diagonals add 2.
		pytest.param(
			(*("." * 7,) * 5, "RR....."),
			15.0,
			id="two-adjacent-pieces",
		),
		# Three neighbors: windows score 100 + 10 + 1 horizontally,
		# 3 vertically and 3 diagonally.
		pytest.param(
			(*("." * 7,) * 5, "RRR...."),
			117.0,
			id="three-adjacent-pieces",
		),
		# Window 0-3 is shared and therefore worthless. The window 1-4
		# holds only the yellow piece (-1). Every other window holds a
		# single piece, and the pieces cancel out.
		pytest.param(
			(*("." * 7,) * 5, "RY....."),
			-1.0,
			id="neighbors-of-opposite-colors",
		),
	])
	def test_scores_known_positions_for_red(
		self,
		evaluator: ConnectFourEvaluator,
		make_state,
		rows,
		expected: float,
	) -> None:
		assert evaluator.evaluate(make_state(*rows), Player.RED) == expected

	def test_yellow_pieces_count_against_red(
		self,
		evaluator: ConnectFourEvaluator,
		make_state,
	) -> None:
		state = make_state(*("." * 7,) * 5, "...Y...")

		assert evaluator.evaluate(state, Player.RED) == -10.0
		assert evaluator.evaluate(state, Player.YELLOW) == 10.0

	def test_every_center_column_piece_earns_the_bonus(
		self,
		evaluator: ConnectFourEvaluator,
		make_state,
	) -> None:
		# Two stacked pieces at the bottom of the center column:
		# - 8 horizontal windows (4 per row) worth 1 each = 8
		# - vertical windows: rows 2-5 hold both pieces (10), rows 1-4
		#   hold one (1) = 11
		# - diagonal windows: 2 for the lower piece and 4 for the upper = 6
		# - center bonus: 2 * 3 = 6
		state = make_state(*("." * 7,) * 4, "...R...", "...R...")

		assert evaluator.evaluate(state, Player.RED) == 31.0

	def test_center_column_is_worth_more_than_an_edge_column(
		self,
		evaluator: ConnectFourEvaluator,
		make_state,
	) -> None:
		center = make_state(*("." * 7,) * 5, "...R...")
		edge = make_state(*("." * 7,) * 5, "R......")

		assert evaluator.evaluate(center, Player.RED) > evaluator.evaluate(
			edge,
			Player.RED,
		)

	def test_ignores_the_player_to_move(
		self,
		evaluator: ConnectFourEvaluator,
		make_state,
	) -> None:
		rows = (*("." * 7,) * 4, "..RY...", ".RRY...")

		red_to_move = make_state(*rows, current_player=Player.RED)
		yellow_to_move = make_state(*rows, current_player=Player.YELLOW)

		assert evaluator.evaluate(red_to_move, Player.RED) == (
			evaluator.evaluate(yellow_to_move, Player.RED)
		)

	def test_does_not_modify_the_state(
		self,
		evaluator: ConnectFourEvaluator,
		make_state,
	) -> None:
		rows = (*("." * 7,) * 4, "..RY...", ".RRY...")
		state = make_state(*rows)

		evaluator.evaluate(state, Player.RED)

		assert state == make_state(*rows)


class TestEvaluateOverPlayedPositions:
	@pytest.fixture(scope="class")
	def open_positions(self) -> list[GameState]:
		"""Non-terminal positions from seeded random playouts."""
		game = ConnectFourGame()
		rng = random.Random(2024)
		positions = []

		for _ in range(40):
			state = game.initial_state()

			while not game.is_terminal(state):
				positions.append(
					GameState(
						board=[row.copy() for row in state.board],
						current_player=state.current_player,
						move_count=state.move_count,
					)
				)
				game.make_move(state, rng.choice(game.legal_moves(state)))

		return positions

	def test_scores_are_zero_sum_between_players(
		self,
		evaluator: ConnectFourEvaluator,
		open_positions: list[GameState],
	) -> None:
		assert len(open_positions) > 500

		for state in open_positions:
			assert evaluator.evaluate(state, Player.RED) == -evaluator.evaluate(
				state,
				Player.YELLOW,
			)

	def test_scores_are_unchanged_by_mirroring_the_board(
		self,
		evaluator: ConnectFourEvaluator,
		open_positions: list[GameState],
	) -> None:
		for state in open_positions:
			assert evaluator.evaluate(
				mirror_state(state),
				Player.RED,
			) == evaluator.evaluate(state, Player.RED)

	def test_scores_are_finite(
		self,
		evaluator: ConnectFourEvaluator,
		open_positions: list[GameState],
	) -> None:
		assert all(
			abs(evaluator.evaluate(state, Player.RED)) < 1_000_000.0
			for state in open_positions
		)
