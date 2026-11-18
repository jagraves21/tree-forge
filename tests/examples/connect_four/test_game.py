import random

import pytest

from connect_four.game import (
	BOARD_COLUMNS,
	BOARD_ROWS,
	CONNECT_LENGTH,
	ConnectFourGame,
	GameState,
	Move,
	Player,
	UndoData,
)


EMPTY_ROWS = ("." * BOARD_COLUMNS,) * BOARD_ROWS

# Red completes the bottom row at columns 0-3 on the seventh move.
RED_WINS_IN_SEVEN = [0, 0, 1, 1, 2, 2, 3]

# Red wins on the ninth move after wasting two moves on column 6.
RED_WINS_IN_NINE = [0, 0, 1, 1, 2, 2, 6, 6, 3]

# A full board without four in a row. Every column alternates colors and
# column 3 is inverted, which breaks every horizontal, vertical, and
# diagonal line of four.
DRAW_COLUMNS = (
	[0] * 6 + [1] * 6 + [2] * 6 + [4] + [3] * 6 + [4] * 5 + [5] * 6 + [6] * 6
)

# Cells (row, column) of a four-in-a-row, covering both edges of the board.
WINNING_LINES = [
	pytest.param([(5, 0), (5, 1), (5, 2), (5, 3)], id="horizontal-bottom-left"),
	pytest.param([(5, 3), (5, 4), (5, 5), (5, 6)], id="horizontal-bottom-right"),
	pytest.param([(0, 0), (0, 1), (0, 2), (0, 3)], id="horizontal-top-left"),
	pytest.param([(2, 2), (2, 3), (2, 4), (2, 5)], id="horizontal-middle"),
	pytest.param([(2, 0), (3, 0), (4, 0), (5, 0)], id="vertical-bottom"),
	pytest.param([(0, 6), (1, 6), (2, 6), (3, 6)], id="vertical-top"),
	pytest.param([(0, 0), (1, 1), (2, 2), (3, 3)], id="down-right-top-left"),
	pytest.param([(2, 3), (3, 4), (4, 5), (5, 6)], id="down-right-bottom-right"),
	pytest.param([(0, 6), (1, 5), (2, 4), (3, 3)], id="down-left-top-right"),
	pytest.param([(2, 3), (3, 2), (4, 1), (5, 0)], id="down-left-bottom-left"),
]


def play_columns(game: ConnectFourGame, columns) -> GameState:
	state = game.initial_state()

	for column in columns:
		game.make_move(state, Move(column))

	return state


def copy_state(state: GameState) -> GameState:
	return GameState(
		board=[row.copy() for row in state.board],
		current_player=state.current_player,
		move_count=state.move_count,
	)


def state_with_pieces(cells, player: Player) -> GameState:
	state = GameState()

	for row, column in cells:
		state.board[row][column] = player

	return state


class TestConstants:
	def test_board_dimensions_and_connect_length(self) -> None:
		assert BOARD_ROWS == 6
		assert BOARD_COLUMNS == 7
		assert CONNECT_LENGTH == 4

	def test_score_constants(self) -> None:
		assert ConnectFourGame.TERMINAL_WIN_SCORE == 1_000_000.0
		assert ConnectFourGame.TERMINAL_MOVE_PENALTY == 1.0


class TestPlayer:
	def test_opponent_of_red_is_yellow(self) -> None:
		assert Player.RED.opponent is Player.YELLOW

	def test_opponent_of_yellow_is_red(self) -> None:
		assert Player.YELLOW.opponent is Player.RED

	@pytest.mark.parametrize("player", list(Player))
	def test_opponent_of_opponent_is_the_player(self, player: Player) -> None:
		assert player.opponent.opponent is player


class TestMove:
	def test_stores_column(self) -> None:
		assert Move(4).column == 4

	def test_moves_with_equal_columns_are_equal_and_hashable(self) -> None:
		assert Move(2) == Move(2)
		assert len({Move(2), Move(2), Move(3)}) == 2

	def test_is_immutable(self) -> None:
		move = Move(1)

		with pytest.raises(AttributeError):
			move.column = 2


class TestGameState:
	def test_default_board_is_empty_with_correct_shape(self) -> None:
		state = GameState()

		assert len(state.board) == BOARD_ROWS
		assert all(len(row) == BOARD_COLUMNS for row in state.board)
		assert all(cell is None for row in state.board for cell in row)

	def test_red_moves_first_with_no_moves_played(self) -> None:
		state = GameState()

		assert state.current_player is Player.RED
		assert state.move_count == 0

	def test_default_boards_and_rows_are_independent(self) -> None:
		first = GameState()
		second = GameState()

		first.board[0][0] = Player.RED

		assert second.board[0][0] is None
		assert first.board[1][0] is None


class TestGameStateString:
	def test_renders_empty_board(self) -> None:
		blank = "│   │   │   │   │   │   │   │"
		separator = "├───┼───┼───┼───┼───┼───┼───┤"

		expected_lines = [
			"  0   1   2   3   4   5   6",
			"┌───┬───┬───┬───┬───┬───┬───┐",
		]

		for index in range(BOARD_ROWS):
			expected_lines.append(blank)

			if index < BOARD_ROWS - 1:
				expected_lines.append(separator)

		expected_lines += [
			"└───┴───┴───┴───┴───┴───┴───┘",
			"Current player: R",
			"Moves played: 0",
		]

		assert str(GameState()) == "\n".join(expected_lines)

	def test_renders_pieces_and_status(self, make_state) -> None:
		state = make_state(
			".......",
			".......",
			".......",
			".......",
			"...Y...",
			"R..R...",
			current_player=Player.YELLOW,
		)

		lines = str(state).splitlines()

		assert lines[-4] == "│ R │   │   │ R │   │   │   │"
		assert lines[-6] == "│   │   │   │ Y │   │   │   │"
		assert lines[-2] == "Current player: Y"
		assert lines[-1] == "Moves played: 3"


class TestUndoData:
	def test_stores_fields(self) -> None:
		undo = UndoData(
			row=5,
			column=3,
			player=Player.RED,
			previous_player=Player.RED,
			previous_move_count=4,
		)

		assert undo.row == 5
		assert undo.column == 3
		assert undo.player is Player.RED
		assert undo.previous_player is Player.RED
		assert undo.previous_move_count == 4

	def test_is_immutable(self) -> None:
		undo = UndoData(0, 0, Player.RED, Player.RED, 0)

		with pytest.raises(AttributeError):
			undo.row = 1


class TestInitialState:
	def test_is_empty_with_red_to_move(self, game: ConnectFourGame) -> None:
		assert game.initial_state() == GameState()

	def test_is_not_terminal(self, game: ConnectFourGame) -> None:
		assert not game.is_terminal(game.initial_state())

	def test_each_call_returns_a_new_state(self, game: ConnectFourGame) -> None:
		first = game.initial_state()
		second = game.initial_state()

		first.board[5][0] = Player.RED

		assert second.board[5][0] is None


class TestCurrentPlayer:
	def test_follows_the_state(self, game: ConnectFourGame) -> None:
		state = game.initial_state()

		assert game.current_player(state) is Player.RED

		game.make_move(state, Move(0))

		assert game.current_player(state) is Player.YELLOW


class TestLegalMoves:
	def test_initial_position_allows_every_column_in_order(
		self,
		game: ConnectFourGame,
	) -> None:
		assert game.legal_moves(game.initial_state()) == tuple(
			Move(column) for column in range(BOARD_COLUMNS)
		)

	def test_returns_a_tuple(self, game: ConnectFourGame) -> None:
		assert isinstance(game.legal_moves(game.initial_state()), tuple)

	def test_full_column_is_excluded(self, game: ConnectFourGame) -> None:
		state = play_columns(game, [3] * BOARD_ROWS)

		moves = game.legal_moves(state)

		assert Move(3) not in moves
		assert len(moves) == BOARD_COLUMNS - 1

	def test_nearly_full_column_is_still_playable(
		self,
		game: ConnectFourGame,
	) -> None:
		state = play_columns(game, [3] * (BOARD_ROWS - 1))

		assert Move(3) in game.legal_moves(state)

	def test_only_the_top_row_decides_availability(
		self,
		game: ConnectFourGame,
		make_state,
	) -> None:
		state = make_state(
			"Y.R....",
			"R.Y....",
			"Y.R....",
			"R.Y....",
			"Y.R....",
			"R.Y....",
		)

		# Columns 0 and 2 are full.
		assert game.legal_moves(state) == (
			Move(1),
			Move(3),
			Move(4),
			Move(5),
			Move(6),
		)

	def test_no_moves_after_a_win(self, game: ConnectFourGame) -> None:
		state = play_columns(game, RED_WINS_IN_SEVEN)

		assert game.legal_moves(state) == ()

	def test_no_moves_on_a_full_board(self, game: ConnectFourGame) -> None:
		state = play_columns(game, DRAW_COLUMNS)

		assert game.legal_moves(state) == ()


class TestMakeMove:
	def test_piece_falls_to_the_bottom_row(self, game: ConnectFourGame) -> None:
		state = game.initial_state()

		game.make_move(state, Move(3))

		assert state.board[BOARD_ROWS - 1][3] is Player.RED

	def test_pieces_stack_in_a_column(self, game: ConnectFourGame) -> None:
		state = play_columns(game, [3, 3, 3])

		assert state.board[5][3] is Player.RED
		assert state.board[4][3] is Player.YELLOW
		assert state.board[3][3] is Player.RED
		assert state.board[2][3] is None

	def test_only_the_chosen_cell_changes(self, game: ConnectFourGame) -> None:
		state = game.initial_state()

		game.make_move(state, Move(2))

		occupied = [
			(row, column)
			for row in range(BOARD_ROWS)
			for column in range(BOARD_COLUMNS)
			if state.board[row][column] is not None
		]

		assert occupied == [(5, 2)]

	def test_switches_player_and_counts_moves(
		self,
		game: ConnectFourGame,
	) -> None:
		state = game.initial_state()

		game.make_move(state, Move(0))

		assert state.current_player is Player.YELLOW
		assert state.move_count == 1

		game.make_move(state, Move(1))

		assert state.current_player is Player.RED
		assert state.move_count == 2

	def test_returns_undo_data(self, game: ConnectFourGame) -> None:
		state = play_columns(game, [4])

		undo = game.make_move(state, Move(4))

		assert undo == UndoData(
			row=4,
			column=4,
			player=Player.YELLOW,
			previous_player=Player.YELLOW,
			previous_move_count=1,
		)

	def test_top_cell_of_a_column_can_be_filled(
		self,
		game: ConnectFourGame,
	) -> None:
		state = play_columns(game, [3] * (BOARD_ROWS - 1))

		undo = game.make_move(state, Move(3))

		assert undo.row == 0
		assert state.board[0][3] is not None

	@pytest.mark.parametrize("column", [-1, BOARD_COLUMNS, 100])
	def test_rejects_columns_outside_the_board(
		self,
		game: ConnectFourGame,
		column: int,
	) -> None:
		with pytest.raises(ValueError, match="Illegal move"):
			game.make_move(game.initial_state(), Move(column))

	def test_rejects_a_full_column(self, game: ConnectFourGame) -> None:
		state = play_columns(game, [3] * BOARD_ROWS)

		with pytest.raises(ValueError, match="Illegal move"):
			game.make_move(state, Move(3))

	def test_rejects_moves_after_the_game_is_won(
		self,
		game: ConnectFourGame,
	) -> None:
		state = play_columns(game, RED_WINS_IN_SEVEN)

		with pytest.raises(ValueError, match="Illegal move"):
			game.make_move(state, Move(6))

	def test_rejected_move_leaves_the_state_unchanged(
		self,
		game: ConnectFourGame,
	) -> None:
		state = play_columns(game, [3] * BOARD_ROWS)
		before = copy_state(state)

		with pytest.raises(ValueError):
			game.make_move(state, Move(3))

		assert state == before


class TestUnmakeMove:
	def test_restores_board_player_and_move_count(
		self,
		game: ConnectFourGame,
	) -> None:
		state = play_columns(game, [3, 2])
		before = copy_state(state)
		move = Move(3)

		undo = game.make_move(state, move)
		game.unmake_move(state, move, undo)

		assert state == before

	def test_restores_a_sequence_in_reverse_order(
		self,
		game: ConnectFourGame,
	) -> None:
		state = game.initial_state()
		moves = [Move(3), Move(3), Move(2), Move(4), Move(3)]
		undos = [game.make_move(state, move) for move in moves]

		for move, undo in reversed(list(zip(moves, undos))):
			game.unmake_move(state, move, undo)

		assert state == GameState()

	def test_restores_a_move_that_ended_the_game(
		self,
		game: ConnectFourGame,
	) -> None:
		state = play_columns(game, RED_WINS_IN_SEVEN[:-1])
		move = Move(RED_WINS_IN_SEVEN[-1])

		undo = game.make_move(state, move)

		assert game.is_terminal(state)

		game.unmake_move(state, move, undo)

		assert not game.is_terminal(state)
		assert state.board[5][3] is None

	def test_random_playouts_unwind_to_the_initial_state(
		self,
		game: ConnectFourGame,
	) -> None:
		rng = random.Random(99)

		for _ in range(25):
			state = game.initial_state()
			history = []

			while not game.is_terminal(state):
				move = rng.choice(game.legal_moves(state))
				history.append((move, game.make_move(state, move)))

			for move, undo in reversed(history):
				game.unmake_move(state, move, undo)

			assert state == GameState()

	def test_rejects_a_move_in_a_different_column(
		self,
		game: ConnectFourGame,
	) -> None:
		state = game.initial_state()
		undo = game.make_move(state, Move(3))

		with pytest.raises(ValueError, match="does not match undo"):
			game.unmake_move(state, Move(4), undo)

	def test_rejects_an_altered_board(self, game: ConnectFourGame) -> None:
		state = game.initial_state()
		move = Move(3)
		undo = game.make_move(state, move)
		state.board[undo.row][undo.column] = Player.YELLOW

		with pytest.raises(ValueError, match="board state does not match"):
			game.unmake_move(state, move, undo)

	def test_rejects_an_unexpected_current_player(
		self,
		game: ConnectFourGame,
	) -> None:
		state = game.initial_state()
		move = Move(3)
		undo = game.make_move(state, move)
		state.current_player = Player.RED

		with pytest.raises(ValueError, match="unexpected current player"):
			game.unmake_move(state, move, undo)

	def test_rejects_undoing_a_piece_that_is_not_on_top(
		self,
		game: ConnectFourGame,
	) -> None:
		state = game.initial_state()
		move = Move(3)
		first_undo = game.make_move(state, move)
		game.make_move(state, move)
		game.make_move(state, move)

		# Yellow is to move again, as it was just after the first move.
		with pytest.raises(ValueError, match="piece is not on top"):
			game.unmake_move(state, move, first_undo)

	def test_rejects_undoing_the_same_move_twice(
		self,
		game: ConnectFourGame,
	) -> None:
		state = game.initial_state()
		move = Move(3)
		undo = game.make_move(state, move)
		game.unmake_move(state, move, undo)

		with pytest.raises(ValueError, match="board state does not match"):
			game.unmake_move(state, move, undo)


class TestWinner:
	@pytest.mark.parametrize("cells", WINNING_LINES)
	@pytest.mark.parametrize("player", list(Player))
	def test_detects_every_kind_of_line_for_each_player(
		self,
		game: ConnectFourGame,
		cells,
		player: Player,
	) -> None:
		assert game.winner(state_with_pieces(cells, player)) is player

	def test_empty_board_has_no_winner(self, game: ConnectFourGame) -> None:
		assert game.winner(game.initial_state()) is None

	def test_detects_a_winner_reached_through_play(
		self,
		game: ConnectFourGame,
	) -> None:
		assert game.winner(play_columns(game, RED_WINS_IN_SEVEN)) is (
			Player.RED
		)

	def test_detects_a_win_by_the_second_player(
		self,
		game: ConnectFourGame,
	) -> None:
		state = play_columns(game, [0, 1, 0, 1, 0, 1, 6, 1])

		assert game.winner(state) is Player.YELLOW

	def test_detects_a_line_longer_than_four(
		self,
		game: ConnectFourGame,
	) -> None:
		cells = [(5, column) for column in range(5)]

		assert game.winner(state_with_pieces(cells, Player.RED)) is Player.RED

	@pytest.mark.parametrize("rows", [
		pytest.param(
			(*EMPTY_ROWS[:5], "RRR...."),
			id="three-in-a-row",
		),
		pytest.param(
			(*EMPTY_ROWS[:5], "RRYRR.."),
			id="line-split-by-opponent",
		),
		pytest.param(
			(*EMPTY_ROWS[:5], "RR.RR.."),
			id="line-split-by-gap",
		),
		pytest.param(
			(*EMPTY_ROWS[:5], "RRR...R"),
			id="three-in-a-row-and-a-distant-piece",
		),
		pytest.param(
			(*EMPTY_ROWS[:4], "R......", "RRR...."),
			id="three-in-a-row-with-a-piece-above-it",
		),
		pytest.param(
			(*EMPTY_ROWS[:2], "R......", "R......", "R......", "Y......"),
			id="vertical-three-capped-by-opponent",
		),
		pytest.param(
			(*EMPTY_ROWS[:3], "...R...", "..R....", ".R....."),
			id="diagonal-of-three",
		),
	])
	def test_returns_none_without_four_in_a_line(
		self,
		game: ConnectFourGame,
		make_state,
		rows,
	) -> None:
		assert game.winner(make_state(*rows)) is None

	def test_horizontal_line_does_not_wrap_to_the_next_row(
		self,
		game: ConnectFourGame,
	) -> None:
		cells = [(4, 5), (4, 6), (5, 0), (5, 1)]

		assert game.winner(state_with_pieces(cells, Player.RED)) is None

	def test_mixed_colors_do_not_form_a_line(
		self,
		game: ConnectFourGame,
	) -> None:
		state = state_with_pieces([(5, 0), (5, 1)], Player.RED)

		for column in (2, 3):
			state.board[5][column] = Player.YELLOW

		assert game.winner(state) is None

	def test_drawn_board_has_no_winner(self, game: ConnectFourGame) -> None:
		assert game.winner(play_columns(game, DRAW_COLUMNS)) is None


class TestIsTerminal:
	def test_won_position_is_terminal(self, game: ConnectFourGame) -> None:
		assert game.is_terminal(play_columns(game, RED_WINS_IN_SEVEN))

	def test_full_board_without_a_winner_is_terminal(
		self,
		game: ConnectFourGame,
	) -> None:
		assert game.is_terminal(play_columns(game, DRAW_COLUMNS))

	def test_one_move_before_the_draw_is_not_terminal(
		self,
		game: ConnectFourGame,
	) -> None:
		state = play_columns(game, DRAW_COLUMNS[:-1])

		assert not game.is_terminal(state)
		assert game.legal_moves(state) == (Move(6),)

	def test_ongoing_game_is_not_terminal(self, game: ConnectFourGame) -> None:
		assert not game.is_terminal(play_columns(game, [3, 3, 2, 4]))

	def test_board_with_only_a_full_top_row_is_terminal(
		self,
		game: ConnectFourGame,
		make_state,
	) -> None:
		state = make_state(
			"RYRYRYR",
			*EMPTY_ROWS[1:],
		)

		assert game.is_terminal(state)


class TestTerminalScore:
	def test_winner_scores_the_win_value_minus_the_moves_played(
		self,
		game: ConnectFourGame,
	) -> None:
		state = play_columns(game, RED_WINS_IN_SEVEN)

		assert game.terminal_score(state, Player.RED) == 1_000_000.0 - 7

	def test_loser_scores_the_negated_value(
		self,
		game: ConnectFourGame,
	) -> None:
		state = play_columns(game, RED_WINS_IN_SEVEN)

		assert game.terminal_score(state, Player.YELLOW) == -(1_000_000.0 - 7)

	def test_scores_are_zero_sum(self, game: ConnectFourGame) -> None:
		state = play_columns(game, RED_WINS_IN_NINE)

		assert game.terminal_score(state, Player.RED) == -game.terminal_score(
			state,
			Player.YELLOW,
		)

	def test_faster_wins_score_higher(self, game: ConnectFourGame) -> None:
		fast = play_columns(game, RED_WINS_IN_SEVEN)
		slow = play_columns(game, RED_WINS_IN_NINE)

		assert game.terminal_score(slow, Player.RED) == 1_000_000.0 - 9
		assert game.terminal_score(fast, Player.RED) > game.terminal_score(
			slow,
			Player.RED,
		)

	def test_slower_losses_hurt_less(self, game: ConnectFourGame) -> None:
		fast = play_columns(game, RED_WINS_IN_SEVEN)
		slow = play_columns(game, RED_WINS_IN_NINE)

		assert game.terminal_score(slow, Player.YELLOW) > game.terminal_score(
			fast,
			Player.YELLOW,
		)

	def test_yellow_win_is_scored_for_yellow(
		self,
		game: ConnectFourGame,
	) -> None:
		state = play_columns(game, [0, 1, 0, 1, 0, 1, 6, 1])

		assert game.terminal_score(state, Player.YELLOW) == 1_000_000.0 - 8
		assert game.terminal_score(state, Player.RED) == -(1_000_000.0 - 8)

	def test_draw_scores_zero_for_both_players(
		self,
		game: ConnectFourGame,
	) -> None:
		state = play_columns(game, DRAW_COLUMNS)

		assert game.terminal_score(state, Player.RED) == 0.0
		assert game.terminal_score(state, Player.YELLOW) == 0.0

	def test_win_always_outweighs_any_evaluation(
		self,
		game: ConnectFourGame,
	) -> None:
		state = play_columns(game, RED_WINS_IN_NINE)

		assert game.terminal_score(state, Player.RED) > 999_000.0

	@pytest.mark.parametrize("columns", [
		pytest.param([], id="empty"),
		pytest.param([3, 3, 2], id="in-progress"),
		pytest.param(DRAW_COLUMNS[:-1], id="one-move-before-draw"),
	])
	def test_rejects_nonterminal_positions(
		self,
		game: ConnectFourGame,
		columns,
	) -> None:
		state = play_columns(game, columns)

		with pytest.raises(ValueError, match="nonterminal"):
			game.terminal_score(state, Player.RED)


class TestPositionKey:
	def test_initial_position(self, game: ConnectFourGame) -> None:
		key = game.position_key(game.initial_state())

		assert key == (
			((0,) * BOARD_COLUMNS,) * BOARD_ROWS,
			Player.RED.value,
			0,
		)

	def test_encodes_pieces_player_and_move_count(
		self,
		game: ConnectFourGame,
	) -> None:
		state = play_columns(game, [3, 3])

		board, player_value, move_count = game.position_key(state)

		assert board[5][3] == Player.RED.value
		assert board[4][3] == Player.YELLOW.value
		assert player_value == Player.RED.value
		assert move_count == 2

	def test_equal_positions_have_equal_keys(
		self,
		game: ConnectFourGame,
	) -> None:
		assert game.position_key(play_columns(game, [3, 4])) == (
			game.position_key(play_columns(game, [3, 4]))
		)

	def test_different_boards_have_different_keys(
		self,
		game: ConnectFourGame,
	) -> None:
		assert game.position_key(play_columns(game, [3])) != (
			game.position_key(play_columns(game, [4]))
		)

	def test_move_count_distinguishes_otherwise_equal_boards(
		self,
		game: ConnectFourGame,
	) -> None:
		first = play_columns(game, [3])
		second = copy_state(first)
		second.move_count += 2

		assert game.position_key(first) != game.position_key(second)

	def test_key_is_hashable(self, game: ConnectFourGame) -> None:
		assert hash(game.position_key(game.initial_state())) is not None

	def test_hash_is_the_hash_of_the_key(self, game: ConnectFourGame) -> None:
		state = play_columns(game, [3, 4])

		assert game.position_hash(state) == hash(game.position_key(state))


class TestDisplay:
	def test_prints_the_state(self, game: ConnectFourGame, capsys) -> None:
		state = play_columns(game, [3])

		game.display(state)

		assert capsys.readouterr().out == str(state) + "\n"
