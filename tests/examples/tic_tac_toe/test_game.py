import random

import pytest

from tree_forge import Outcome
from tic_tac_toe.game import (
	BOARD_SIZE,
	GameState,
	Move,
	Player,
	TicTacToeGame,
	UndoData,
)


ALL_MOVES = tuple(
	Move(row, column)
	for row in range(BOARD_SIZE)
	for column in range(BOARD_SIZE)
)

WINNING_LINES = [
	pytest.param(((0, 0), (0, 1), (0, 2)), id="top-row"),
	pytest.param(((1, 0), (1, 1), (1, 2)), id="middle-row"),
	pytest.param(((2, 0), (2, 1), (2, 2)), id="bottom-row"),
	pytest.param(((0, 0), (1, 0), (2, 0)), id="left-column"),
	pytest.param(((0, 1), (1, 1), (2, 1)), id="middle-column"),
	pytest.param(((0, 2), (1, 2), (2, 2)), id="right-column"),
	pytest.param(((0, 0), (1, 1), (2, 2)), id="main-diagonal"),
	pytest.param(((0, 2), (1, 1), (2, 0)), id="anti-diagonal"),
]

# X wins on the top row after five moves.
X_WINS_MOVES = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]

# A complete game without a winner.
DRAW_ROWS = ("XOX", "XOO", "OXX")


def play(game: TicTacToeGame, state: GameState, moves) -> None:
	for row, column in moves:
		game.make_move(state, Move(row, column))


class TestBoardSize:
	def test_board_is_three_by_three(self) -> None:
		assert BOARD_SIZE == 3


class TestPlayer:
	def test_opponent_of_x_is_o(self) -> None:
		assert Player.X.opponent is Player.O

	def test_opponent_of_o_is_x(self) -> None:
		assert Player.O.opponent is Player.X

	@pytest.mark.parametrize("player", list(Player))
	def test_opponent_of_opponent_is_the_player(self, player: Player) -> None:
		assert player.opponent.opponent is player

	def test_players_have_distinct_values(self) -> None:
		assert Player.X.value != Player.O.value


class TestMove:
	def test_stores_row_and_column(self) -> None:
		move = Move(1, 2)

		assert move.row == 1
		assert move.column == 2

	def test_moves_with_equal_coordinates_are_equal_and_hashable(self) -> None:
		assert Move(1, 2) == Move(1, 2)
		assert hash(Move(1, 2)) == hash(Move(1, 2))
		assert len({Move(1, 2), Move(1, 2), Move(2, 1)}) == 2

	def test_moves_are_ordered_by_row_then_column(self) -> None:
		assert Move(0, 2) < Move(1, 0)
		assert Move(1, 0) < Move(1, 1)
		assert sorted([Move(2, 0), Move(0, 1), Move(0, 0)]) == [
			Move(0, 0),
			Move(0, 1),
			Move(2, 0),
		]

	def test_is_immutable(self) -> None:
		move = Move(0, 0)

		with pytest.raises(AttributeError):
			move.row = 1


class TestGameState:
	def test_default_board_is_empty(self) -> None:
		state = GameState()

		assert len(state.board) == BOARD_SIZE
		assert all(len(row) == BOARD_SIZE for row in state.board)
		assert all(cell is None for row in state.board for cell in row)

	def test_x_moves_first_by_default(self) -> None:
		assert GameState().current_player is Player.X

	def test_default_boards_are_independent(self) -> None:
		first = GameState()
		second = GameState()

		first.board[0][0] = Player.X

		assert second.board[0][0] is None

	def test_default_board_rows_are_independent(self) -> None:
		state = GameState()

		state.board[0][0] = Player.X

		assert state.board[1][0] is None

	def test_states_with_equal_contents_are_equal(self) -> None:
		assert GameState() == GameState()
		assert GameState() != GameState(current_player=Player.O)


class TestGameStateString:
	def test_renders_empty_board(self) -> None:
		assert str(GameState()) == "\n".join([
			"    0   1   2",
			"  ┌───┬───┬───┐",
			"0 │   │   │   │",
			"  ├───┼───┼───┤",
			"1 │   │   │   │",
			"  ├───┼───┼───┤",
			"2 │   │   │   │",
			"  └───┴───┴───┘",
			"Current player: X",
		])

	def test_renders_pieces_in_their_cells(self, make_state) -> None:
		state = make_state(
			"O..",
			".X.",
			"..O",
			current_player=Player.O,
		)

		assert str(state) == "\n".join([
			"    0   1   2",
			"  ┌───┬───┬───┐",
			"0 │ O │   │   │",
			"  ├───┼───┼───┤",
			"1 │   │ X │   │",
			"  ├───┼───┼───┤",
			"2 │   │   │ O │",
			"  └───┴───┴───┘",
			"Current player: O",
		])


class TestUndoData:
	def test_stores_fields(self) -> None:
		undo = UndoData(
			row=1,
			column=2,
			player=Player.X,
			previous_player=Player.O,
		)

		assert undo.row == 1
		assert undo.column == 2
		assert undo.player is Player.X
		assert undo.previous_player is Player.O

	def test_is_immutable(self) -> None:
		undo = UndoData(0, 0, Player.X, Player.X)

		with pytest.raises(AttributeError):
			undo.row = 1


class TestInitialState:
	def test_initial_state_is_an_empty_board_with_x_to_move(
		self,
		game: TicTacToeGame,
	) -> None:
		state = game.initial_state()

		assert state == GameState()

	def test_each_initial_state_is_a_new_object(
		self,
		game: TicTacToeGame,
	) -> None:
		first = game.initial_state()
		second = game.initial_state()

		first.board[0][0] = Player.X

		assert second.board[0][0] is None

	def test_initial_state_is_not_terminal(self, game: TicTacToeGame) -> None:
		assert not game.is_terminal(game.initial_state())


class TestCurrentPlayer:
	def test_reports_the_player_to_move(self, game: TicTacToeGame) -> None:
		state = game.initial_state()

		assert game.current_player(state) is Player.X

		game.make_move(state, Move(0, 0))

		assert game.current_player(state) is Player.O

	def test_follows_the_state_attribute(self, game: TicTacToeGame) -> None:
		state = GameState(current_player=Player.O)

		assert game.current_player(state) is Player.O


class TestLegalMoves:
	def test_empty_board_has_nine_moves_in_row_major_order(
		self,
		game: TicTacToeGame,
	) -> None:
		assert game.legal_moves(game.initial_state()) == ALL_MOVES

	def test_returns_a_tuple(self, game: TicTacToeGame) -> None:
		assert isinstance(game.legal_moves(game.initial_state()), tuple)

	def test_excludes_occupied_cells(self, game: TicTacToeGame) -> None:
		state = game.initial_state()
		play(game, state, [(1, 1), (0, 0)])

		moves = game.legal_moves(state)

		assert len(moves) == 7
		assert Move(1, 1) not in moves
		assert Move(0, 0) not in moves

	def test_preserves_row_major_order_with_gaps(
		self,
		game: TicTacToeGame,
		make_state,
	) -> None:
		state = make_state("X.O", ".X.", "O..")

		assert game.legal_moves(state) == (
			Move(0, 1),
			Move(1, 0),
			Move(1, 2),
			Move(2, 1),
			Move(2, 2),
		)

	def test_last_empty_cell_is_the_only_move(
		self,
		game: TicTacToeGame,
		make_state,
	) -> None:
		state = make_state("XOX", "OXO", "OX.")

		assert game.legal_moves(state) == (Move(2, 2),)

	def test_no_moves_once_a_player_has_won(
		self,
		game: TicTacToeGame,
	) -> None:
		state = game.initial_state()
		play(game, state, X_WINS_MOVES)

		assert game.legal_moves(state) == ()

	def test_no_moves_on_a_full_board(
		self,
		game: TicTacToeGame,
		make_state,
	) -> None:
		assert game.legal_moves(make_state(*DRAW_ROWS)) == ()


class TestMakeMove:
	def test_places_the_current_players_marker(
		self,
		game: TicTacToeGame,
	) -> None:
		state = game.initial_state()

		game.make_move(state, Move(1, 2))

		assert state.board[1][2] is Player.X

	def test_only_the_chosen_cell_changes(self, game: TicTacToeGame) -> None:
		state = game.initial_state()

		game.make_move(state, Move(1, 2))

		occupied = [
			(row, column)
			for row in range(BOARD_SIZE)
			for column in range(BOARD_SIZE)
			if state.board[row][column] is not None
		]

		assert occupied == [(1, 2)]

	def test_switches_the_current_player(self, game: TicTacToeGame) -> None:
		state = game.initial_state()

		game.make_move(state, Move(0, 0))
		assert state.current_player is Player.O

		game.make_move(state, Move(0, 1))
		assert state.current_player is Player.X

	def test_alternates_markers(self, game: TicTacToeGame) -> None:
		state = game.initial_state()

		play(game, state, [(0, 0), (0, 1), (0, 2)])

		assert state.board[0] == [Player.X, Player.O, Player.X]

	def test_returns_undo_data(self, game: TicTacToeGame) -> None:
		state = game.initial_state()

		undo = game.make_move(state, Move(2, 1))

		assert undo == UndoData(
			row=2,
			column=1,
			player=Player.X,
			previous_player=Player.X,
		)

	def test_undo_data_records_the_mover_for_the_second_player(
		self,
		game: TicTacToeGame,
	) -> None:
		state = game.initial_state()
		game.make_move(state, Move(0, 0))

		undo = game.make_move(state, Move(1, 1))

		assert undo.player is Player.O

	@pytest.mark.parametrize("move", [
		pytest.param(Move(0, 0), id="occupied"),
		pytest.param(Move(3, 0), id="row-too-large"),
		pytest.param(Move(0, 3), id="column-too-large"),
		pytest.param(Move(-1, 0), id="negative-row"),
		pytest.param(Move(0, -1), id="negative-column"),
	])
	def test_rejects_illegal_moves(
		self,
		game: TicTacToeGame,
		move: Move,
	) -> None:
		state = game.initial_state()
		game.make_move(state, Move(0, 0))

		with pytest.raises(ValueError, match="Illegal move"):
			game.make_move(state, move)

	def test_rejected_move_leaves_the_state_unchanged(
		self,
		game: TicTacToeGame,
	) -> None:
		state = game.initial_state()
		game.make_move(state, Move(0, 0))
		before = GameState(
			board=[row.copy() for row in state.board],
			current_player=state.current_player,
		)

		with pytest.raises(ValueError):
			game.make_move(state, Move(0, 0))

		assert state == before

	def test_rejects_moves_after_the_game_is_won(
		self,
		game: TicTacToeGame,
	) -> None:
		state = game.initial_state()
		play(game, state, X_WINS_MOVES)

		with pytest.raises(ValueError, match="Illegal move"):
			game.make_move(state, Move(2, 2))


class TestUnmakeMove:
	def test_restores_the_previous_state(self, game: TicTacToeGame) -> None:
		state = game.initial_state()
		move = Move(1, 1)

		undo = game.make_move(state, move)
		game.unmake_move(state, move, undo)

		assert state == GameState()

	def test_restores_the_previous_player(self, game: TicTacToeGame) -> None:
		state = game.initial_state()
		game.make_move(state, Move(0, 0))
		move = Move(1, 1)

		undo = game.make_move(state, move)
		game.unmake_move(state, move, undo)

		assert state.current_player is Player.O
		assert state.board[0][0] is Player.X
		assert state.board[1][1] is None

	def test_restores_a_sequence_of_moves_in_reverse_order(
		self,
		game: TicTacToeGame,
	) -> None:
		state = game.initial_state()
		moves = [Move(0, 0), Move(1, 1), Move(0, 1)]
		undos = [game.make_move(state, move) for move in moves]

		for move, undo in reversed(list(zip(moves, undos))):
			game.unmake_move(state, move, undo)

		assert state == GameState()

	def test_restores_a_move_that_ended_the_game(
		self,
		game: TicTacToeGame,
	) -> None:
		state = game.initial_state()
		play(game, state, X_WINS_MOVES[:-1])

		move = Move(*X_WINS_MOVES[-1])
		undo = game.make_move(state, move)

		assert game.is_terminal(state)

		game.unmake_move(state, move, undo)

		assert not game.is_terminal(state)
		assert state.board[0][2] is None

	def test_random_playouts_unwind_to_the_initial_state(
		self,
		game: TicTacToeGame,
	) -> None:
		rng = random.Random(1234)

		for _ in range(50):
			state = game.initial_state()
			history = []

			while not game.is_terminal(state):
				move = rng.choice(game.legal_moves(state))
				history.append((move, game.make_move(state, move)))

			for move, undo in reversed(history):
				game.unmake_move(state, move, undo)

			assert state == GameState()

	def test_rejects_a_move_that_does_not_match_the_undo_data(
		self,
		game: TicTacToeGame,
	) -> None:
		state = game.initial_state()
		undo = game.make_move(state, Move(0, 0))

		with pytest.raises(ValueError, match="does not match undo"):
			game.unmake_move(state, Move(0, 1), undo)

	@pytest.mark.parametrize("move", [Move(1, 0), Move(0, 1)])
	def test_rejects_moves_differing_in_a_single_coordinate(
		self,
		game: TicTacToeGame,
		move: Move,
	) -> None:
		state = game.initial_state()
		undo = game.make_move(state, Move(0, 0))

		with pytest.raises(ValueError, match="does not match undo"):
			game.unmake_move(state, move, undo)

	def test_rejects_an_altered_board(self, game: TicTacToeGame) -> None:
		state = game.initial_state()
		move = Move(0, 0)
		undo = game.make_move(state, move)
		state.board[0][0] = Player.O

		with pytest.raises(ValueError, match="board state does not match"):
			game.unmake_move(state, move, undo)

	def test_rejects_an_unexpected_current_player(
		self,
		game: TicTacToeGame,
	) -> None:
		state = game.initial_state()
		move = Move(0, 0)
		undo = game.make_move(state, move)
		state.current_player = Player.X

		with pytest.raises(ValueError, match="unexpected current player"):
			game.unmake_move(state, move, undo)

	def test_rejects_undoing_the_same_move_twice(
		self,
		game: TicTacToeGame,
	) -> None:
		state = game.initial_state()
		move = Move(0, 0)
		undo = game.make_move(state, move)
		game.unmake_move(state, move, undo)

		with pytest.raises(ValueError, match="board state does not match"):
			game.unmake_move(state, move, undo)

	def test_failed_unmake_leaves_the_state_unchanged(
		self,
		game: TicTacToeGame,
	) -> None:
		state = game.initial_state()
		move = Move(0, 0)
		undo = game.make_move(state, move)
		state.current_player = Player.X

		with pytest.raises(ValueError):
			game.unmake_move(state, move, undo)

		assert state.board[0][0] is Player.X


class TestWinner:
	@pytest.mark.parametrize("line", WINNING_LINES)
	@pytest.mark.parametrize("player", list(Player))
	def test_detects_every_winning_line_for_each_player(
		self,
		game: TicTacToeGame,
		line,
		player: Player,
	) -> None:
		state = GameState()

		for row, column in line:
			state.board[row][column] = player

		assert game.winner(state) is player

	def test_empty_board_has_no_winner(self, game: TicTacToeGame) -> None:
		assert game.winner(game.initial_state()) is None

	def test_detects_winner_reached_through_play(
		self,
		game: TicTacToeGame,
	) -> None:
		state = game.initial_state()
		play(game, state, X_WINS_MOVES)

		assert game.winner(state) is Player.X

	def test_detects_a_win_by_the_second_player(
		self,
		game: TicTacToeGame,
	) -> None:
		state = game.initial_state()
		play(game, state, [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2), (1, 2)])

		assert game.winner(state) is Player.O

	@pytest.mark.parametrize("rows", [
		pytest.param(("XX.", "...", "..."), id="two-in-a-row"),
		pytest.param(("XOX", "...", "..."), id="blocked-row"),
		pytest.param(("XO.", "XO.", "O.."), id="no-complete-line"),
		pytest.param(("XXO", "OOX", "XOX"), id="full-board-without-line"),
		pytest.param(DRAW_ROWS, id="drawn-game"),
	])
	def test_returns_none_without_a_complete_line(
		self,
		game: TicTacToeGame,
		make_state,
		rows,
	) -> None:
		assert game.winner(make_state(*rows)) is None

	def test_line_of_a_single_player_wins_on_a_full_board(
		self,
		game: TicTacToeGame,
		make_state,
	) -> None:
		state = make_state("XOX", "OXO", "XOX")

		assert game.winner(state) is Player.X

	def test_empty_cells_do_not_form_a_winning_line(
		self,
		game: TicTacToeGame,
		make_state,
	) -> None:
		assert game.winner(make_state("...", "...", "...")) is None


class TestIsTerminal:
	def test_won_position_is_terminal(self, game: TicTacToeGame) -> None:
		state = game.initial_state()
		play(game, state, X_WINS_MOVES)

		assert game.is_terminal(state)

	def test_full_board_without_a_winner_is_terminal(
		self,
		game: TicTacToeGame,
		make_state,
	) -> None:
		assert game.is_terminal(make_state(*DRAW_ROWS))

	def test_win_on_the_last_cell_is_terminal(
		self,
		game: TicTacToeGame,
		make_state,
	) -> None:
		assert game.is_terminal(make_state("XOX", "OXO", "XOX"))

	def test_unfinished_game_is_not_terminal(
		self,
		game: TicTacToeGame,
		make_state,
	) -> None:
		assert not game.is_terminal(make_state("XO.", ".X.", "..O"))

	def test_board_missing_one_cell_is_not_terminal(
		self,
		game: TicTacToeGame,
		make_state,
	) -> None:
		assert not game.is_terminal(make_state("XOX", "OXO", "OX."))


class TestOutcome:
	@pytest.fixture
	def won_state(self, game: TicTacToeGame) -> GameState:
		state = game.initial_state()
		play(game, state, X_WINS_MOVES)

		return state

	def test_winner_has_won(
		self,
		game: TicTacToeGame,
		won_state: GameState,
	) -> None:
		assert game.outcome(won_state, Player.X) is Outcome.WIN

	def test_loser_has_lost(
		self,
		game: TicTacToeGame,
		won_state: GameState,
	) -> None:
		assert game.outcome(won_state, Player.O) is Outcome.LOSS

	def test_draw_is_a_draw_for_both_players(
		self,
		game: TicTacToeGame,
		make_state,
	) -> None:
		state = make_state(*DRAW_ROWS)

		assert game.outcome(state, Player.X) is Outcome.DRAW
		assert game.outcome(state, Player.O) is Outcome.DRAW

	def test_second_player_win_is_relative_to_o(
		self,
		game: TicTacToeGame,
		make_state,
	) -> None:
		state = make_state("OOO", "XX.", "X..")

		assert game.outcome(state, Player.O) is Outcome.WIN
		assert game.outcome(state, Player.X) is Outcome.LOSS

	def test_outcome_does_not_depend_on_the_player_to_move(
		self,
		game: TicTacToeGame,
		make_state,
	) -> None:
		state = make_state("XXX", "OO.", "...", current_player=Player.X)

		assert game.outcome(state, Player.X) is Outcome.WIN

	def test_a_win_for_one_player_is_a_loss_for_the_other(
		self,
		game: TicTacToeGame,
		won_state: GameState,
	) -> None:
		assert {
			game.outcome(won_state, Player.X),
			game.outcome(won_state, Player.O),
		} == {Outcome.WIN, Outcome.LOSS}

	def test_game_does_not_assign_numerical_values(
		self,
		game: TicTacToeGame,
	) -> None:
		assert not hasattr(game, "terminal_score")

	@pytest.mark.parametrize("rows", [
		pytest.param(("...", "...", "..."), id="empty"),
		pytest.param(("XO.", ".X.", "..O"), id="in-progress"),
		pytest.param(("XOX", "OXO", "OX."), id="one-cell-left"),
	])
	def test_rejects_nonterminal_positions(
		self,
		game: TicTacToeGame,
		make_state,
		rows,
	) -> None:
		with pytest.raises(ValueError, match="nonterminal"):
			game.outcome(make_state(*rows), Player.X)


class TestDisplay:
	def test_prints_the_state(self, game: TicTacToeGame, capsys) -> None:
		state = game.initial_state()
		game.make_move(state, Move(1, 1))

		game.display(state)

		assert capsys.readouterr().out == str(state) + "\n"
