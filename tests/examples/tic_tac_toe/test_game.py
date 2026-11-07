import pytest

from tic_tac_toe.game import (
	BOARD_SIZE,
	GameState,
	Move,
	Player,
	TicTacToeGame,
)


@pytest.fixture
def game():
	return TicTacToeGame()


@pytest.fixture
def state(game):
	return game.initial_state()


def play_moves(game, state, moves):
	"""Apply a sequence of moves to a game state."""
	for row, column in moves:
		game.make_move(state, Move(row, column))


DRAW_MOVES = [
	(0, 0), (0, 1),
	(0, 2), (1, 0),
	(1, 1), (2, 0),
	(1, 2), (2, 2),
	(2, 1),
]


# Initial state

def test_initial_state_is_empty(game, state):
	assert state.current_player is Player.X
	assert all(
		cell is None
		for row in state.board
		for cell in row
	)


def test_initial_state_is_not_terminal(game, state):
	assert not game.is_terminal(state)


def test_initial_state_has_nine_legal_moves(game, state):
	moves = game.legal_moves(state)

	assert len(moves) == 9
	assert set(moves) == {
		Move(row, column)
		for row in range(BOARD_SIZE)
		for column in range(BOARD_SIZE)
	}


# Move generation

def test_legal_moves_exclude_occupied_positions(game, state):
	game.make_move(state, Move(1, 1))

	moves = game.legal_moves(state)

	assert len(moves) == 8
	assert Move(1, 1) not in moves


def test_legal_moves_returns_empty_for_terminal_position(game, state):
	play_moves(
		game,
		state,
		[
			(0, 0),
			(1, 0),
			(0, 1),
			(1, 1),
			(0, 2),
		],
	)

	assert game.is_terminal(state)
	assert game.legal_moves(state) == ()


def test_illegal_move_raises_value_error(game, state):
	game.make_move(state, Move(0, 0))

	with pytest.raises(ValueError):
		game.make_move(state, Move(0, 0))


def test_move_outside_board_raises_value_error(game, state):
	with pytest.raises(ValueError):
		game.make_move(state, Move(3, 0))


# Making and undoing moves

def test_make_move_places_current_player_marker(game, state):
	game.make_move(state, Move(1, 2))

	assert state.board[1][2] is Player.X


def test_make_move_switches_current_player(game, state):
	game.make_move(state, Move(0, 0))

	assert game.current_player(state) is Player.O


def test_make_move_returns_undo_data(game, state):
	move = Move(0, 0)

	undo_data = game.make_move(state, move)

	assert undo_data.row == move.row
	assert undo_data.column == move.column
	assert undo_data.player is Player.X
	assert undo_data.previous_player is Player.X


def test_unmake_move_restores_previous_state(game, state):
	original_board = [row.copy() for row in state.board]
	original_player = state.current_player

	move = Move(1, 1)
	undo_data = game.make_move(state, move)

	game.unmake_move(state, move, undo_data)

	assert state.board == original_board
	assert state.current_player is original_player


def test_unmake_move_restores_multiple_moves(game, state):
	original_hash = game.position_hash(state)

	moves = [
		Move(0, 0),
		Move(1, 1),
		Move(0, 1),
	]
	undo_data = []

	for move in moves:
		undo_data.append(game.make_move(state, move))

	for move, undo in reversed(list(zip(moves, undo_data))):
		game.unmake_move(state, move, undo)

	assert game.position_hash(state) == original_hash


def test_unmake_move_rejects_mismatched_move(game, state):
	move = Move(0, 0)
	undo_data = game.make_move(state, move)

	with pytest.raises(ValueError):
		game.unmake_move(state, Move(0, 1), undo_data)


# Winner detection

@pytest.mark.parametrize(
	"moves, expected_winner",
	[
		# Horizontal win
		(
			[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)],
			Player.X,
		),

		# Vertical win
		(
			[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
			Player.X,
		),

		# Diagonal win
		(
			[(0, 0), (0, 1), (1, 1), (0, 2), (2, 2)],
			Player.X,
		),

		# Reverse diagonal win
		(
			[(0, 2), (0, 0), (1, 1), (0, 1), (2, 0)],
			Player.X,
		),
	],
)
def test_detects_x_wins(game, state, moves, expected_winner):
	play_moves(game, state, moves)

	assert game.winner(state) is expected_winner


def test_detects_o_win(game, state):
	play_moves(
		game,
		state,
		[
			(0, 0), (1, 0),
			(0, 1), (1, 1),
			(2, 2), (1, 2),
		],
	)

	assert game.winner(state) is Player.O


def test_winner_returns_none_when_no_winner(game, state):
	assert game.winner(state) is None


# Terminal detection

def test_detects_win_as_terminal(game, state):
	play_moves(
		game,
		state,
		[
			(0, 0), (1, 0),
			(0, 1), (1, 1),
			(0, 2),
		],
	)

	assert game.is_terminal(state)


def test_detects_full_board_draw_as_terminal(game, state):
	play_moves(game, state, DRAW_MOVES)

	assert game.is_terminal(state)
	assert game.winner(state) is None


# Terminal scoring

def test_terminal_score_is_positive_for_winner(game, state):
	play_moves(
		game,
		state,
		[
			(0, 0), (1, 0),
			(0, 1), (1, 1),
			(0, 2),
		],
	)

	assert game.terminal_score(state, Player.X) == 1.0


def test_terminal_score_is_negative_for_loser(game, state):
	play_moves(
		game,
		state,
		[
			(0, 0), (1, 0),
			(0, 1), (1, 1),
			(0, 2),
		],
	)

	assert game.terminal_score(state, Player.O) == -1.0


def test_terminal_score_is_zero_for_draw(game, state):
	play_moves(game, state, DRAW_MOVES)

	assert game.terminal_score(state, Player.X) == 0.0
	assert game.terminal_score(state, Player.O) == 0.0


def test_terminal_score_rejects_nonterminal_position(game, state):
	with pytest.raises(ValueError):
		game.terminal_score(state, Player.X)


# Position hashing

def test_identical_positions_have_identical_hashes(game):
	state_a = game.initial_state()
	state_b = game.initial_state()

	assert game.position_hash(state_a) == game.position_hash(state_b)


def test_different_positions_have_different_hashes(game):
	state_a = game.initial_state()
	state_b = game.initial_state()

	game.make_move(state_a, Move(0, 0))
	game.make_move(state_b, Move(0, 1))

	assert game.position_hash(state_a) != game.position_hash(state_b)


def test_position_hash_accounts_for_current_player(game):
	state_a = game.initial_state()
	state_b = game.initial_state()

	state_b.current_player = Player.O

	assert game.position_hash(state_a) != game.position_hash(state_b)


# Board display

def test_state_string_contains_board_coordinates(state):
	output = str(state)

	assert "0" in output
	assert "1" in output
	assert "2" in output


def test_state_string_displays_players(game, state):
	game.make_move(state, Move(1, 1))

	output = str(state)

	assert "X" in output
	assert "Current player: O" in output

