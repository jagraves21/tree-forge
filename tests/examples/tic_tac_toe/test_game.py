import pytest

from tic_tac_toe import Player, TicTacToe


@pytest.fixture
def game():
	return TicTacToe()


@pytest.fixture
def state(game):
	return game.initial_state()


def test_initial_state(game, state):
	assert state.board == [None] * 9
	assert state.current_player == Player.X
	assert state.winner is None
	assert state.move_count == 0
	assert not game.is_terminal(state)


def test_initial_legal_moves(game, state):
	assert game.legal_moves(state) == list(range(9))


def test_make_move(game, state):
	undo = game.make_move(state, 4)

	assert state.board[4] == Player.X
	assert state.current_player == Player.O
	assert state.move_count == 1
	assert state.winner is None

	game.unmake_move(state, 4, undo)

	assert state.board[4] is None
	assert state.current_player == Player.X
	assert state.move_count == 0


def test_illegal_move(game, state):
	with pytest.raises(ValueError):
		game.make_move(state, 9)

	with pytest.raises(ValueError):
		game.make_move(state, -1)


def test_occupied_position(game, state):
	game.make_move(state, 0)

	with pytest.raises(ValueError):
		game.make_move(state, 0)


def test_x_wins(game, state):
	for move in (0, 3, 1, 4, 2):
		game.make_move(state, move)

	assert state.winner == Player.X
	assert game.is_terminal(state)
	assert game.legal_moves(state) == []
	assert game.terminal_score(state, Player.X) == 1.0
	assert game.terminal_score(state, Player.O) == -1.0


def test_o_wins(game, state):
	for move in (0, 3, 1, 4, 8, 5):
		game.make_move(state, move)

	assert state.winner == Player.O
	assert game.is_terminal(state)
	assert game.terminal_score(state, Player.O) == 1.0
	assert game.terminal_score(state, Player.X) == -1.0


def test_draw(game, state):
	for move in (0, 1, 2, 4, 3, 5, 7, 6, 8):
		game.make_move(state, move)

	assert state.winner is None
	assert state.move_count == 9
	assert game.is_terminal(state)
	assert game.legal_moves(state) == []
	assert game.terminal_score(state, Player.X) == 0.0
	assert game.terminal_score(state, Player.O) == 0.0


def test_unmake_sequence(game, state):
	moves = (0, 4, 1, 5)
	undos = []

	for move in moves:
		undos.append(game.make_move(state, move))

	for move, undo in zip(reversed(moves), reversed(undos)):
		game.unmake_move(state, move, undo)

	assert state.board == [None] * 9
	assert state.current_player == Player.X
	assert state.winner is None
	assert state.move_count == 0

