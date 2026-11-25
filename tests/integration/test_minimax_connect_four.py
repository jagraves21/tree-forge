import pytest

from connect_four import (
	ConnectFourCanonicalizer,
	ConnectFourEvaluator,
	ConnectFourGame,
	ConnectFourMoveOrdering,
	GameState,
	Move,
	Player,
)
from tree_forge import Minimax, SearchLimits


WIN_SCORE = ConnectFourEvaluator.WIN_SCORE
MOVE_PENALTY = ConnectFourEvaluator.MOVE_PENALTY

# Red has the bottom row at columns 0-2 and wins by playing column 3.
RED_CAN_WIN = [0, 0, 1, 1, 2, 2]

# Yellow has the bottom row at columns 0-2 and threatens column 3, while
# Red, who is to move, has no win of its own.
YELLOW_THREATENS = [4, 0, 4, 1, 5, 2]

# Red holds the middle of the bottom row with open ends on both sides.
# Playing column 1 or 4 creates two winning threats that cannot both be
# stopped, so Red wins on its third move from here (the seventh move).
RED_FORCES_WIN = [2, 2, 3, 3]

MID_GAME = [3, 3, 4, 2, 3, 5]


def make_search(game: ConnectFourGame, depth: int, **kwargs) -> Minimax:
	return Minimax(
		game,
		evaluator=ConnectFourEvaluator(),
		limits=SearchLimits(max_depth=depth),
		**kwargs,
	)


@pytest.fixture
def game() -> ConnectFourGame:
	return ConnectFourGame()


@pytest.fixture
def play(game: ConnectFourGame):
	"""Play a sequence of columns from the initial position."""

	def factory(columns) -> GameState:
		state = game.initial_state()

		for column in columns:
			game.make_move(state, Move(column))

		return state

	return factory


class TestMinimaxConnectFourTactics:
	@pytest.mark.parametrize("alpha_beta", [False, True])
	def test_takes_an_immediate_win(
		self,
		game: ConnectFourGame,
		play,
		alpha_beta: bool,
	) -> None:
		result = make_search(game, 2, alpha_beta=alpha_beta).search(
			play(RED_CAN_WIN)
		)

		assert result.best_move == Move(3)
		assert result.score == WIN_SCORE - 7 * MOVE_PENALTY

	@pytest.mark.parametrize("alpha_beta", [False, True])
	def test_blocks_an_immediate_loss(
		self,
		game: ConnectFourGame,
		play,
		alpha_beta: bool,
	) -> None:
		result = make_search(game, 2, alpha_beta=alpha_beta).search(
			play(YELLOW_THREATENS)
		)

		assert result.best_move == Move(3)
		assert result.score > -WIN_SCORE / 2

	def test_every_other_reply_to_a_threat_loses(
		self,
		game: ConnectFourGame,
		play,
	) -> None:
		state = play(YELLOW_THREATENS)

		for move in game.legal_moves(state):
			if move == Move(3):
				continue

			undo = game.make_move(state, move)
			reply = make_search(game, 1).search(state)
			game.unmake_move(state, move, undo)

			assert reply.score == WIN_SCORE - 8 * MOVE_PENALTY

	def test_finds_a_forced_win_through_a_double_threat(
		self,
		game: ConnectFourGame,
		play,
	) -> None:
		result = make_search(game, 3, alpha_beta=True).search(
			play(RED_FORCES_WIN)
		)

		assert result.best_move in {Move(1), Move(4)}
		assert result.score == WIN_SCORE - 7 * MOVE_PENALTY

	def test_a_search_too_shallow_to_see_the_win_scores_it_heuristically(
		self,
		game: ConnectFourGame,
		play,
	) -> None:
		result = make_search(game, 2, alpha_beta=True).search(
			play(RED_FORCES_WIN)
		)

		assert abs(result.score) < WIN_SCORE / 2

	def test_a_lost_position_is_scored_as_a_loss(
		self,
		game: ConnectFourGame,
		play,
	) -> None:
		# After Red plays column 4 Yellow cannot stop both threats.
		state = play([*RED_FORCES_WIN, 4])

		result = make_search(game, 2, alpha_beta=True).search(state)

		assert game.current_player(state) is Player.YELLOW
		assert result.score == -(WIN_SCORE - 7 * MOVE_PENALTY)


class TestMinimaxConnectFourConfigurations:
	@pytest.mark.parametrize("columns", [
		pytest.param([], id="initial"),
		pytest.param(MID_GAME, id="mid-game"),
		pytest.param(RED_FORCES_WIN, id="tactical"),
	])
	@pytest.mark.parametrize("configuration", [
		pytest.param(dict(alpha_beta=True), id="alpha-beta"),
		pytest.param(
			dict(canonicalizer=ConnectFourCanonicalizer()),
			id="table",
		),
		pytest.param(
			dict(
				alpha_beta=True,
				canonicalizer=ConnectFourCanonicalizer(),
			),
			id="table-alpha-beta",
		),
	])
	def test_configurations_match_plain_minimax(
		self,
		game: ConnectFourGame,
		play,
		columns,
		configuration: dict,
	) -> None:
		state = play(columns)
		reference = make_search(game, 4).search(state)

		result = make_search(game, 4, **configuration).search(state)

		assert result.score == reference.score

	@pytest.mark.parametrize("columns", [
		pytest.param([], id="initial"),
		pytest.param(MID_GAME, id="mid-game"),
	])
	def test_move_ordering_does_not_change_the_value(
		self,
		game: ConnectFourGame,
		play,
		columns,
	) -> None:
		state = play(columns)
		reference = make_search(game, 4, alpha_beta=True).search(state)

		ordered = make_search(
			game,
			4,
			alpha_beta=True,
			canonicalizer=ConnectFourCanonicalizer(),
			move_ordering=ConnectFourMoveOrdering(),
		).search(state)

		assert ordered.score == reference.score

	def test_symmetry_lets_the_table_skip_mirrored_positions(
		self,
		game: ConnectFourGame,
		play,
	) -> None:
		state = play(MID_GAME)

		plain = make_search(game, 4).search(state)
		cached = make_search(
			game,
			4,
			canonicalizer=ConnectFourCanonicalizer(),
		).search(state)

		assert cached.statistics.transposition_table_hits > 0
		assert cached.statistics.nodes_visited < plain.statistics.nodes_visited

	def test_search_leaves_the_position_unchanged(
		self,
		game: ConnectFourGame,
		play,
	) -> None:
		state = play(MID_GAME)
		expected = play(MID_GAME)

		make_search(
			game,
			4,
			alpha_beta=True,
			canonicalizer=ConnectFourCanonicalizer(),
			move_ordering=ConnectFourMoveOrdering(),
		).search(state)

		assert state == expected


class TestConnectFourSelfPlay:
	def test_two_searching_players_finish_a_legal_game(
		self,
		game: ConnectFourGame,
	) -> None:
		search = make_search(
			game,
			3,
			alpha_beta=True,
			canonicalizer=ConnectFourCanonicalizer(),
			move_ordering=ConnectFourMoveOrdering(),
		)
		state: GameState = game.initial_state()
		moves_played = 0

		while not game.is_terminal(state):
			result = search.search(state)

			assert result.best_move in game.legal_moves(state)

			game.make_move(state, result.best_move)
			moves_played += 1

		assert state.move_count == moves_played
		assert game.is_terminal(state)
