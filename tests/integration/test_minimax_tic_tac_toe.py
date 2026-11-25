from math import inf

import pytest

from tic_tac_toe import (
	GameState,
	Move,
	Player,
	TicTacToeCanonicalizer,
	TicTacToeEvaluator,
	TicTacToeGame,
	TicTacToeMoveOrdering,
)
from tree_forge import Minimax, Outcome, SearchLimits


FULL_DEPTH = SearchLimits(max_depth=9)

# Each configuration must produce the same minimax value.
CONFIGURATIONS = {
	"plain": dict(),
	"alpha-beta": dict(alpha_beta=True),
	"table": dict(canonicalizer=TicTacToeCanonicalizer()),
	"table-alpha-beta": dict(
		alpha_beta=True,
		canonicalizer=TicTacToeCanonicalizer(),
	),
	"table-alpha-beta-ordering": dict(
		alpha_beta=True,
		canonicalizer=TicTacToeCanonicalizer(),
		move_ordering=TicTacToeMoveOrdering(),
	),
}

# Opening move sequences, from a nearly full game to an early one.
POSITIONS = {
	"after-two-moves": [(1, 1), (0, 0)],
	"after-three-moves": [(0, 0), (1, 1), (2, 2)],
	"after-four-moves": [(0, 0), (1, 1), (0, 1), (0, 2)],
	"after-five-moves": [(1, 1), (0, 0), (0, 2), (2, 0), (1, 0)],
}


def make_search(game: TicTacToeGame, depth: int = 9, **kwargs) -> Minimax:
	return Minimax(
		game,
		evaluator=TicTacToeEvaluator(),
		limits=SearchLimits(max_depth=depth),
		**kwargs,
	)


def state_after(game: TicTacToeGame, moves) -> GameState:
	state = game.initial_state()

	for row, column in moves:
		game.make_move(state, Move(row, column))

	return state


@pytest.fixture
def game() -> TicTacToeGame:
	return TicTacToeGame()


class TestMinimaxTicTacToeValues:
	@pytest.mark.parametrize("name", [
		"alpha-beta",
		"table-alpha-beta",
		"table-alpha-beta-ordering",
	])
	def test_empty_board_is_a_draw_with_perfect_play(
		self,
		game: TicTacToeGame,
		name: str,
	) -> None:
		result = make_search(game, **CONFIGURATIONS[name]).search(
			game.initial_state()
		)

		assert result.score == 0.0

	@pytest.mark.parametrize("position", sorted(POSITIONS))
	@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
	def test_every_configuration_matches_plain_minimax(
		self,
		game: TicTacToeGame,
		position: str,
		name: str,
	) -> None:
		state = state_after(game, POSITIONS[position])
		reference = make_search(game).search(state)

		result = make_search(game, **CONFIGURATIONS[name]).search(state)

		assert result.score == reference.score

	@pytest.mark.parametrize("position", sorted(POSITIONS))
	@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
	def test_chosen_move_achieves_the_reported_score(
		self,
		game: TicTacToeGame,
		position: str,
		name: str,
	) -> None:
		state = state_after(game, POSITIONS[position])
		root_player = game.current_player(state)

		result = make_search(game, **CONFIGURATIONS[name]).search(state)

		assert result.best_move in game.legal_moves(state)

		undo = game.make_move(state, result.best_move)
		child = make_search(game).search(state)
		game.unmake_move(state, result.best_move, undo)

		# The reply is searched from the opponent's perspective.
		assert -child.score == result.score
		assert game.current_player(state) is root_player

	def test_pruning_reduces_the_search(self, game: TicTacToeGame) -> None:
		state = state_after(game, POSITIONS["after-three-moves"])

		plain = make_search(game).search(state)
		pruned = make_search(game, alpha_beta=True).search(state)
		cached = make_search(
			game,
			alpha_beta=True,
			canonicalizer=TicTacToeCanonicalizer(),
		).search(state)

		assert pruned.statistics.nodes_visited < plain.statistics.nodes_visited
		assert cached.statistics.nodes_visited < pruned.statistics.nodes_visited
		assert cached.statistics.transposition_table_hits > 0

	def test_a_shared_table_stays_correct_across_a_whole_game(
		self,
		game: TicTacToeGame,
	) -> None:
		search = make_search(
			game,
			alpha_beta=True,
			canonicalizer=TicTacToeCanonicalizer(),
		)
		state = game.initial_state()

		while not game.is_terminal(state):
			result = search.search(state)
			pieces = sum(cell is not None for row in state.board for cell in row)

			# Unpruned search is only affordable once a few pieces are down.
			if pieces >= 2:
				assert result.score == make_search(game).search(state).score

			game.make_move(state, result.best_move)

		assert game.winner(state) is None


class TestMinimaxTicTacToeTactics:
	def test_takes_an_immediate_win_instead_of_blocking(
		self,
		game: TicTacToeGame,
	) -> None:
		# X can win at (0, 2) even though O also threatens (1, 2).
		state = state_after(game, [(0, 0), (1, 0), (0, 1), (1, 1)])

		result = make_search(game, alpha_beta=True).search(state)

		assert result.best_move == Move(0, 2)
		assert result.score == inf

	def test_blocks_an_immediate_loss(self, game: TicTacToeGame) -> None:
		# O to move; X threatens to complete the top row.
		state = state_after(game, [(0, 0), (1, 1), (0, 1)])

		result = make_search(game, alpha_beta=True).search(state)

		assert result.best_move == Move(0, 2)

	@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
	def test_finds_a_forced_win_through_a_double_threat(
		self,
		game: TicTacToeGame,
		name: str,
	) -> None:
		# X to move. Playing the center threatens both (2, 2) and (0, 2),
		# and O has no line of its own to counter with.
		state = state_after(game, [(0, 0), (0, 1), (2, 0), (1, 0)])

		result = make_search(game, **CONFIGURATIONS[name]).search(state)

		assert result.score == inf
		assert result.best_move == Move(1, 1)

	def test_sees_that_the_double_threat_cannot_be_stopped(
		self,
		game: TicTacToeGame,
	) -> None:
		# The same position one move later, with O to move and lost.
		state = state_after(
			game,
			[(0, 0), (0, 1), (2, 0), (1, 0), (1, 1)],
		)

		result = make_search(game, alpha_beta=True).search(state)

		assert game.current_player(state) is Player.O
		assert result.score == -inf


class TestSelfPlay:
	def test_two_perfect_players_draw(self, game: TicTacToeGame) -> None:
		state = game.initial_state()
		searches = {
			Player.X: make_search(
				game,
				alpha_beta=True,
				canonicalizer=TicTacToeCanonicalizer(),
				move_ordering=TicTacToeMoveOrdering(),
			),
			Player.O: make_search(
				game,
				alpha_beta=True,
				canonicalizer=TicTacToeCanonicalizer(),
				move_ordering=TicTacToeMoveOrdering(),
			),
		}

		while not game.is_terminal(state):
			result = searches[game.current_player(state)].search(state)

			assert result.best_move is not None

			game.make_move(state, result.best_move)

		assert game.winner(state) is None
		assert game.outcome(state, Player.X) is Outcome.DRAW

	def test_unpruned_players_draw_from_a_late_position(
		self,
		game: TicTacToeGame,
	) -> None:
		state = state_after(game, POSITIONS["after-three-moves"])
		search = make_search(game)

		while not game.is_terminal(state):
			result = search.search(state)
			game.make_move(state, result.best_move)

		assert game.winner(state) is None
