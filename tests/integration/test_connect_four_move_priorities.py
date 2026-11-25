"""How Connect Four prefers fast wins, slow losses, and blocked threats.

The expected behavior is established by a small oracle that works without
scores: it finds forced wins and losses by playing the game out and counting
plies. That keeps these tests independent of how positions are scored.
"""

import random
from functools import cache

import pytest

from connect_four import (
	ConnectFourCanonicalizer,
	ConnectFourEvaluator,
	ConnectFourGame,
	ConnectFourMoveOrdering,
	GameState,
	Move,
	Player,
	RandomConnectFourMoveOrdering,
)
from tree_forge import Minimax, SearchLimits
from tree_forge.caching import TranspositionTable


WIN_SCORE = ConnectFourEvaluator.WIN_SCORE
MOVE_PENALTY = ConnectFourEvaluator.MOVE_PENALTY

GAME = ConnectFourGame()


# Positions, as the columns played to reach them, where the player to move
# can win both quickly and slowly: (columns, fastest column, plies to win).
FAST_AND_SLOW_WINS = [
	pytest.param(
		[0, 2, 4, 3, 0, 1, 6, 6, 1, 1, 5, 0, 3, 5],
		2,
		1,
		id="immediate-win-or-win-in-three-via-column-1",
	),
	pytest.param(
		[5, 0, 2, 4, 2, 5, 1, 6, 4, 3, 2, 5, 3, 6, 6, 1, 1, 1],
		2,
		1,
		id="immediate-win-or-three-slower-wins",
	),
	pytest.param(
		[3, 6, 4, 1, 6, 5, 1, 2, 5, 4, 2, 5, 0, 1, 3, 1],
		0,
		1,
		id="immediate-win-in-column-0-or-slower-in-column-3",
	),
]

# Positions where every move loses, but one move loses later than the rest:
# (columns, the column that delays the loss, plies until it is lost).
FORCED_LOSSES = [
	pytest.param(
		[4, 5, 4, 5, 2, 0, 6, 0, 2, 4, 4, 1, 3, 2, 0, 1, 5, 0, 3, 2, 4, 6, 5, 4, 0, 2],
		2,
		4,
		id="blocking-in-column-2-delays-the-loss",
	),
	pytest.param(
		[6, 5, 5, 2, 6, 5, 6, 0, 1, 5, 3, 2, 0, 0, 4, 6, 3, 4, 6, 5, 1, 0, 1, 1, 1, 3],
		5,
		4,
		id="blocking-in-column-5-delays-the-loss",
	),
	pytest.param(
		[6, 6, 1, 5, 0, 3, 4, 1, 1, 1, 3, 3, 2, 6, 0, 0, 6, 0, 4, 3, 6, 1, 0, 3, 0, 5],
		3,
		4,
		id="blocking-in-column-3-delays-the-loss",
	),
	pytest.param(
		[1, 2, 1, 4, 3, 3, 2, 1, 5, 2, 0, 5, 5, 6, 6, 2, 5, 3, 6, 5, 5, 3, 2, 6, 3, 2],
		1,
		4,
		id="blocking-in-column-1-delays-the-loss",
	),
]

# Yellow has the bottom row at columns 2-4 and wins at column 1 or 5. Red,
# to move, can stop only one of the two winning replies.
YELLOW_THREATENS_TWICE = [0, 2, 6, 3, 0, 4]

# Red has the bottom row at columns 2-4 and wins at column 1 or 5. Yellow,
# to move, can stop only one of the two winning replies.
RED_THREATENS_TWICE = [3, 0, 4, 6, 2]

# Red can win at column 3 while Yellow, whose turn is next, can win at
# column 6 with its three stacked pieces.
RED_WINS_FIRST = [0, 6, 1, 6, 2, 6]


def play(columns) -> GameState:
	state = GAME.initial_state()

	for column in columns:
		GAME.make_move(state, Move(column))

	return state


class ForcedResults:
	"""Forced wins and losses found by playing the game out, without scores."""

	def __init__(self) -> None:
		self.memo: dict = {}

	def solve(self, state: GameState, depth: int) -> tuple[str, int]:
		"""Return ("W" or "L", plies) if forced within depth, else ("U", 0).

		The result is for the player to move.
		"""
		key = (
			tuple(tuple(cell for cell in row) for row in state.board),
			state.current_player,
			depth,
		)

		if key in self.memo:
			return self.memo[key]

		if GAME.is_terminal(state):
			result = ("L", 0) if GAME.winner(state) is not None else ("U", 0)
		elif depth == 0:
			result = ("U", 0)
		else:
			wins, losses, unknown = [], [], False

			for move in GAME.legal_moves(state):
				child = self.after(state, move, depth)

				if child[0] == "W":
					wins.append(child[1])
				elif child[0] == "L":
					losses.append(child[1])
				else:
					unknown = True

			if wins:
				result = ("W", min(wins))
			elif unknown:
				result = ("U", 0)
			else:
				result = ("L", max(losses))

		self.memo[key] = result

		return result

	def after(self, state: GameState, move: Move, depth: int) -> tuple[str, int]:
		"""Return what a move achieves for the player who plays it."""
		undo_data = GAME.make_move(state, move)
		reply = self.solve(state, depth - 1)
		GAME.unmake_move(state, move, undo_data)

		if reply[0] == "L":
			return ("W", reply[1] + 1)

		if reply[0] == "W":
			return ("L", reply[1] + 1)

		return ("U", 0)

	@staticmethod
	def rank(result: tuple[str, int]) -> tuple[int, int]:
		"""Order results from worst to best: slow loss, fast loss, ..., fast win."""
		kind, plies = result

		if kind == "W":
			return (2, -plies)

		if kind == "U":
			return (1, 0)

		return (0, plies)


CONFIGURATIONS = {
	"plain": lambda: dict(),
	"alpha-beta": lambda: dict(alpha_beta=True),
	"alpha-beta-table": lambda: dict(
		alpha_beta=True,
		canonicalizer=ConnectFourCanonicalizer(),
	),
	"alpha-beta-table-ordering": lambda: dict(
		alpha_beta=True,
		canonicalizer=ConnectFourCanonicalizer(),
		move_ordering=ConnectFourMoveOrdering(GAME),
	),
	"alpha-beta-tracked-ordering": lambda: dict(
		alpha_beta=True,
		canonicalizer=ConnectFourCanonicalizer(),
		transposition_table=TranspositionTable(dependency_tracking=True),
		move_ordering=ConnectFourMoveOrdering(GAME),
	),
}

ORDERED_CONFIGURATIONS = [
	"alpha-beta-table-ordering",
	"alpha-beta-tracked-ordering",
]


def search(depth: int, configuration: str, **overrides):
	arguments = CONFIGURATIONS[configuration]()
	arguments.update(overrides)

	return Minimax(
		GAME,
		evaluator=ConnectFourEvaluator(),
		limits=SearchLimits(max_depth=depth),
		**arguments,
	)


class TestFasterWinsAreChosenOverSlowerOnes:
	@pytest.mark.parametrize("configuration", CONFIGURATIONS)
	@pytest.mark.parametrize("columns, fastest, plies", FAST_AND_SLOW_WINS)
	def test_picks_the_move_that_wins_soonest(
		self,
		configuration: str,
		columns,
		fastest: int,
		plies: int,
	) -> None:
		state = play(columns)

		result = search(4, configuration).search(state)

		assert result.best_move == Move(fastest)
		assert result.score == WIN_SCORE - (state.move_count + plies) * MOVE_PENALTY

	@pytest.mark.parametrize("columns, fastest, plies", FAST_AND_SLOW_WINS)
	def test_a_slower_win_scores_less_than_the_fastest(
		self,
		columns,
		fastest: int,
		plies: int,
	) -> None:
		state = play(columns)
		oracle = ForcedResults()
		scores = {}

		for move in GAME.legal_moves(state):
			if oracle.after(state, move, 4)[0] != "W":
				continue

			undo_data = GAME.make_move(state, move)
			scores[move.column] = -search(3, "plain").search(state).score
			GAME.unmake_move(state, move, undo_data)

		# Every winning move scores a win, but the fastest scores the most.
		assert len(scores) > 1
		assert max(scores, key=scores.get) == fastest
		assert min(scores.values()) > WIN_SCORE / 2


class TestSlowerLossesAreChosenOverFasterOnes:
	@pytest.mark.parametrize("configuration", CONFIGURATIONS)
	@pytest.mark.parametrize("columns, delaying, plies", FORCED_LOSSES)
	def test_picks_the_move_that_loses_latest(
		self,
		configuration: str,
		columns,
		delaying: int,
		plies: int,
	) -> None:
		state = play(columns)

		result = search(4, configuration).search(state)

		assert result.best_move == Move(delaying)
		assert result.score == -(
			WIN_SCORE - (state.move_count + plies) * MOVE_PENALTY
		)

	@pytest.mark.parametrize("columns, delaying, plies", FORCED_LOSSES)
	def test_every_move_really_loses_and_only_one_delays(
		self,
		columns,
		delaying: int,
		plies: int,
	) -> None:
		state = play(columns)
		oracle = ForcedResults()

		results = {
			move.column: oracle.after(state, move, 4)
			for move in GAME.legal_moves(state)
		}

		assert {kind for kind, _ in results.values()} == {"L"}
		assert results[delaying] == ("L", plies)
		assert sum(result == ("L", plies) for result in results.values()) == 1


@cache
def decisive_positions(seed: int) -> list:
	"""Random positions where the oracle finds a forced win or loss.

	Each is (columns played, what every move achieves according to the oracle).
	Computed once per seed because the oracle does not depend on the search.
	"""
	rng = random.Random(seed)
	oracle = ForcedResults()
	positions = []

	for _ in range(120):
		columns = []
		state = GAME.initial_state()
		target = rng.randint(14, 34)

		while state.move_count < target and not GAME.is_terminal(state):
			move = rng.choice(GAME.legal_moves(state))
			columns.append(move.column)
			GAME.make_move(state, move)

		if GAME.is_terminal(state):
			continue

		results = {
			move.column: oracle.after(state, move, 4)
			for move in GAME.legal_moves(state)
		}

		if max(results.values(), key=oracle.rank)[0] != "U":
			positions.append((columns, results))

	return positions


class TestSpeedMatchesTheOracleOnRandomPositions:
	@pytest.mark.parametrize("configuration", CONFIGURATIONS)
	@pytest.mark.parametrize("seed", [1, 2])
	def test_the_chosen_move_is_as_fast_or_slow_as_it_should_be(
		self,
		configuration: str,
		seed: int,
	) -> None:
		rank = ForcedResults.rank
		positions = decisive_positions(seed)

		assert len(positions) >= 15

		# Without pruning the search is slow, so it checks fewer positions.
		if configuration == "plain":
			positions = positions[:12]

		for columns, results in positions:
			best = max(results.values(), key=rank)
			chosen = search(4, configuration).search(play(columns)).best_move

			assert rank(results[chosen.column]) == rank(best)

	def test_the_positions_include_real_speed_choices(self) -> None:
		choices = 0

		for seed in (1, 2):
			for _, results in decisive_positions(seed):
				kinds = {kind for kind, _ in results.values()}
				plies = {plies for kind, plies in results.values() if kind != "U"}

				choices += len(plies) > 1 or len(kinds) > 1

		assert choices >= 15


class TestImmediateThreatsAreStoppedWhenDefeatIsUnavoidable:
	@staticmethod
	def values_of_every_move(state: GameState) -> dict[int, float]:
		"""Score each move for the player to move, searching each reply."""
		values = {}

		for move in GAME.legal_moves(state):
			undo_data = GAME.make_move(state, move)
			values[move.column] = -search(3, "plain").search(state).score
			GAME.unmake_move(state, move, undo_data)

		return values

	@staticmethod
	def winning_replies(state: GameState, move: Move) -> int:
		player = state.current_player
		undo_data = GAME.make_move(state, move)
		count = 0

		for reply in GAME.legal_moves(state):
			reply_undo = GAME.make_move(state, reply)
			count += GAME.winner(state) is player.opponent
			GAME.unmake_move(state, reply, reply_undo)

		GAME.unmake_move(state, move, undo_data)

		return count

	@pytest.mark.parametrize("configuration", ORDERED_CONFIGURATIONS)
	def test_red_stops_one_of_two_threats(self, configuration: str) -> None:
		state = play(YELLOW_THREATENS_TWICE)

		result = search(4, configuration).search(state)

		# Columns 1 and 5 are the two blocks; Red prefers the lower one.
		assert result.best_move == Move(1)
		assert self.winning_replies(state, result.best_move) == 1

	@pytest.mark.parametrize("configuration", ORDERED_CONFIGURATIONS)
	def test_yellow_stops_one_of_two_threats(self, configuration: str) -> None:
		state = play(RED_THREATENS_TWICE)

		result = search(4, configuration).search(state)

		# Yellow prefers the higher of the two blocking columns.
		assert result.best_move == Move(5)
		assert self.winning_replies(state, result.best_move) == 1

	@pytest.mark.parametrize("columns", [
		pytest.param(YELLOW_THREATENS_TWICE, id="red-to-move"),
		pytest.param(RED_THREATENS_TWICE, id="yellow-to-move"),
	])
	def test_the_scores_cannot_tell_the_moves_apart(self, columns) -> None:
		# This is why the game's outcome cannot express the preference: every
		# move loses on the same move, so every move is worth the same.
		values = self.values_of_every_move(play(columns))

		assert len(values) == 7
		assert len(set(values.values())) == 1
		assert max(values.values()) < -WIN_SCORE / 2

	def test_without_ordering_the_first_move_is_taken(self) -> None:
		state = play(YELLOW_THREATENS_TWICE)

		result = search(4, "alpha-beta").search(state)

		assert result.best_move == Move(0)

	@pytest.mark.parametrize("seed", range(10))
	def test_random_ordering_still_stops_a_threat(self, seed: int) -> None:
		state = play(YELLOW_THREATENS_TWICE)
		ordering = RandomConnectFourMoveOrdering(GAME, random.Random(seed))

		result = search(
			4,
			"alpha-beta-table-ordering",
			move_ordering=ordering,
		).search(state)

		assert result.best_move in {Move(1), Move(5)}

	def test_the_choice_does_not_depend_on_what_was_searched_before(
		self,
	) -> None:
		state = play(YELLOW_THREATENS_TWICE)
		table = TranspositionTable()

		earlier = search(
			4,
			"alpha-beta-table",
			transposition_table=table,
		).search(state)
		warm = search(
			4,
			"alpha-beta-table-ordering",
			transposition_table=table,
		).search(state)
		fresh = search(4, "alpha-beta-table-ordering").search(state)

		assert earlier.best_move == Move(0)
		assert warm.best_move == fresh.best_move == Move(1)

	@pytest.mark.parametrize("configuration", ORDERED_CONFIGURATIONS)
	def test_a_win_is_still_taken_instead_of_a_block(
		self,
		configuration: str,
	) -> None:
		state = play(RED_WINS_FIRST)

		result = search(4, configuration).search(state)

		# Column 6 would stop Yellow's win, but column 3 wins at once.
		assert result.best_move == Move(3)
		assert result.score == WIN_SCORE - (state.move_count + 1)

	@pytest.mark.parametrize("configuration", ORDERED_CONFIGURATIONS)
	def test_a_block_that_delays_the_loss_is_chosen_for_its_score(
		self,
		configuration: str,
	) -> None:
		state = play([4, 0, 4, 1, 5, 2])

		result = search(4, configuration).search(state)

		# Only column 3 stops Yellow from winning at once.
		assert result.best_move == Move(3)
		assert result.score > -WIN_SCORE / 2

	def test_preferring_blocks_does_not_change_the_value_of_any_position(
		self,
	) -> None:
		for columns in (
			[],
			[3, 3, 4, 2, 3, 5],
			YELLOW_THREATENS_TWICE,
			RED_THREATENS_TWICE,
			RED_WINS_FIRST,
		):
			state = play(columns)

			plain = search(4, "alpha-beta").search(state)
			ordered = search(4, "alpha-beta-table-ordering").search(state)

			assert ordered.score == plain.score
