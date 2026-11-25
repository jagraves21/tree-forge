"""Dependency tracking on real games.

These tests use independently computed game values to check that cached
information stays correct no matter which positions are searched, and in
which order.
"""

import random
from functools import cache
from math import inf

import pytest

from connect_four import (
	ConnectFourCanonicalizer,
	ConnectFourEvaluator,
	ConnectFourGame,
	ConnectFourMoveOrdering,
)
from connect_four import Move as ConnectFourMove
from tic_tac_toe import (
	GameState,
	Player,
	TicTacToeCanonicalizer,
	TicTacToeGame,
)
from tree_forge import Evaluator, Minimax, SearchLimits
from tree_forge.caching import TranspositionTable


SYMBOLS = {".": None, "X": Player.X, "O": Player.O}

GAME = TicTacToeGame()
EVALUATOR = Evaluator()


def copy_state(state: GameState) -> GameState:
	return GameState(
		board=[row.copy() for row in state.board],
		current_player=state.current_player,
	)


def state_from_key(key: str) -> GameState:
	"""Rebuild a position from its key, such as "X.O......"."""
	cells = key[1:]

	return GameState(
		board=[
			[SYMBOLS[cells[row * 3 + column]] for column in range(3)]
			for row in range(3)
		],
		current_player=Player[key[0]],
	)


@cache
def true_value(key: str) -> float:
	"""Game-theoretic value for the player to move, computed independently."""
	state = state_from_key(key)

	if GAME.is_terminal(state):
		return EVALUATOR.evaluate(
			state,
			state.current_player,
			GAME.outcome(state, state.current_player),
			0,
			0,
		)

	best = -inf
	canonicalizer = TicTacToeCanonicalizer()

	for move in GAME.legal_moves(state):
		undo_data = GAME.make_move(state, move)
		best = max(best, -true_value(canonicalizer.position_key(state)))
		GAME.unmake_move(state, move, undo_data)

	return best


def random_positions(seed: int, count: int) -> list[GameState]:
	rng = random.Random(seed)
	positions = []

	while len(positions) < count:
		state = GAME.initial_state()

		for _ in range(rng.randint(0, 6)):
			if GAME.is_terminal(state):
				break

			GAME.make_move(state, rng.choice(GAME.legal_moves(state)))

		if not GAME.is_terminal(state):
			positions.append(state)

	return positions


def make_search(alpha_beta: bool, tracking: bool):
	table = TranspositionTable(dependency_tracking=tracking)
	search = Minimax(
		GAME,
		limits=SearchLimits(max_depth=None),
		alpha_beta=alpha_beta,
		canonicalizer=TicTacToeCanonicalizer(),
		transposition_table=table,
	)

	return search, table


def snapshot(table: TranspositionTable) -> dict:
	return {
		key: (entry.lower, entry.upper, entry.solved)
		for key, entry in table._entries.items()
	}


def unsound_entries(table: TranspositionTable) -> list:
	"""Entries whose bounds exclude the true value of their position."""
	problems = []

	for key, entry in table._entries.items():
		value = true_value(key)

		if entry.lower is not None and entry.lower > value:
			problems.append((key, "lower bound above the value"))

		if entry.upper is not None and entry.upper < value:
			problems.append((key, "upper bound below the value"))

		if entry.solved and not (entry.lower == entry.upper == value):
			problems.append((key, "solved but not exactly the value"))

	return problems


def entries_that_could_be_improved(table: TranspositionTable) -> list:
	"""Entries that the table's own information could still improve."""
	return [
		key
		for key, entry in table._entries.items()
		if table._refreshed(key, entry) != entry
	]


CONFIGURATIONS = [
	pytest.param(False, False, id="plain"),
	pytest.param(False, True, id="plain-tracked"),
	pytest.param(True, False, id="alpha-beta"),
	pytest.param(True, True, id="alpha-beta-tracked"),
]


class TestTicTacToeKnowledgeStaysCorrect:
	@pytest.mark.parametrize("alpha_beta, tracking", CONFIGURATIONS)
	@pytest.mark.parametrize("seed", [1, 2, 3])
	def test_every_stored_bound_contains_the_true_value(
		self,
		alpha_beta: bool,
		tracking: bool,
		seed: int,
	) -> None:
		search, table = make_search(alpha_beta, tracking)

		for state in random_positions(seed, 10):
			search.search(copy_state(state))

			assert unsound_entries(table) == []

	@pytest.mark.parametrize("alpha_beta, tracking", CONFIGURATIONS)
	@pytest.mark.parametrize("seed", [1, 2, 3])
	def test_every_search_returns_the_true_value(
		self,
		alpha_beta: bool,
		tracking: bool,
		seed: int,
	) -> None:
		search, _ = make_search(alpha_beta, tracking)
		canonicalizer = TicTacToeCanonicalizer()

		for state in random_positions(seed, 10):
			result = search.search(copy_state(state))

			assert result.score == true_value(canonicalizer.position_key(state))

	@pytest.mark.parametrize("alpha_beta", [False, True])
	@pytest.mark.parametrize("seed", [1, 2, 3])
	def test_no_stored_information_is_left_unused(
		self,
		alpha_beta: bool,
		seed: int,
	) -> None:
		search, table = make_search(alpha_beta, tracking=True)

		for state in random_positions(seed, 10):
			search.search(copy_state(state))

			assert entries_that_could_be_improved(table) == []

	def test_tracking_proves_far_more_positions_from_the_same_searches(
		self,
	) -> None:
		tracked, tracked_table = make_search(True, tracking=True)
		untracked, untracked_table = make_search(True, tracking=False)

		for state in random_positions(5, 14):
			tracked.search(copy_state(state))
			untracked.search(copy_state(state))

		def proven(table: TranspositionTable) -> int:
			return sum(entry.solved for entry in table._entries.values())

		# Later searches complete positions that earlier ones had to prune.
		assert proven(tracked_table) > 2 * proven(untracked_table)

	def test_later_searches_report_the_improvements_they_cause(self) -> None:
		search, _ = make_search(True, tracking=True)

		refreshed = 0

		for state in random_positions(5, 14):
			statistics = search.search(copy_state(state)).statistics
			refreshed += statistics.transposition_table_refreshes

		assert refreshed > 0


class TestTicTacToeSearchOrder:
	@staticmethod
	def play(order, alpha_beta: bool):
		search, table = make_search(alpha_beta, tracking=True)
		canonicalizer = TicTacToeCanonicalizer()
		scores = {}

		for state in order:
			scores[canonicalizer.position_key(state)] = search.search(
				copy_state(state)
			).score

		return snapshot(table), scores

	@pytest.mark.parametrize("seed", range(6))
	def test_without_pruning_every_order_leaves_identical_knowledge(
		self,
		seed: int,
	) -> None:
		starts = random_positions(11, 12)
		first, second = starts[:], starts[:]

		random.Random(seed).shuffle(first)
		random.Random(seed + 100).shuffle(second)

		assert self.play(first, False)[0] == self.play(second, False)[0]

	@pytest.mark.parametrize("seed", range(6))
	def test_with_pruning_every_order_returns_the_same_values(
		self,
		seed: int,
	) -> None:
		starts = random_positions(11, 12)
		first, second = starts[:], starts[:]

		random.Random(seed).shuffle(first)
		random.Random(seed + 100).shuffle(second)

		assert self.play(first, True)[1] == self.play(second, True)[1]

	@pytest.mark.parametrize("seed", range(4))
	def test_with_pruning_every_order_leaves_correct_knowledge(
		self,
		seed: int,
	) -> None:
		starts = random_positions(11, 12)
		random.Random(seed).shuffle(starts)

		search, table = make_search(True, tracking=True)

		for state in starts:
			search.search(copy_state(state))

		assert unsound_entries(table) == []
		assert entries_that_could_be_improved(table) == []

	def test_positions_proven_in_any_order_have_the_true_value(self) -> None:
		starts = random_positions(11, 12)
		random.Random(3).shuffle(starts)

		search, table = make_search(True, tracking=True)

		for state in starts:
			search.search(copy_state(state))

		for key, entry in table._entries.items():
			if entry.solved:
				assert entry.score == true_value(key)


class TestConnectFourIterativeDeepening:
	@staticmethod
	def make(game: ConnectFourGame, depth: int, table, **kwargs) -> Minimax:
		return Minimax(
			game,
			evaluator=ConnectFourEvaluator(),
			limits=SearchLimits(max_depth=depth),
			canonicalizer=ConnectFourCanonicalizer(),
			transposition_table=table,
			**kwargs,
		)

	@pytest.mark.parametrize("columns", [
		pytest.param([], id="initial"),
		pytest.param([3, 3, 4, 2, 3, 5], id="mid-game"),
	])
	def test_each_deeper_search_matches_a_search_without_a_table(
		self,
		columns,
	) -> None:
		game = ConnectFourGame()
		table = TranspositionTable(dependency_tracking=True)
		state = game.initial_state()

		for column in columns:
			game.make_move(state, ConnectFourMove(column))

		for depth in (1, 2, 3, 4):
			cached = self.make(
				game,
				depth,
				table,
				alpha_beta=True,
				move_ordering=ConnectFourMoveOrdering(),
			).search(state)
			reference = Minimax(
				game,
				evaluator=ConnectFourEvaluator(),
				limits=SearchLimits(max_depth=depth),
			).search(state)

			assert cached.score == reference.score

	def test_dependency_statistics_are_measured_only_with_tracking(self) -> None:
		game = ConnectFourGame()

		tracked = self.make(
			game,
			3,
			TranspositionTable(dependency_tracking=True),
		).search(game.initial_state()).statistics
		untracked = self.make(
			game,
			3,
			TranspositionTable(dependency_tracking=False),
		).search(game.initial_state()).statistics

		assert tracked.transposition_table_refreshes is not None
		assert tracked.transposition_table_invalidations == 0
		assert untracked.transposition_table_refreshes is None
		assert untracked.transposition_table_invalidations is None
