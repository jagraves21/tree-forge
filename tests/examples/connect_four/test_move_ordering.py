import random

import pytest

from connect_four import (
	BOARD_COLUMNS,
	ConnectFourGame,
	ConnectFourMoveOrdering,
	GameState,
	Move,
	Player,
	RandomConnectFourMoveOrdering,
)


ALL_MOVES = [Move(column) for column in range(BOARD_COLUMNS)]

# Red has the bottom row at columns 0-2 and wins by playing column 3.
RED_WINS_AT_THREE = [0, 0, 1, 1, 2, 2]

# Red has the bottom row at columns 1-3 and wins at column 0 or column 4.
RED_WINS_AT_ZERO_AND_FOUR = [1, 1, 2, 2, 3, 3]

# Yellow has the bottom row at columns 0-2 and threatens column 3 while
# Red, who is to move, has no win of its own.
YELLOW_THREATENS_THREE = [4, 0, 4, 1, 5, 2]

BLOCKING_REASON = (
	"The check for an opponent's immediate win does not depend on the "
	"candidate move, so blocking moves are not prioritized."
)


def copy_state(state: GameState) -> GameState:
	return GameState(
		board=[row.copy() for row in state.board],
		current_player=state.current_player,
		move_count=state.move_count,
	)


class IdentityShuffle(random.Random):
	"""Random source that leaves every sequence in its original order."""

	def shuffle(self, x) -> None:
		return None


class ReversingShuffle(random.Random):
	"""Random source that reverses every sequence it shuffles."""

	def shuffle(self, x) -> None:
		x.reverse()


class TestConnectFourMoveOrdering:
	@pytest.fixture
	def ordering(self, game: ConnectFourGame) -> ConnectFourMoveOrdering:
		return ConnectFourMoveOrdering(game)

	def test_stores_game(self, game: ConnectFourGame) -> None:
		assert ConnectFourMoveOrdering(game).game is game

	def test_red_prefers_columns_from_left_to_right(
		self,
		ordering: ConnectFourMoveOrdering,
		game: ConnectFourGame,
	) -> None:
		ordered = ordering.order_moves(
			game.initial_state(),
			list(reversed(ALL_MOVES)),
			Player.RED,
		)

		assert ordered == ALL_MOVES

	def test_yellow_prefers_columns_from_right_to_left(
		self,
		ordering: ConnectFourMoveOrdering,
		game: ConnectFourGame,
	) -> None:
		ordered = ordering.order_moves(
			game.initial_state(),
			ALL_MOVES,
			Player.YELLOW,
		)

		assert ordered == list(reversed(ALL_MOVES))

	def test_side_preference_follows_the_player_argument(
		self,
		ordering: ConnectFourMoveOrdering,
		game: ConnectFourGame,
	) -> None:
		state = game.initial_state()

		assert state.current_player is Player.RED
		assert ordering.order_moves(state, ALL_MOVES, Player.YELLOW)[0] == Move(6)

	def test_winning_move_comes_first(
		self,
		ordering: ConnectFourMoveOrdering,
		game: ConnectFourGame,
		play,
	) -> None:
		state = play(RED_WINS_AT_THREE)

		ordered = ordering.order_moves(state, ALL_MOVES, Player.RED)

		assert ordered == [
			Move(3),
			Move(0),
			Move(1),
			Move(2),
			Move(4),
			Move(5),
			Move(6),
		]

	def test_winning_moves_are_ordered_by_side_preference(
		self,
		ordering: ConnectFourMoveOrdering,
		play,
	) -> None:
		state = play(RED_WINS_AT_ZERO_AND_FOUR)

		red_order = ordering.order_moves(state, ALL_MOVES, Player.RED)

		assert red_order[:2] == [Move(0), Move(4)]
		assert red_order[2:] == [Move(1), Move(2), Move(3), Move(5), Move(6)]

	def test_a_winning_move_is_only_prioritized_for_the_winning_player(
		self,
		ordering: ConnectFourMoveOrdering,
		play,
	) -> None:
		state = play(RED_WINS_AT_THREE)

		yellow_order = ordering.order_moves(state, ALL_MOVES, Player.YELLOW)

		# Yellow has no immediate win, so its plain right-first preference
		# applies and column 3 is not promoted.
		assert yellow_order == list(reversed(ALL_MOVES))

	def test_orders_only_the_moves_it_is_given(
		self,
		ordering: ConnectFourMoveOrdering,
		play,
	) -> None:
		state = play(RED_WINS_AT_THREE)

		ordered = ordering.order_moves(
			state,
			[Move(6), Move(3), Move(1)],
			Player.RED,
		)

		assert ordered == [Move(3), Move(1), Move(6)]

	def test_returns_every_move_exactly_once(
		self,
		ordering: ConnectFourMoveOrdering,
		play,
	) -> None:
		state = play(RED_WINS_AT_ZERO_AND_FOUR)

		ordered = ordering.order_moves(state, ALL_MOVES, Player.RED)

		assert len(ordered) == len(ALL_MOVES)
		assert sorted(ordered, key=lambda m: m.column) == ALL_MOVES

	@pytest.mark.parametrize("columns", [
		pytest.param([], id="initial"),
		pytest.param(RED_WINS_AT_THREE, id="red-can-win"),
		pytest.param(YELLOW_THREATENS_THREE, id="yellow-threatens"),
	])
	@pytest.mark.parametrize("player", list(Player))
	def test_leaves_the_state_unchanged(
		self,
		ordering: ConnectFourMoveOrdering,
		play,
		columns,
		player: Player,
	) -> None:
		state = play(columns)
		before = copy_state(state)

		ordering.order_moves(state, ALL_MOVES, player)

		assert state == before

	def test_does_not_modify_the_given_moves(
		self,
		ordering: ConnectFourMoveOrdering,
		game: ConnectFourGame,
	) -> None:
		moves = list(reversed(ALL_MOVES))
		before = list(moves)

		ordering.order_moves(game.initial_state(), moves, Player.RED)

		assert moves == before

	def test_empty_input_gives_empty_output(
		self,
		ordering: ConnectFourMoveOrdering,
		game: ConnectFourGame,
	) -> None:
		assert ordering.order_moves(game.initial_state(), [], Player.RED) == []

	@pytest.mark.xfail(strict=True, reason=BLOCKING_REASON)
	def test_blocking_move_is_prioritized(
		self,
		ordering: ConnectFourMoveOrdering,
		play,
	) -> None:
		state = play(YELLOW_THREATENS_THREE)

		ordered = ordering.order_moves(state, ALL_MOVES, Player.RED)

		assert ordered[0] == Move(3)


class TestRandomConnectFourMoveOrdering:
	def test_stores_game_and_supplied_random_source(
		self,
		game: ConnectFourGame,
	) -> None:
		rng = random.Random(1)

		ordering = RandomConnectFourMoveOrdering(game, rng)

		assert ordering.game is game
		assert ordering.rng is rng

	def test_creates_its_own_random_source_by_default(
		self,
		game: ConnectFourGame,
	) -> None:
		first = RandomConnectFourMoveOrdering(game)
		second = RandomConnectFourMoveOrdering(game)

		assert isinstance(first.rng, random.Random)
		assert first.rng is not second.rng

	def test_shuffles_moves_of_equal_priority(
		self,
		game: ConnectFourGame,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(game, ReversingShuffle())

		ordered = ordering.order_moves(
			game.initial_state(),
			ALL_MOVES,
			Player.RED,
		)

		assert ordered == list(reversed(ALL_MOVES))

	def test_winning_moves_precede_shuffled_moves(
		self,
		game: ConnectFourGame,
		play,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(game, ReversingShuffle())

		ordered = ordering.order_moves(
			play(RED_WINS_AT_THREE),
			ALL_MOVES,
			Player.RED,
		)

		assert ordered == [
			Move(3),
			Move(6),
			Move(5),
			Move(4),
			Move(2),
			Move(1),
			Move(0),
		]

	def test_winning_moves_come_first_for_any_random_source(
		self,
		game: ConnectFourGame,
		play,
	) -> None:
		state = play(RED_WINS_AT_ZERO_AND_FOUR)

		for seed in range(20):
			ordering = RandomConnectFourMoveOrdering(
				game,
				random.Random(seed),
			)

			ordered = ordering.order_moves(state, ALL_MOVES, Player.RED)

			assert set(ordered[:2]) == {Move(0), Move(4)}

	def test_uses_the_player_argument_to_find_winning_moves(
		self,
		game: ConnectFourGame,
		play,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(game, IdentityShuffle())
		state = play(RED_WINS_AT_THREE)

		ordered = ordering.order_moves(state, ALL_MOVES, Player.YELLOW)

		assert ordered == ALL_MOVES

	def test_same_seed_gives_the_same_order(
		self,
		game: ConnectFourGame,
	) -> None:
		state = game.initial_state()

		first = RandomConnectFourMoveOrdering(game, random.Random(7))
		second = RandomConnectFourMoveOrdering(game, random.Random(7))

		assert first.order_moves(state, ALL_MOVES, Player.RED) == (
			second.order_moves(state, ALL_MOVES, Player.RED)
		)

	def test_order_varies_between_calls(
		self,
		game: ConnectFourGame,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(game, random.Random(5))
		state = game.initial_state()

		first_moves = {
			ordering.order_moves(state, ALL_MOVES, Player.RED)[0]
			for _ in range(40)
		}

		assert len(first_moves) > 1

	def test_returns_every_move_exactly_once(
		self,
		game: ConnectFourGame,
		play,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(game, random.Random(3))

		ordered = ordering.order_moves(
			play(RED_WINS_AT_THREE),
			ALL_MOVES,
			Player.RED,
		)

		assert len(ordered) == len(ALL_MOVES)
		assert sorted(ordered, key=lambda m: m.column) == ALL_MOVES

	@pytest.mark.parametrize("columns", [
		pytest.param([], id="initial"),
		pytest.param(RED_WINS_AT_THREE, id="red-can-win"),
		pytest.param(YELLOW_THREATENS_THREE, id="yellow-threatens"),
	])
	@pytest.mark.parametrize("player", list(Player))
	def test_leaves_the_state_unchanged(
		self,
		game: ConnectFourGame,
		play,
		columns,
		player: Player,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(game, random.Random(0))
		state = play(columns)
		before = copy_state(state)

		ordering.order_moves(state, ALL_MOVES, player)

		assert state == before

	def test_does_not_modify_the_given_moves(
		self,
		game: ConnectFourGame,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(game, ReversingShuffle())
		moves = list(ALL_MOVES)

		ordering.order_moves(game.initial_state(), moves, Player.RED)

		assert moves == ALL_MOVES

	def test_empty_input_gives_empty_output(
		self,
		game: ConnectFourGame,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(game, random.Random(0))

		assert ordering.order_moves(game.initial_state(), [], Player.RED) == []

	@pytest.mark.xfail(strict=True, reason=BLOCKING_REASON)
	def test_blocking_move_is_prioritized(
		self,
		game: ConnectFourGame,
		play,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(game, IdentityShuffle())
		state = play(YELLOW_THREATENS_THREE)

		ordered = ordering.order_moves(state, ALL_MOVES, Player.RED)

		assert ordered[0] == Move(3)
