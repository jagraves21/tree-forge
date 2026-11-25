import random

import pytest

from tree_forge import MoveOrdering
from connect_four import (
	BOARD_COLUMNS,
	ConnectFourGame,
	ConnectFourMoveOrdering,
	GameState,
	Move,
	Player,
	RandomizedConnectFourMoveOrdering,
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

# Yellow has the bottom row at columns 2-4 and wins at column 1 or 5, which
# are both open, while Red cannot win. Red can stop only one of the threats.
YELLOW_THREATENS_TWICE = [0, 2, 6, 3, 0, 4]

# Red has the bottom row at columns 2-4 and wins at column 1 or 5, and Yellow,
# who is to move, cannot win.
RED_THREATENS_TWICE = [3, 0, 4, 6, 2]


def copy_state(state: GameState) -> GameState:
	return GameState(
		board=[row.copy() for row in state.board],
		current_player=state.current_player,
		move_count=state.move_count,
	)


def winning_replies(
	game: ConnectFourGame,
	state: GameState,
	move: Move,
) -> int:
	"""Count the replies that win for the opponent after a move is played.

	The move is played for the player to move in the state, by playing it out.
	"""
	player = state.current_player
	undo = game.make_move(state, move)
	count = 0

	for reply in game.legal_moves(state):
		reply_undo = game.make_move(state, reply)
		count += game.winner(state) is player.opponent
		game.unmake_move(state, reply, reply_undo)

	game.unmake_move(state, move, undo)

	return count


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
	def ordering(self) -> ConnectFourMoveOrdering:
		return ConnectFourMoveOrdering()

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

		# Yellow cannot win, so column 3 is not a winning move for it. It
		# does stop Red's win, so it comes first, and Yellow's preference for
		# higher columns orders the rest.
		assert yellow_order == [
			Move(3),
			Move(6),
			Move(5),
			Move(4),
			Move(2),
			Move(1),
			Move(0),
		]

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


class TestConnectFourMoveOrderingThreats:
	"""Moves are ordered by how many immediate winning replies they leave."""

	@pytest.fixture
	def ordering(self) -> ConnectFourMoveOrdering:
		return ConnectFourMoveOrdering()

	def test_the_only_blocking_move_comes_first(
		self,
		ordering: ConnectFourMoveOrdering,
		play,
	) -> None:
		state = play(YELLOW_THREATENS_THREE)

		ordered = ordering.order_moves(state, ALL_MOVES, Player.RED)

		# Column 3 blocks Yellow's only winning reply. Every other move
		# leaves it, and Red prefers lower columns.
		assert ordered == [
			Move(3),
			Move(0),
			Move(1),
			Move(2),
			Move(4),
			Move(5),
			Move(6),
		]

	def test_with_two_threats_the_moves_that_block_one_come_first(
		self,
		ordering: ConnectFourMoveOrdering,
		play,
	) -> None:
		state = play(YELLOW_THREATENS_TWICE)

		ordered = ordering.order_moves(state, ALL_MOVES, Player.RED)

		# Columns 1 and 5 each leave Yellow one winning reply, any other
		# move leaves two. Red prefers lower columns within each group.
		assert ordered == [
			Move(1),
			Move(5),
			Move(0),
			Move(2),
			Move(3),
			Move(4),
			Move(6),
		]

	def test_the_same_holds_for_yellow(
		self,
		ordering: ConnectFourMoveOrdering,
		play,
	) -> None:
		state = play(RED_THREATENS_TWICE)

		ordered = ordering.order_moves(state, ALL_MOVES, Player.YELLOW)

		# Yellow prefers higher columns within each group.
		assert ordered == [
			Move(5),
			Move(1),
			Move(6),
			Move(4),
			Move(3),
			Move(2),
			Move(0),
		]

	def test_blocking_outranks_the_side_preference(
		self,
		ordering: ConnectFourMoveOrdering,
		make_state,
	) -> None:
		# Red has three in column 0 and wins by playing there.
		state = make_state(
			".......",
			".......",
			".......",
			"R......",
			"R......",
			"R......",
		)

		ordered = ordering.order_moves(state, ALL_MOVES, Player.YELLOW)

		assert ordered == [
			Move(0),
			Move(6),
			Move(5),
			Move(4),
			Move(3),
			Move(2),
			Move(1),
		]

	def test_winning_outranks_blocking(
		self,
		ordering: ConnectFourMoveOrdering,
		make_state,
	) -> None:
		# Red wins at column 3 and Yellow wins at column 6.
		state = make_state(
			".......",
			".......",
			".......",
			"......Y",
			"......Y",
			"RRR...Y",
		)

		ordered = ordering.order_moves(state, ALL_MOVES, Player.RED)

		assert ordered == [
			Move(3),
			Move(6),
			Move(0),
			Move(1),
			Move(2),
			Move(4),
			Move(5),
		]

	def test_a_move_that_opens_a_winning_cell_comes_last(
		self,
		ordering: ConnectFourMoveOrdering,
		make_state,
	) -> None:
		# Yellow has three in the second row. Its fourth cell, above the empty
		# column 3, only becomes playable once something lands in column 3.
		state = make_state(
			".......",
			".......",
			".......",
			".......",
			"YYY....",
			"RRY....",
		)

		ordered = ordering.order_moves(state, ALL_MOVES, Player.RED)

		assert ordered == [
			Move(0),
			Move(1),
			Move(2),
			Move(4),
			Move(5),
			Move(6),
			Move(3),
		]

	def test_a_piece_in_the_top_row_opens_no_cell_above_the_board(
		self,
		ordering: ConnectFourMoveOrdering,
		make_state,
	) -> None:
		# Column 0 has room for one more piece, in its top row. Yellow's
		# diagonal (0, 1), (1, 2), (2, 3) would be completed by a piece on
		# the cell above it, which does not exist.
		state = make_state(
			".Y.....",
			"R.Y....",
			"Y..Y...",
			"R......",
			"Y......",
			"R......",
		)

		ordered = ordering.order_moves(
			state,
			[Move(0), Move(2), Move(4)],
			Player.RED,
		)

		assert ordered == [Move(0), Move(2), Move(4)]

	def test_covering_a_winning_cell_that_opens_another_does_not_help(
		self,
		ordering: ConnectFourMoveOrdering,
		game: ConnectFourGame,
		make_state,
	) -> None:
		# Yellow wins at (5, 3) and, once that is covered, at (4, 3) too.
		state = make_state(
			".......",
			".......",
			".......",
			".......",
			"YYY....",
			"YYY....",
		)

		ordered = ordering.order_moves(state, ALL_MOVES, Player.RED)

		# Every move leaves Yellow one winning reply, column 3 included, so
		# only Red's preference for lower columns decides.
		assert ordered == ALL_MOVES

		# Playing it out agrees: covering (5, 3) still leaves a winning reply.
		assert winning_replies(game, state, Move(3)) == 1
		assert winning_replies(game, state, Move(6)) == 1

	@pytest.mark.parametrize("rows", [
		pytest.param(
			(
				".......",
				".......",
				".......",
				".......",
				".......",
				"RRR....",
			),
			id="horizontal",
		),
		pytest.param(
			(
				".......",
				".......",
				".......",
				"R......",
				"R......",
				"R......",
			),
			id="vertical",
		),
		pytest.param(
			(
				".......",
				".......",
				".......",
				"..RY...",
				".RYR...",
				"RYRY...",
			),
			id="rising-diagonal",
		),
		pytest.param(
			(
				".......",
				".......",
				".......",
				"...YR..",
				"...RYR.",
				"...YRYR",
			),
			id="falling-diagonal",
		),
	])
	def test_recognizes_a_win_in_every_direction(
		self,
		ordering: ConnectFourMoveOrdering,
		make_state,
		rows,
	) -> None:
		state = make_state(*rows)

		ordered = ordering.order_moves(state, ALL_MOVES, Player.RED)

		assert ordered[0] in {Move(0), Move(3)}
		assert state.board[2][ordered[0].column] is None

	def test_a_move_in_a_full_column_comes_last(
		self,
		ordering: ConnectFourMoveOrdering,
		play,
	) -> None:
		state = play([3] * 6)

		ordered = ordering.order_moves(state, ALL_MOVES, Player.RED)

		assert ordered[-1] == Move(3)

	@pytest.mark.parametrize("seed", range(5))
	def test_agrees_with_counting_replies_by_playing_them_out(
		self,
		ordering: ConnectFourMoveOrdering,
		game: ConnectFourGame,
		seed: int,
	) -> None:
		rng = random.Random(seed)
		checked = 0

		for _ in range(60):
			state = game.initial_state()

			for _ in range(rng.randint(0, 38)):
				if game.is_terminal(state):
					break

				game.make_move(state, rng.choice(game.legal_moves(state)))

			if game.is_terminal(state):
				continue

			player = state.current_player
			moves = game.legal_moves(state)

			def key(move: Move) -> tuple[int, int, int]:
				undo = game.make_move(state, move)

				if game.winner(state) is player:
					group, replies = 0, 0
				else:
					group = 1
					replies = 0

					for reply in game.legal_moves(state):
						reply_undo = game.make_move(state, reply)
						replies += game.winner(state) is player.opponent
						game.unmake_move(state, reply, reply_undo)

				game.unmake_move(state, move, undo)

				side = move.column if player is Player.RED else -move.column

				return (group, replies, side)

			assert ordering.order_moves(state, moves, player) == sorted(
				moves,
				key=key,
			)
			checked += 1

		assert checked > 20


class TestRandomizedConnectFourMoveOrdering:
	def test_extends_the_tactical_ordering(self) -> None:
		assert issubclass(
			RandomizedConnectFourMoveOrdering,
			ConnectFourMoveOrdering,
		)

	def test_stores_supplied_random_source(self) -> None:
		rng = random.Random(1)

		ordering = RandomizedConnectFourMoveOrdering(rng)

		assert ordering.rng is rng

	def test_creates_its_own_random_source_by_default(self) -> None:
		first = RandomizedConnectFourMoveOrdering()
		second = RandomizedConnectFourMoveOrdering()

		assert isinstance(first.rng, random.Random)
		assert first.rng is not second.rng

	def test_shuffles_moves_of_equal_priority(
		self,
		game: ConnectFourGame,
	) -> None:
		ordering = RandomizedConnectFourMoveOrdering(ReversingShuffle())

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
		ordering = RandomizedConnectFourMoveOrdering(ReversingShuffle())

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
		play,
	) -> None:
		state = play(RED_WINS_AT_ZERO_AND_FOUR)

		for seed in range(20):
			ordering = RandomizedConnectFourMoveOrdering(
				random.Random(seed),
			)

			ordered = ordering.order_moves(state, ALL_MOVES, Player.RED)

			assert set(ordered[:2]) == {Move(0), Move(4)}

	def test_uses_the_player_argument_to_find_threats(
		self,
		play,
	) -> None:
		ordering = RandomizedConnectFourMoveOrdering(IdentityShuffle())
		state = play(RED_WINS_AT_THREE)

		ordered = ordering.order_moves(state, ALL_MOVES, Player.YELLOW)

		# Yellow cannot win, but it can stop Red from winning in column 3.
		assert ordered == [
			Move(3),
			Move(0),
			Move(1),
			Move(2),
			Move(4),
			Move(5),
			Move(6),
		]

	def test_moves_that_leave_fewer_threats_come_first(
		self,
		play,
	) -> None:
		ordering = RandomizedConnectFourMoveOrdering(ReversingShuffle())
		state = play(YELLOW_THREATENS_TWICE)

		ordered = ordering.order_moves(state, ALL_MOVES, Player.RED)

		# Columns 1 and 5 leave one threat, the rest two. Each group is
		# shuffled, here by reversing it.
		assert ordered == [
			Move(5),
			Move(1),
			Move(6),
			Move(4),
			Move(3),
			Move(2),
			Move(0),
		]

	def test_blocking_moves_come_first_for_any_random_source(
		self,
		play,
	) -> None:
		state = play(YELLOW_THREATENS_TWICE)

		for seed in range(20):
			ordering = RandomizedConnectFourMoveOrdering(
				random.Random(seed),
			)

			ordered = ordering.order_moves(state, ALL_MOVES, Player.RED)

			assert set(ordered[:2]) == {Move(1), Move(5)}

	def test_same_seed_gives_the_same_order(
		self,
		game: ConnectFourGame,
	) -> None:
		state = game.initial_state()

		first = RandomizedConnectFourMoveOrdering(random.Random(7))
		second = RandomizedConnectFourMoveOrdering(random.Random(7))

		assert first.order_moves(state, ALL_MOVES, Player.RED) == (
			second.order_moves(state, ALL_MOVES, Player.RED)
		)

	def test_order_varies_between_calls(
		self,
		game: ConnectFourGame,
	) -> None:
		ordering = RandomizedConnectFourMoveOrdering(random.Random(5))
		state = game.initial_state()

		first_moves = {
			ordering.order_moves(state, ALL_MOVES, Player.RED)[0]
			for _ in range(40)
		}

		assert len(first_moves) > 1

	def test_returns_every_move_exactly_once(
		self,
		play,
	) -> None:
		ordering = RandomizedConnectFourMoveOrdering(random.Random(3))

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
		play,
		columns,
		player: Player,
	) -> None:
		ordering = RandomizedConnectFourMoveOrdering(random.Random(0))
		state = play(columns)
		before = copy_state(state)

		ordering.order_moves(state, ALL_MOVES, player)

		assert state == before

	def test_does_not_modify_the_given_moves(
		self,
		game: ConnectFourGame,
	) -> None:
		ordering = RandomizedConnectFourMoveOrdering(ReversingShuffle())
		moves = list(ALL_MOVES)

		ordering.order_moves(game.initial_state(), moves, Player.RED)

		assert moves == ALL_MOVES

	def test_empty_input_gives_empty_output(
		self,
		game: ConnectFourGame,
	) -> None:
		ordering = RandomizedConnectFourMoveOrdering(random.Random(0))

		assert ordering.order_moves(game.initial_state(), [], Player.RED) == []


class TestRandomConnectFourMoveOrdering:
	def test_does_not_extend_the_tactical_ordering(self) -> None:
		assert not issubclass(
			RandomConnectFourMoveOrdering,
			ConnectFourMoveOrdering,
		)

	def test_does_not_extend_the_randomized_ordering(self) -> None:
		assert not issubclass(
			RandomConnectFourMoveOrdering,
			RandomizedConnectFourMoveOrdering,
		)

	def test_is_a_move_ordering(self) -> None:
		assert isinstance(RandomConnectFourMoveOrdering(), MoveOrdering)

	def test_stores_supplied_random_source(self) -> None:
		rng = random.Random(1)

		ordering = RandomConnectFourMoveOrdering(rng)

		assert ordering.rng is rng

	def test_creates_its_own_random_source_by_default(self) -> None:
		first = RandomConnectFourMoveOrdering()
		second = RandomConnectFourMoveOrdering()

		assert isinstance(first.rng, random.Random)
		assert first.rng is not second.rng

	def test_shuffles_all_moves_with_the_random_source(
		self,
		game: ConnectFourGame,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(ReversingShuffle())

		ordered = ordering.order_moves(
			game.initial_state(),
			ALL_MOVES,
			Player.RED,
		)

		assert ordered == list(reversed(ALL_MOVES))

	def test_ignores_winning_moves(
		self,
		play,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(ReversingShuffle())

		ordered = ordering.order_moves(
			play(RED_WINS_AT_THREE),
			ALL_MOVES,
			Player.RED,
		)

		# The winning move in column 3 gets no priority.
		assert ordered == list(reversed(ALL_MOVES))

	def test_ignores_threats_and_side_preference(
		self,
		play,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(IdentityShuffle())

		for columns in (RED_WINS_AT_THREE, YELLOW_THREATENS_TWICE):
			for player in Player:
				ordered = ordering.order_moves(
					play(columns),
					ALL_MOVES,
					player,
				)

				assert ordered == ALL_MOVES

	def test_first_move_is_not_always_a_winning_move(
		self,
		play,
	) -> None:
		state = play(RED_WINS_AT_THREE)

		first_moves = {
			RandomConnectFourMoveOrdering(random.Random(seed))
			.order_moves(state, ALL_MOVES, Player.RED)[0]
			for seed in range(40)
		}

		assert len(first_moves) > 1

	def test_same_seed_gives_the_same_order(
		self,
		game: ConnectFourGame,
	) -> None:
		state = game.initial_state()

		first = RandomConnectFourMoveOrdering(random.Random(7))
		second = RandomConnectFourMoveOrdering(random.Random(7))

		assert first.order_moves(state, ALL_MOVES, Player.RED) == (
			second.order_moves(state, ALL_MOVES, Player.RED)
		)

	def test_order_varies_between_calls(
		self,
		game: ConnectFourGame,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(random.Random(5))
		state = game.initial_state()

		first_moves = {
			ordering.order_moves(state, ALL_MOVES, Player.RED)[0]
			for _ in range(40)
		}

		assert len(first_moves) > 1

	def test_returns_every_move_exactly_once(
		self,
		play,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(random.Random(3))

		ordered = ordering.order_moves(
			play(RED_WINS_AT_THREE),
			ALL_MOVES,
			Player.RED,
		)

		assert len(ordered) == len(ALL_MOVES)
		assert sorted(ordered, key=lambda m: m.column) == ALL_MOVES

	def test_orders_only_the_moves_it_is_given(
		self,
		game: ConnectFourGame,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(random.Random(0))
		moves = [Move(6), Move(3), Move(1)]

		ordered = ordering.order_moves(
			game.initial_state(),
			moves,
			Player.RED,
		)

		assert sorted(ordered, key=lambda m: m.column) == [
			Move(1),
			Move(3),
			Move(6),
		]

	@pytest.mark.parametrize("columns", [
		pytest.param([], id="initial"),
		pytest.param(RED_WINS_AT_THREE, id="red-can-win"),
		pytest.param(YELLOW_THREATENS_THREE, id="yellow-threatens"),
	])
	@pytest.mark.parametrize("player", list(Player))
	def test_leaves_the_state_unchanged(
		self,
		play,
		columns,
		player: Player,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(random.Random(0))
		state = play(columns)
		before = copy_state(state)

		ordering.order_moves(state, ALL_MOVES, player)

		assert state == before

	def test_does_not_modify_the_given_moves(
		self,
		game: ConnectFourGame,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(ReversingShuffle())
		moves = list(ALL_MOVES)

		ordering.order_moves(game.initial_state(), moves, Player.RED)

		assert moves == ALL_MOVES

	def test_returns_a_new_list(self, game: ConnectFourGame) -> None:
		ordering = RandomConnectFourMoveOrdering(random.Random(0))
		moves = list(ALL_MOVES)

		ordered = ordering.order_moves(
			game.initial_state(),
			moves,
			Player.RED,
		)

		assert isinstance(ordered, list)
		assert ordered is not moves

	def test_empty_input_gives_empty_output(
		self,
		game: ConnectFourGame,
	) -> None:
		ordering = RandomConnectFourMoveOrdering(random.Random(0))

		assert ordering.order_moves(game.initial_state(), [], Player.RED) == []
