import pytest

from tic_tac_toe import (
	BOARD_SIZE,
	GameState,
	Move,
	Player,
	TicTacToeGame,
	TicTacToeMoveOrdering,
)


CENTER = Move(1, 1)
CORNERS = [Move(0, 0), Move(0, 2), Move(2, 0), Move(2, 2)]
EDGES = [Move(0, 1), Move(1, 0), Move(1, 2), Move(2, 1)]

ALL_MOVES = [
	Move(row, column)
	for row in range(BOARD_SIZE)
	for column in range(BOARD_SIZE)
]


@pytest.fixture
def ordering() -> TicTacToeMoveOrdering:
	return TicTacToeMoveOrdering()


class TestPriorities:
	def test_covers_every_board_cell_exactly_once(self) -> None:
		assert set(TicTacToeMoveOrdering.PRIORITIES) == {
			(move.row, move.column) for move in ALL_MOVES
		}

	def test_center_ranks_before_corners_before_edges(self) -> None:
		priorities = TicTacToeMoveOrdering.PRIORITIES

		assert priorities[(1, 1)] == 0
		assert {priorities[(m.row, m.column)] for m in CORNERS} == {1}
		assert {priorities[(m.row, m.column)] for m in EDGES} == {2}


class TestOrderMoves:
	def test_orders_center_then_corners_then_edges(
		self,
		ordering: TicTacToeMoveOrdering,
	) -> None:
		ordered = ordering.order_moves(GameState(), ALL_MOVES, Player.X)

		assert ordered == [CENTER, *CORNERS, *EDGES]

	def test_preserves_input_order_within_each_group(
		self,
		ordering: TicTacToeMoveOrdering,
	) -> None:
		ordered = ordering.order_moves(
			GameState(),
			list(reversed(ALL_MOVES)),
			Player.X,
		)

		assert ordered == [
			CENTER,
			*reversed(CORNERS),
			*reversed(EDGES),
		]

	def test_orders_a_partial_selection(
		self,
		ordering: TicTacToeMoveOrdering,
	) -> None:
		moves = [Move(0, 1), Move(2, 2), Move(1, 1), Move(1, 0), Move(0, 0)]

		ordered = ordering.order_moves(GameState(), moves, Player.X)

		assert ordered == [
			Move(1, 1),
			Move(2, 2),
			Move(0, 0),
			Move(0, 1),
			Move(1, 0),
		]

	def test_corners_come_first_when_the_center_is_unavailable(
		self,
		ordering: TicTacToeMoveOrdering,
	) -> None:
		moves = [move for move in ALL_MOVES if move != CENTER]

		ordered = ordering.order_moves(GameState(), moves, Player.X)

		assert ordered == [*CORNERS, *EDGES]

	def test_returns_every_move_exactly_once(
		self,
		ordering: TicTacToeMoveOrdering,
	) -> None:
		ordered = ordering.order_moves(GameState(), ALL_MOVES, Player.X)

		assert len(ordered) == len(ALL_MOVES)
		assert sorted(ordered) == sorted(ALL_MOVES)

	def test_orders_the_legal_moves_of_a_played_position(
		self,
		ordering: TicTacToeMoveOrdering,
	) -> None:
		game = TicTacToeGame()
		state = game.initial_state()
		game.make_move(state, CENTER)
		moves = game.legal_moves(state)

		ordered = ordering.order_moves(state, moves, game.current_player(state))

		assert ordered == [*CORNERS, *EDGES]

	def test_result_is_independent_of_the_player(
		self,
		ordering: TicTacToeMoveOrdering,
	) -> None:
		as_x = ordering.order_moves(GameState(), ALL_MOVES, Player.X)
		as_o = ordering.order_moves(GameState(), ALL_MOVES, Player.O)

		assert as_x == as_o

	def test_accepts_any_sequence_type(
		self,
		ordering: TicTacToeMoveOrdering,
	) -> None:
		ordered = ordering.order_moves(
			GameState(),
			tuple(ALL_MOVES),
			Player.X,
		)

		assert ordered == [CENTER, *CORNERS, *EDGES]

	def test_does_not_modify_the_input(
		self,
		ordering: TicTacToeMoveOrdering,
	) -> None:
		moves = list(reversed(ALL_MOVES))
		before = list(moves)

		ordering.order_moves(GameState(), moves, Player.X)

		assert moves == before

	def test_returns_a_new_list(
		self,
		ordering: TicTacToeMoveOrdering,
	) -> None:
		moves = list(ALL_MOVES)

		ordered = ordering.order_moves(GameState(), moves, Player.X)

		assert isinstance(ordered, list)
		assert ordered is not moves

	def test_empty_input_gives_empty_output(
		self,
		ordering: TicTacToeMoveOrdering,
	) -> None:
		assert ordering.order_moves(GameState(), [], Player.X) == []

	def test_single_move_is_returned_unchanged(
		self,
		ordering: TicTacToeMoveOrdering,
	) -> None:
		assert ordering.order_moves(GameState(), [Move(2, 1)], Player.X) == [
			Move(2, 1)
		]
