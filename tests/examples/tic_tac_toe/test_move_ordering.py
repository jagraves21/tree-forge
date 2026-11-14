from tic_tac_toe import (
	GameState,
	Move,
	Player,
	TicTacToeGame,
	TicTacToeMoveOrdering,
)


class TestTicTacToeMoveOrdering:
	def setup_method(self) -> None:
		self.game = TicTacToeGame()
		self.ordering = TicTacToeMoveOrdering()

	def test_orders_center_before_corners_and_edges(self) -> None:
		moves = [
			Move(0, 1),
			Move(2, 2),
			Move(1, 1),
			Move(1, 0),
			Move(0, 0),
		]

		ordered = self.ordering.order_moves(
			self.game.initial_state(),
			moves,
			Player.X,
		)

		assert ordered == [
			Move(1, 1),
			Move(2, 2),
			Move(0, 0),
			Move(0, 1),
			Move(1, 0),
		]

	def test_preserves_order_within_priority_groups(self) -> None:
		moves = [
			Move(2, 2),
			Move(0, 0),
			Move(2, 1),
			Move(0, 1),
		]

		ordered = self.ordering.order_moves(
			self.game.initial_state(),
			moves,
			Player.X,
		)

		assert ordered == [
			Move(2, 2),
			Move(0, 0),
			Move(2, 1),
			Move(0, 1),
		]

	def test_returns_every_legal_move_exactly_once(self) -> None:
		state = self.game.initial_state()
		moves = self.game.legal_moves(state)

		ordered = self.ordering.order_moves(
			state,
			moves,
			self.game.current_player(state),
		)

		assert len(ordered) == len(moves)
		assert sorted(ordered) == sorted(moves)

	def test_orders_remaining_moves_after_board_is_partially_filled(
		self,
	) -> None:
		state = self.game.initial_state()

		self.game.make_move(state, Move(1, 1))
		moves = self.game.legal_moves(state)

		ordered = self.ordering.order_moves(
			state,
			moves,
			self.game.current_player(state),
		)

		assert Move(0, 0) in ordered[:4]
		assert Move(0, 1) not in ordered[:4]
		assert len(ordered) == 8

