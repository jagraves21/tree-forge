import pytest

from tic_tac_toe import (
	BOARD_SIZE,
	GameState,
	Move,
	Player,
	TicTacToeGame,
)
from tic_tac_toe.canonicalizer import TicTacToeCanonicalizer


class TestTicTacToeCanonicalizer:
	@pytest.fixture(autouse=True)
	def setup(self) -> None:
		self.game = TicTacToeGame()
		self.canonicalizer = TicTacToeCanonicalizer(self.game)

	def make_state(
		self,
		board: list[list[Player | None]],
		current_player: Player = Player.X,
	) -> GameState:
		return GameState(
			board=board,
			current_player=current_player,
		)

	# Position keys

	def test_position_key_includes_current_player(self) -> None:
		state = self.game.initial_state()

		assert self.canonicalizer.position_key(state) == "X........."

		state.current_player = Player.O

		assert self.canonicalizer.position_key(state) == "O........."

	def test_position_key_flattens_board_row_wise(self) -> None:
		state = self.make_state(
			[
				[Player.X, None, None],
				[None, Player.O, None],
				[None, None, Player.X],
			]
		)

		assert self.canonicalizer.position_key(state) == "X X...O...X".replace(
			" ",
			"",
		)

	def test_identical_positions_have_identical_keys(self) -> None:
		state_a = self.game.initial_state()
		state_b = self.game.initial_state()

		assert (
			self.canonicalizer.position_key(state_a)
			== self.canonicalizer.position_key(state_b)
		)

	def test_different_positions_have_different_keys(self) -> None:
		state_a = self.game.initial_state()
		state_b = self.game.initial_state()

		self.game.make_move(state_a, Move(0, 0))
		self.game.make_move(state_b, Move(0, 1))

		assert (
			self.canonicalizer.position_key(state_a)
			!= self.canonicalizer.position_key(state_b)
		)

	# Position hashing

	def test_position_hash_matches_position_key(self) -> None:
		state = self.game.initial_state()

		assert self.canonicalizer.position_hash(state) == (
			self.canonicalizer._position_string_to_int(
				self.canonicalizer.position_key(state)
			)
		)

	def test_identical_positions_have_identical_hashes(self) -> None:
		state_a = self.game.initial_state()
		state_b = self.game.initial_state()

		assert (
			self.canonicalizer.position_hash(state_a)
			== self.canonicalizer.position_hash(state_b)
		)

	def test_different_positions_have_different_hashes(self) -> None:
		state_a = self.game.initial_state()
		state_b = self.game.initial_state()

		self.game.make_move(state_a, Move(0, 0))
		self.game.make_move(state_b, Move(0, 1))

		assert (
			self.canonicalizer.position_hash(state_a)
			!= self.canonicalizer.position_hash(state_b)
		)

	def test_position_hash_accounts_for_current_player(self) -> None:
		state_a = self.game.initial_state()
		state_b = self.game.initial_state()

		state_b.current_player = Player.O

		assert (
			self.canonicalizer.position_hash(state_a)
			!= self.canonicalizer.position_hash(state_b)
		)

	def test_position_hash_is_collision_free_for_sample_positions(self) -> None:
		positions = []

		for current_player in Player:
			for first_row in range(BOARD_SIZE):
				for first_column in range(BOARD_SIZE):
					state = self.game.initial_state()
					state.current_player = current_player
					state.board[first_row][first_column] = Player.X

					positions.append(
						self.canonicalizer.position_hash(state)
					)

		assert len(positions) == len(set(positions))

	# Board symmetries

	def test_iter_symmetries_returns_eight_symmetries(self) -> None:
		state = self.game.initial_state()

		symmetries = list(
			self.canonicalizer.iter_symmetries(state)
		)

		assert len(symmetries) == 8

	def test_iter_symmetries_preserves_board_contents(self) -> None:
		state = self.make_state(
			[
				[Player.X, None, None],
				[None, None, Player.O],
				[None, Player.X, None],
			]
		)

		original_pieces = sorted(
			cell.value if cell is not None else 0
			for row in state.board
			for cell in row
		)

		for key, _ in self.canonicalizer.iter_symmetries(state):
			symmetric_pieces = sorted(
				0 if cell == "." else 1 if cell == "X" else 2
				for cell in key[1:]
			)

			assert symmetric_pieces == original_pieces

	def test_iter_symmetries_preserves_current_player(self) -> None:
		state = self.game.initial_state()

		for current_player in Player:
			state.current_player = current_player

			for key, _ in self.canonicalizer.iter_symmetries(state):
				assert key[0] == current_player.name

	def test_iter_symmetries_transforms_moves_correctly(self) -> None:
		state = self.game.initial_state()

		for _, transform in self.canonicalizer.iter_symmetries(state):
			transformed = {
				transform(Move(row, column))
				for row in range(BOARD_SIZE)
				for column in range(BOARD_SIZE)
			}

			assert len(transformed) == BOARD_SIZE * BOARD_SIZE
			assert all(
				0 <= move.row < BOARD_SIZE
				and 0 <= move.column < BOARD_SIZE
				for move in transformed
			)

	def test_iter_symmetries_transformations_are_reversible(self) -> None:
		state = self.game.initial_state()

		for _, transform in self.canonicalizer.iter_symmetries(state):
			transformed = {
				transform(Move(row, column))
				for row in range(BOARD_SIZE)
				for column in range(BOARD_SIZE)
			}

			assert len(transformed) == BOARD_SIZE * BOARD_SIZE

	# Canonical position keys

	def test_canonical_position_key_matches_position_key_for_empty_board(
		self,
	) -> None:
		state = self.game.initial_state()

		assert (
			self.canonicalizer.canonical_position_key(state)
			== self.canonicalizer.position_key(state)
		)

	def test_canonical_position_key_is_independent_of_rotation(self) -> None:
		state = self.make_state(
			[
				[Player.X, None, None],
				[None, None, Player.O],
				[None, Player.X, None],
			]
		)

		canonical_key = self.canonicalizer.canonical_position_key(state)

		for rotation in range(1, 4):
			_, transform = list(
				self.canonicalizer.iter_symmetries(state)
			)[rotation * 2]

			symmetric = self.make_state(
				[
					[None for _ in range(BOARD_SIZE)]
					for _ in range(BOARD_SIZE)
				],
				state.current_player,
			)

			for row in range(BOARD_SIZE):
				for column in range(BOARD_SIZE):
					move = transform(Move(row, column))
					symmetric.board[move.row][move.column] = (
						state.board[row][column]
					)

			assert (
				self.canonicalizer.canonical_position_key(symmetric)
				== canonical_key
			)

	def test_canonical_position_key_is_independent_of_reflection(self) -> None:
		state = self.make_state(
			[
				[Player.X, None, None],
				[None, None, Player.O],
				[None, Player.X, None],
			]
		)

		_, transform = list(
			self.canonicalizer.iter_symmetries(state)
		)[1]

		reflected = self.make_state(
			[
				[None for _ in range(BOARD_SIZE)]
				for _ in range(BOARD_SIZE)
			],
			state.current_player,
		)

		for row in range(BOARD_SIZE):
			for column in range(BOARD_SIZE):
				move = transform(Move(row, column))
				reflected.board[move.row][move.column] = (
					state.board[row][column]
				)

		assert (
			self.canonicalizer.canonical_position_key(reflected)
			== self.canonicalizer.canonical_position_key(state)
		)

	def test_canonical_position_key_matches_all_eight_symmetries(
		self,
	) -> None:
		state = self.make_state(
			[
				[Player.X, None, Player.O],
				[None, Player.X, None],
				[Player.O, None, None],
			]
		)

		canonical_key = self.canonicalizer.canonical_position_key(state)

		for key, transform in self.canonicalizer.iter_symmetries(state):
			symmetric = self.make_state(
				[
					[None for _ in range(BOARD_SIZE)]
					for _ in range(BOARD_SIZE)
				],
				state.current_player,
			)

			for row in range(BOARD_SIZE):
				for column in range(BOARD_SIZE):
					move = transform(Move(row, column))
					symmetric.board[move.row][move.column] = (
						state.board[row][column]
					)

			assert (
				self.canonicalizer.canonical_position_key(symmetric)
				== canonical_key
			)

	def test_canonical_position_key_includes_current_player(self) -> None:
		board = [
			[Player.X, None, Player.O],
			[None, Player.X, None],
			[Player.O, None, None],
		]

		x_to_move = self.make_state(board, Player.X)
		o_to_move = self.make_state(
			[row.copy() for row in board],
			Player.O,
		)

		assert (
			self.canonicalizer.canonical_position_key(x_to_move)
			!= self.canonicalizer.canonical_position_key(o_to_move)
		)

	def test_canonical_position_key_is_lexicographically_smallest(
		self,
	) -> None:
		state = self.make_state(
			[
				[Player.X, None, Player.O],
				[None, Player.X, None],
				[Player.O, None, None],
			]
		)

		canonical_key = self.canonicalizer.canonical_position_key(state)

		symmetry_keys = [
			key
			for key, _ in self.canonicalizer.iter_symmetries(state)
		]

		assert canonical_key == min(symmetry_keys)

	# Canonical position hashing

	def test_canonical_position_hash_matches_canonical_position_key(
		self,
	) -> None:
		state = self.game.initial_state()

		assert self.canonicalizer.canonical_position_hash(state) == (
			self.canonicalizer._position_string_to_int(
				self.canonicalizer.canonical_position_key(state)
			)
		)

	def test_symmetric_positions_have_identical_canonical_hashes(
		self,
	) -> None:
		state = self.make_state(
			[
				[Player.X, None, Player.O],
				[None, Player.X, None],
				[Player.O, None, None],
			]
		)

		canonical_hash = self.canonicalizer.canonical_position_hash(state)

		for _, transform in self.canonicalizer.iter_symmetries(state):
			symmetric = self.make_state(
				[
					[None for _ in range(BOARD_SIZE)]
					for _ in range(BOARD_SIZE)
				],
				state.current_player,
			)

			for row in range(BOARD_SIZE):
				for column in range(BOARD_SIZE):
					move = transform(Move(row, column))
					symmetric.board[move.row][move.column] = (
						state.board[row][column]
					)

			assert (
				self.canonicalizer.canonical_position_hash(symmetric)
				== canonical_hash
			)

	def test_canonical_position_hash_accounts_for_current_player(
		self,
	) -> None:
		board = [
			[Player.X, None, Player.O],
			[None, Player.X, None],
			[Player.O, None, None],
		]

		x_to_move = self.make_state(board, Player.X)
		o_to_move = self.make_state(
			[row.copy() for row in board],
			Player.O,
		)

		assert (
			self.canonicalizer.canonical_position_hash(x_to_move)
			!= self.canonicalizer.canonical_position_hash(o_to_move)
		)

	# Canonicalization and move transformations

	def test_returns_lexicographically_smallest_key(self) -> None:
		state = self.make_state(
			[
				[Player.X, None, Player.O],
				[None, Player.X, None],
				[Player.O, None, None],
			]
		)

		result = self.canonicalizer.canonicalize(state)

		expected_key = min(
			key
			for key, _ in self.canonicalizer.iter_symmetries(state)
		)

		assert result.key == expected_key

	def test_returns_valid_move_transformations(self) -> None:
		state = self.make_state(
			[
				[Player.X, None, Player.O],
				[None, Player.X, None],
				[Player.O, None, None],
			]
		)

		result = self.canonicalizer.canonicalize(state)

		for row in range(BOARD_SIZE):
			for column in range(BOARD_SIZE):
				move = Move(row, column)
				canonical_move = result.to_canonical_move(move)
				restored_move = result.from_canonical_move(canonical_move)

				assert 0 <= canonical_move.row < BOARD_SIZE
				assert 0 <= canonical_move.column < BOARD_SIZE
				assert restored_move == move

	def test_round_trips_all_board_moves(self) -> None:
		state = self.game.initial_state()
		result = self.canonicalizer.canonicalize(state)

		for row in range(BOARD_SIZE):
			for column in range(BOARD_SIZE):
				move = Move(row, column)

				assert (
					result.from_canonical_move(
						result.to_canonical_move(move)
					)
					== move
				)

	def test_maps_legal_moves_to_canonical_coordinates(self) -> None:
		state = self.make_state(
			[
				[Player.X, None, Player.O],
				[None, Player.X, None],
				[Player.O, None, None],
			]
		)

		result = self.canonicalizer.canonicalize(state)

		for move in self.game.legal_moves(state):
			canonical_move = result.to_canonical_move(move)

			assert (
				result.from_canonical_move(canonical_move)
				== move
			)

	def test_symmetric_states_have_same_canonical_key(self) -> None:
		state = self.make_state(
			[
				[Player.X, None, Player.O],
				[None, Player.X, None],
				[Player.O, None, None],
			]
		)

		_, transform = list(
			self.canonicalizer.iter_symmetries(state)
		)[1]

		symmetric_state = self.make_state(
			[
				[None for _ in range(BOARD_SIZE)]
				for _ in range(BOARD_SIZE)
			],
			state.current_player,
		)

		for row in range(BOARD_SIZE):
			for column in range(BOARD_SIZE):
				move = transform(Move(row, column))
				symmetric_state.board[move.row][move.column] = (
					state.board[row][column]
				)

		first = self.canonicalizer.canonicalize(state)
		second = self.canonicalizer.canonicalize(symmetric_state)

		assert first.key == second.key

	def test_handles_empty_board(self) -> None:
		state = self.game.initial_state()

		result = self.canonicalizer.canonicalize(state)

		assert result.key == self.canonicalizer.canonical_position_key(state)

		for row in range(BOARD_SIZE):
			for column in range(BOARD_SIZE):
				move = Move(row, column)

				assert (
					result.from_canonical_move(
						result.to_canonical_move(move)
					)
					== move
				)

	def test_rejects_invalid_canonical_move(self) -> None:
		state = self.game.initial_state()
		result = self.canonicalizer.canonicalize(state)

		with pytest.raises(ValueError):
			result.from_canonical_move(Move(-1, -1))

