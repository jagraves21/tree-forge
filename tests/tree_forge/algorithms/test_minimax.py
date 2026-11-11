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

	def asymmetric_state(self) -> GameState:
		return self.make_state(
			[
				[Player.X, None, Player.O],
				[None, Player.X, None],
				[Player.O, None, None],
			]
		)

	def transform_state(
		self,
		state: GameState,
		transform,
	) -> GameState:
		"""Apply a move transformation to every board position."""
		board = [
			[None for _ in range(BOARD_SIZE)]
			for _ in range(BOARD_SIZE)
		]

		for row in range(BOARD_SIZE):
			for column in range(BOARD_SIZE):
				move = Move(row, column)
				transformed_move = transform(move)

				board[transformed_move.row][transformed_move.column] = (
					state.board[row][column]
				)

		return self.make_state(board, state.current_player)

	@staticmethod
	def encode_position(position: str) -> int:
		"""Independently encode a position string as a base-three integer."""
		values = {
			".": 0,
			"X": 1,
			"O": 2,
		}

		result = 0

		for symbol in position:
			result = result * 3 + values[symbol]

		return result

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

		assert self.canonicalizer.position_key(state) == "XX...O...X"

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

	def test_position_hash_encodes_position_key(self) -> None:
		state = self.asymmetric_state()
		position_key = self.canonicalizer.position_key(state)

		assert self.canonicalizer.position_hash(state) == (
			self.encode_position(position_key)
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

	def test_position_hash_is_collision_free_for_sample_positions(
		self,
	) -> None:
		hashes = set()

		for current_player in Player:
			for row in range(BOARD_SIZE):
				for column in range(BOARD_SIZE):
					state = self.game.initial_state()
					state.current_player = current_player
					state.board[row][column] = Player.X

					hashes.add(
						self.canonicalizer.position_hash(state)
					)

		assert len(hashes) == 2 * BOARD_SIZE * BOARD_SIZE

	# Board symmetries

	def test_iter_symmetries_returns_eight_symmetries(self) -> None:
		state = self.asymmetric_state()

		symmetries = list(
			self.canonicalizer.iter_symmetries(state)
		)

		assert len(symmetries) == 8

	def test_iter_symmetries_preserves_board_contents(self) -> None:
		state = self.asymmetric_state()

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
		state = self.asymmetric_state()

		for current_player in Player:
			state.current_player = current_player

			for key, _ in self.canonicalizer.iter_symmetries(state):
				assert key[0] == current_player.name

	def test_iter_symmetries_transforms_moves_correctly(self) -> None:
		state = self.asymmetric_state()

		for _, transform in self.canonicalizer.iter_symmetries(state):
			transformed_moves = {
				transform(Move(row, column))
				for row in range(BOARD_SIZE)
				for column in range(BOARD_SIZE)
			}

			assert len(transformed_moves) == BOARD_SIZE * BOARD_SIZE
			assert all(
				0 <= move.row < BOARD_SIZE
				and 0 <= move.column < BOARD_SIZE
				for move in transformed_moves
			)

	def test_iter_symmetries_transformations_are_reversible(self) -> None:
		state = self.asymmetric_state()

		for _, transform in self.canonicalizer.iter_symmetries(state):
			transformed_moves = {
				transform(Move(row, column))
				for row in range(BOARD_SIZE)
				for column in range(BOARD_SIZE)
			}

			assert len(transformed_moves) == BOARD_SIZE * BOARD_SIZE

			for row in range(BOARD_SIZE):
				for column in range(BOARD_SIZE):
					move = Move(row, column)
					transformed_move = transform(move)

					assert transformed_move in transformed_moves

	def test_iter_symmetries_transforms_board_to_reported_key(
		self,
	) -> None:
		state = self.asymmetric_state()

		for key, transform in self.canonicalizer.iter_symmetries(state):
			transformed_state = self.transform_state(state, transform)

			assert (
				self.canonicalizer.position_key(transformed_state)
				== key
			)

	# Canonical position keys

	def test_canonical_position_key_matches_position_key_for_empty_board(
		self,
	) -> None:
		state = self.game.initial_state()

		assert (
			self.canonicalizer.canonical_position_key(state)
			== self.canonicalizer.position_key(state)
		)

	def test_canonical_position_key_matches_all_eight_symmetries(
		self,
	) -> None:
		state = self.asymmetric_state()
		canonical_key = self.canonicalizer.canonical_position_key(state)

		for _, transform in self.canonicalizer.iter_symmetries(state):
			symmetric_state = self.transform_state(state, transform)

			assert (
				self.canonicalizer.canonical_position_key(symmetric_state)
				== canonical_key
			)

	def test_canonical_position_key_includes_current_player(self) -> None:
		state_x = self.asymmetric_state()
		state_o = self.asymmetric_state()
		state_o.current_player = Player.O

		assert (
			self.canonicalizer.canonical_position_key(state_x)
			!= self.canonicalizer.canonical_position_key(state_o)
		)

	def test_canonical_position_key_is_lexicographically_smallest(
		self,
	) -> None:
		state = self.asymmetric_state()

		symmetry_keys = [
			key
			for key, _ in self.canonicalizer.iter_symmetries(state)
		]

		assert (
			self.canonicalizer.canonical_position_key(state)
			== min(symmetry_keys)
		)

	# Canonical position hashing

	def test_canonical_position_hash_encodes_canonical_position_key(
		self,
	) -> None:
		state = self.asymmetric_state()
		canonical_key = self.canonicalizer.canonical_position_key(state)

		assert self.canonicalizer.canonical_position_hash(state) == (
			self.encode_position(canonical_key)
		)

	def test_symmetric_positions_have_identical_canonical_hashes(
		self,
	) -> None:
		state = self.asymmetric_state()
		canonical_hash = self.canonicalizer.canonical_position_hash(state)

		for _, transform in self.canonicalizer.iter_symmetries(state):
			symmetric_state = self.transform_state(state, transform)

			assert (
				self.canonicalizer.canonical_position_hash(symmetric_state)
				== canonical_hash
			)

	def test_canonical_position_hash_accounts_for_current_player(
		self,
	) -> None:
		state_x = self.asymmetric_state()
		state_o = self.asymmetric_state()
		state_o.current_player = Player.O

		assert (
			self.canonicalizer.canonical_position_hash(state_x)
			!= self.canonicalizer.canonical_position_hash(state_o)
		)

	# Canonicalization and move transformations

	def test_canonicalize_returns_lexicographically_smallest_key(
		self,
	) -> None:
		state = self.asymmetric_state()

		result = self.canonicalizer.canonicalize(state)

		expected_key = min(
			key
			for key, _ in self.canonicalizer.iter_symmetries(state)
		)

		assert result.key == expected_key

	def test_canonicalize_returns_reversible_move_transformations(
		self,
	) -> None:
		state = self.asymmetric_state()
		result = self.canonicalizer.canonicalize(state)

		for row in range(BOARD_SIZE):
			for column in range(BOARD_SIZE):
				move = Move(row, column)
				canonical_move = result.to_canonical_move(move)
				restored_move = result.from_canonical_move(canonical_move)

				assert 0 <= canonical_move.row < BOARD_SIZE
				assert 0 <= canonical_move.column < BOARD_SIZE
				assert restored_move == move

	def test_canonicalize_round_trips_all_board_moves(self) -> None:
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

	def test_canonicalize_round_trips_all_legal_moves(self) -> None:
		state = self.asymmetric_state()
		result = self.canonicalizer.canonicalize(state)

		for move in self.game.legal_moves(state):
			assert (
				result.from_canonical_move(
					result.to_canonical_move(move)
				)
				== move
			)

	def test_canonicalize_maps_moves_to_valid_coordinates(self) -> None:
		state = self.asymmetric_state()
		result = self.canonicalizer.canonicalize(state)

		for row in range(BOARD_SIZE):
			for column in range(BOARD_SIZE):
				canonical_move = result.to_canonical_move(
					Move(row, column)
				)

				assert 0 <= canonical_move.row < BOARD_SIZE
				assert 0 <= canonical_move.column < BOARD_SIZE

	def test_canonicalize_uses_transformation_that_produces_its_key(
		self,
	) -> None:
		state = self.asymmetric_state()
		result = self.canonicalizer.canonicalize(state)

		transformed_state = self.transform_state(
			state,
			result.to_canonical_move,
		)

		assert (
			self.canonicalizer.position_key(transformed_state)
			== result.key
		)

	def test_symmetric_states_have_same_canonical_key(self) -> None:
		state = self.asymmetric_state()
		first = self.canonicalizer.canonicalize(state)

		for _, transform in self.canonicalizer.iter_symmetries(state):
			symmetric_state = self.transform_state(state, transform)
			second = self.canonicalizer.canonicalize(symmetric_state)

			assert first.key == second.key

	def test_canonicalize_handles_empty_board(self) -> None:
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

	def test_from_canonical_move_rejects_invalid_move(self) -> None:
		state = self.game.initial_state()
		result = self.canonicalizer.canonicalize(state)

		with pytest.raises(ValueError):
			result.from_canonical_move(Move(-1, -1))

