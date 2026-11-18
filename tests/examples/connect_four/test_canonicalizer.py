import pytest

from connect_four import (
	BOARD_COLUMNS,
	BOARD_ROWS,
	ConnectFourGame,
	GameState,
	Move,
	Player,
)
from connect_four.canonicalizer import ConnectFourCanonicalizer


EMPTY_BOARD_CELLS = "." * (BOARD_ROWS * BOARD_COLUMNS)


def mirror_state(state: GameState) -> GameState:
	return GameState(
		board=[row[::-1] for row in state.board],
		current_player=state.current_player,
		move_count=state.move_count,
	)


def mirror_columns(columns) -> list[int]:
	return [BOARD_COLUMNS - 1 - column for column in columns]


@pytest.fixture
def canonicalizer() -> ConnectFourCanonicalizer:
	return ConnectFourCanonicalizer()


class TestPositionKey:
	def test_initial_position(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		game: ConnectFourGame,
	) -> None:
		key = canonicalizer.position_key(game.initial_state())

		assert key == "1:0:" + EMPTY_BOARD_CELLS

	def test_encodes_player_move_count_and_board_row_by_row(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		# Red's piece lands on the bottom row, which is cells 35 to 41.
		key = canonicalizer.position_key(play([3]))

		assert key == "2:1:" + "." * 35 + "...R..."

	def test_encodes_yellow_pieces(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		key = canonicalizer.position_key(play([3, 3]))

		assert key == "1:2:" + "." * 28 + "...Y..." + "...R..."

	def test_key_length_is_fixed_for_a_given_move_count(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		prefix, _, board = canonicalizer.position_key(play([0, 1])).rpartition(":")

		assert prefix == "1:2"
		assert len(board) == BOARD_ROWS * BOARD_COLUMNS

	def test_equal_positions_have_equal_keys(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		assert canonicalizer.position_key(play([3, 4])) == (
			canonicalizer.position_key(play([3, 4]))
		)

	def test_different_boards_have_different_keys(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		assert canonicalizer.position_key(play([3])) != (
			canonicalizer.position_key(play([4]))
		)

	def test_move_count_is_part_of_the_key(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		state = play([3])
		other = GameState(
			board=[row.copy() for row in state.board],
			current_player=state.current_player,
			move_count=state.move_count + 2,
		)

		assert canonicalizer.position_key(state) != (
			canonicalizer.position_key(other)
		)

	def test_player_to_move_is_part_of_the_key(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		state = play([3])
		other = GameState(
			board=[row.copy() for row in state.board],
			current_player=Player.RED,
			move_count=state.move_count,
		)

		assert canonicalizer.position_key(state) != (
			canonicalizer.position_key(other)
		)


class TestPositionHash:
	@staticmethod
	def decode(value: int) -> str:
		return value.to_bytes((value.bit_length() + 7) // 8, "big").decode()

	def test_encodes_the_position_key_reversibly(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		state = play([3, 2, 3])

		assert self.decode(canonicalizer.position_hash(state)) == (
			canonicalizer.position_key(state)
		)

	def test_equal_positions_have_equal_hashes(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		assert canonicalizer.position_hash(play([3, 4])) == (
			canonicalizer.position_hash(play([3, 4]))
		)

	def test_different_positions_have_different_hashes(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		hashes = {
			canonicalizer.position_hash(play(columns))
			for columns in ([], [0], [1], [0, 1], [1, 0], [3, 3])
		}

		assert len(hashes) == 6


class TestIterSymmetries:
	def test_yields_the_identity_and_the_mirror_image(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		state = play([0])

		(first_key, first), (second_key, second) = list(
			canonicalizer.iter_symmetries(state)
		)

		assert first_key == canonicalizer.position_key(state)
		assert second_key == canonicalizer.position_key(mirror_state(state))
		assert first(Move(1)) == Move(1)
		assert second(Move(1)) == Move(5)

	def test_mirror_maps_every_column_to_its_opposite(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		game: ConnectFourGame,
	) -> None:
		_, mirror = list(canonicalizer.iter_symmetries(game.initial_state()))[1]

		for column in range(BOARD_COLUMNS):
			assert mirror(Move(column)) == Move(BOARD_COLUMNS - 1 - column)

	def test_mirror_is_its_own_inverse(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		game: ConnectFourGame,
	) -> None:
		_, mirror = list(canonicalizer.iter_symmetries(game.initial_state()))[1]

		for column in range(BOARD_COLUMNS):
			assert mirror(mirror(Move(column))) == Move(column)

	def test_symmetric_position_has_a_single_distinct_key(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		state = play([3, 3, 3])

		keys = {key for key, _ in canonicalizer.iter_symmetries(state)}

		assert len(keys) == 1

	def test_asymmetric_position_has_two_distinct_keys(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		keys = {key for key, _ in canonicalizer.iter_symmetries(play([0]))}

		assert len(keys) == 2

	def test_preserves_player_and_move_count(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		for key, _ in canonicalizer.iter_symmetries(play([0, 1, 2])):
			assert key.startswith("2:3:")


class TestCanonicalPositionKey:
	def test_is_the_smaller_of_the_two_symmetric_keys(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		state = play([0, 5, 1])

		assert canonicalizer.canonical_position_key(state) == min(
			key for key, _ in canonicalizer.iter_symmetries(state)
		)

	def test_mirrored_positions_share_a_canonical_key(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		columns = [0, 1, 0, 2, 5]

		assert canonicalizer.canonical_position_key(play(columns)) == (
			canonicalizer.canonical_position_key(play(mirror_columns(columns)))
		)

	def test_exact_keys_of_mirrored_positions_differ(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		assert canonicalizer.position_key(play([0])) != (
			canonicalizer.position_key(play([6]))
		)

	def test_non_mirrored_positions_stay_distinct(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		assert canonicalizer.canonical_position_key(play([0])) != (
			canonicalizer.canonical_position_key(play([1]))
		)

	def test_distinguishes_the_player_to_move(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		state = play([3])
		other = GameState(
			board=[row.copy() for row in state.board],
			current_player=Player.RED,
			move_count=state.move_count,
		)

		assert canonicalizer.canonical_position_key(state) != (
			canonicalizer.canonical_position_key(other)
		)


class TestCanonicalPositionHash:
	def test_encodes_the_canonical_key_reversibly(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		state = play([0, 5, 1])
		value = canonicalizer.canonical_position_hash(state)

		assert TestPositionHash.decode(value) == (
			canonicalizer.canonical_position_key(state)
		)

	def test_mirrored_positions_share_a_canonical_hash(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		columns = [0, 1, 0, 2, 5]

		assert canonicalizer.canonical_position_hash(play(columns)) == (
			canonicalizer.canonical_position_hash(play(mirror_columns(columns)))
		)

	def test_distinct_positions_have_distinct_canonical_hashes(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		assert canonicalizer.canonical_position_hash(play([0])) != (
			canonicalizer.canonical_position_hash(play([1]))
		)


class TestCanonicalize:
	def test_key_is_the_canonical_position_key(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		state = play([0, 5, 1])

		assert canonicalizer.canonicalize(state).key == (
			canonicalizer.canonical_position_key(state)
		)

	def test_position_already_in_canonical_form_maps_moves_to_themselves(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		# Red's piece on the right edge gives the smaller key, so this
		# position is already canonical.
		position = canonicalizer.canonicalize(play([6]))

		for column in range(BOARD_COLUMNS):
			assert position.to_canonical_move(Move(column)) == Move(column)

	def test_mirrored_position_maps_moves_to_their_mirror_image(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		position = canonicalizer.canonicalize(play([0]))

		for column in range(BOARD_COLUMNS):
			assert position.to_canonical_move(Move(column)) == Move(
				BOARD_COLUMNS - 1 - column
			)

	def test_mirrored_positions_agree_on_the_canonical_key_and_best_move(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		original = canonicalizer.canonicalize(play([0]))
		mirrored = canonicalizer.canonicalize(play([6]))

		assert original.key == mirrored.key
		assert original.to_canonical_move(Move(1)) == (
			mirrored.to_canonical_move(Move(5))
		)

	def test_transformation_produces_the_canonical_board(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		columns = [0, 1, 0, 2, 5]
		position = canonicalizer.canonicalize(play(columns))

		canonical_columns = [
			position.to_canonical_move(Move(column)).column
			for column in columns
		]

		assert canonicalizer.position_key(play(canonical_columns)) == (
			position.key
		)

	def test_moves_round_trip_through_both_transformations(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		play,
	) -> None:
		position = canonicalizer.canonicalize(play([0, 5, 1]))

		for column in range(BOARD_COLUMNS):
			move = Move(column)

			assert position.from_canonical_move(
				position.to_canonical_move(move)
			) == move

	def test_symmetric_position_round_trips_every_move(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		game: ConnectFourGame,
	) -> None:
		position = canonicalizer.canonicalize(game.initial_state())

		for column in range(BOARD_COLUMNS):
			move = Move(column)

			assert position.from_canonical_move(
				position.to_canonical_move(move)
			) == move

	@pytest.mark.parametrize("column", [-1, BOARD_COLUMNS, 99])
	def test_rejects_canonical_moves_outside_the_board(
		self,
		canonicalizer: ConnectFourCanonicalizer,
		game: ConnectFourGame,
		column: int,
	) -> None:
		position = canonicalizer.canonicalize(game.initial_state())

		with pytest.raises(ValueError, match="Invalid canonical move"):
			position.from_canonical_move(Move(column))
