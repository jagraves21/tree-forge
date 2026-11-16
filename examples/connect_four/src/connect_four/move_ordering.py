import random
from collections.abc import Sequence

from tree_forge import MoveOrdering

from .game import (
	ConnectFourGame,
	GameState,
	Move,
	Player,
)


class ConnectFourMoveOrdering(
	MoveOrdering[GameState, Move, Player]
):
	"""Order Connect Four moves by tactical priority and player preference."""

	def __init__(self, game: ConnectFourGame) -> None:
		self.game = game

	def order_moves(
		self,
		state: GameState,
		moves: Sequence[Move],
		player: Player,
	) -> Sequence[Move]:
		opponent = player.opponent

		def priority(move: Move) -> tuple[int, int]:
			# Test whether this move wins immediately.
			undo_data = self.game.make_move(state, move)

			try:
				if self.game.winner(state) is player:
					return (0, self._side_priority(move, player))
			finally:
				self.game.unmake_move(state, move, undo_data)

			# Test whether the opponent could win immediately
			# if allowed to play in this position.
			original_player = state.current_player
			state.current_player = opponent

			try:
				opponent_can_win = any(
					self._is_winning_move(state, opponent, candidate)
					for candidate in self.game.legal_moves(state)
				)
			finally:
				state.current_player = original_player

			if opponent_can_win:
				return (1, self._side_priority(move, player))

			return (2, self._side_priority(move, player))

		return sorted(moves, key=priority)

	def _is_winning_move(
		self,
		state: GameState,
		player: Player,
		move: Move,
	) -> bool:
		undo_data = self.game.make_move(state, move)

		try:
			return self.game.winner(state) is player
		finally:
			self.game.unmake_move(state, move, undo_data)

	@staticmethod
	def _side_priority(move: Move, player: Player) -> int:
		"""Prefer left for Red and right for Yellow."""
		if player is Player.RED:
			return move.column

		return -move.column


class RandomConnectFourMoveOrdering(
	MoveOrdering[GameState, Move, Player]
):
	"""Order moves by tactical priority, randomizing within each group."""

	def __init__(
		self,
		game: ConnectFourGame,
		rng: random.Random | None = None,
	) -> None:
		self.game = game
		self.rng = rng if rng is not None else random.Random()

	def order_moves(
		self,
		state: GameState,
		moves: Sequence[Move],
		player: Player,
	) -> Sequence[Move]:
		opponent = player.opponent

		groups: list[list[Move]] = [[], [], []]

		for move in moves:
			# Winning moves have the highest priority.
			undo_data = self.game.make_move(state, move)

			try:
				if self.game.winner(state) is player:
					groups[0].append(move)
					continue
			finally:
				self.game.unmake_move(state, move, undo_data)

			# Check whether the opponent could win immediately
			# after this move.
			original_player = state.current_player
			state.current_player = opponent

			try:
				opponent_can_win = any(
					self._is_winning_move(state, opponent, candidate)
					for candidate in self.game.legal_moves(state)
				)
			finally:
				state.current_player = original_player

			if opponent_can_win:
				groups[1].append(move)
			else:
				groups[2].append(move)

		# Randomize only within groups of equal tactical priority.
		ordered_moves: list[Move] = []

		for group in groups:
			self.rng.shuffle(group)
			ordered_moves.extend(group)

		return ordered_moves

	def _is_winning_move(
		self,
		state: GameState,
		player: Player,
		move: Move,
	) -> bool:
		undo_data = self.game.make_move(state, move)

		try:
			return self.game.winner(state) is player
		finally:
			self.game.unmake_move(state, move, undo_data)

