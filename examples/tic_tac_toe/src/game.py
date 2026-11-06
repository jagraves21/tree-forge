from dataclasses import dataclass
from enum import Enum


class Player(Enum):
	X = "X"
	O = "O"


@dataclass(slots=True)
class State:
	board: list[Player | None]
	current_player: Player = Player.X
	winner: Player | None = None
	move_count: int = 0


@dataclass(frozen=True, slots=True)
class Undo:
	previous_player: Player
	previous_winner: Player | None
	previous_move_count: int


class TicTacToe:
	WINNING_LINES = (
		(0, 1, 2),
		(3, 4, 5),
		(6, 7, 8),
		(0, 3, 6),
		(1, 4, 7),
		(2, 5, 8),
		(0, 4, 8),
		(2, 4, 6),
	)

	def initial_state(self) -> State:
		return State(board=[None] * 9)

	def current_player(self, state: State) -> Player:
		return state.current_player

	def legal_moves(self, state: State) -> tuple[int, ...]:
		if self.is_terminal(state):
			return ()

		return tuple(
			i for i, cell in enumerate(state.board)
			if cell is None
		)

	def make_move(self, state: State, move: int) -> Undo:
		if move not in self.legal_moves(state):
			raise ValueError(f"Illegal move: {move}")

		undo = Undo(
			previous_player=state.current_player,
			previous_winner=state.winner,
			previous_move_count=state.move_count,
		)

		state.board[move] = state.current_player
		state.move_count += 1
		state.winner = self._find_winner(state)

		state.current_player = (
			Player.O if state.current_player == Player.X else Player.X
		)

		return undo

	def unmake_move(self, state: State, move: int, undo: Undo) -> None:
		if not 0 <= move < 9 or state.board[move] is None:
			raise ValueError(f"Cannot undo move: {move}")

		state.board[move] = None
		state.current_player = undo.previous_player
		state.winner = undo.previous_winner
		state.move_count = undo.previous_move_count

	def is_terminal(self, state: State) -> bool:
		return state.winner is not None or state.move_count == 9

	def terminal_score(self, state: State, player: Player) -> float:
		if not self.is_terminal(state):
			raise ValueError("Position is not terminal")

		if state.winner is None:
			return 0.0

		return 1.0 if state.winner == player else -1.0

	def _find_winner(self, state: State) -> Player | None:
		for a, b, c in self.WINNING_LINES:
			player = state.board[a]

			if (
				player is not None
				and player == state.board[b]
				and player == state.board[c]
			):
				return player

		return None

