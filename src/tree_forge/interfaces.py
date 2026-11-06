
from collections.abc import Iterable
from typing import Hashable, Protocol, TypeVar


StateT = TypeVar("StateT")
MoveT = TypeVar("MoveT")
PlayerT = TypeVar("PlayerT", bound=Hashable)
UndoT = TypeVar("UndoT")


class Game(Protocol[StateT, MoveT, PlayerT, UndoT]):
	"""Interface implemented by supported games."""

	def initial_state(self) -> StateT:
		"""Return a new initial game state."""
		...

	def current_player(self, state: StateT) -> PlayerT:
		"""Return the player whose turn it is."""
		...

	def legal_moves(self, state: StateT) -> Iterable[MoveT]:
		"""Return all legal moves in the current state."""
		...

	def make_move(self, state: StateT, move: MoveT) -> UndoT:
		"""Apply a move and return the information needed to undo it."""
		...

	def unmake_move(
		self,
		state: StateT,
		move: MoveT,
		undo: UndoT,
	) -> None:
		"""Restore the state to its condition before the move."""
		...

	def is_terminal(self, state: StateT) -> bool:
		"""Return whether the game has ended."""
		...

	def terminal_score(
		self,
		state: StateT,
		player: PlayerT,
	) -> float:
		"""Return the terminal score from the specified player's perspective."""
		...

