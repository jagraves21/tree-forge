from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, Hashable, TypeVar


StateT = TypeVar("StateT")
MoveT = TypeVar("MoveT")
PlayerT = TypeVar("PlayerT", bound=Hashable)
UndoT = TypeVar("UndoT")


class Game(ABC, Generic[StateT, MoveT, PlayerT, UndoT]):
	"""Interface that all supported games must implement.

	The framework assumes deterministic, turn-based, two-player,
	zero-sum games.

	Game state is mutable during search. Every call to make_move()
	must be reversible through unmake_move().
	"""

	@abstractmethod
	def initial_state(self) -> StateT:
		"""Return a new initial game state."""
		raise NotImplementedError

	@abstractmethod
	def current_player(self, state: StateT) -> PlayerT:
		"""Return the player whose turn it is."""
		raise NotImplementedError

	@abstractmethod
	def legal_moves(self, state: StateT) -> Sequence[MoveT]:
		"""Return all legal moves in the current position."""
		raise NotImplementedError

	@abstractmethod
	def make_move(self, state: StateT, move: MoveT) -> UndoT:
		"""Apply a move and return the information needed to undo it.

		The supplied state should be modified in place.
		"""
		raise NotImplementedError

	@abstractmethod
	def unmake_move(
		self,
		state: StateT,
		move: MoveT,
		undo_data: UndoT,
	) -> None:
		"""Restore the state to exactly what it was before make_move()."""
		raise NotImplementedError

	@abstractmethod
	def is_terminal(self, state: StateT) -> bool:
		"""Return whether the position is terminal."""
		raise NotImplementedError

	@abstractmethod
	def terminal_score(
		self,
		state: StateT,
		player: PlayerT,
	) -> float:
		"""Return the terminal score from the specified player's perspective.

		Positive scores indicate a win, negative scores indicate a loss,
		and zero indicates a draw.

		Scores must be zero-sum between the two players.
		"""
		raise NotImplementedError

	@abstractmethod
	def position_hash(self, state: StateT) -> int:
		"""Return a hash identifying the complete game position.

		The hash must account for all state information relevant to
		future legal moves and position evaluation, including whose
		turn it is.
		"""
		raise NotImplementedError


class Evaluator(ABC, Generic[StateT, PlayerT]):
	"""Optional interface for game-specific position evaluation.

	Implementations evaluate nonterminal positions from the
	specified player's perspective.
	"""

	@abstractmethod
	def evaluate(
		self,
		state: StateT,
		player: PlayerT,
	) -> float:
		"""Return a heuristic score for the specified player."""
		raise NotImplementedError


class MoveOrdering(ABC, Generic[StateT, MoveT, PlayerT]):
	"""Optional interface for game-specific move ordering."""

	@abstractmethod
	def order_moves(
		self,
		state: StateT,
		moves: Sequence[MoveT],
		player: PlayerT,
	) -> Sequence[MoveT]:
		"""Return the supplied legal moves in preferred search order.

		Implementations must not add illegal moves or omit legal moves.
		"""
		raise NotImplementedError

