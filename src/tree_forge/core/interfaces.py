from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, Hashable, TypeVar

from .models import CanonicalPosition


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
	def unmake_move(self, state: StateT, move: MoveT, undo_data: UndoT) -> None:
		"""Restore the state to exactly what it was before make_move()."""
		raise NotImplementedError

	@abstractmethod
	def is_terminal(self, state: StateT) -> bool:
		"""Return whether the position is terminal."""
		raise NotImplementedError

	@abstractmethod
	def terminal_score(self, state: StateT, player: PlayerT) -> float:
		"""Return the terminal score from the specified player's perspective.

		Positive scores indicate a win, negative scores indicate a loss,
		and zero indicates a draw.

		Scores must be zero-sum between the two players.
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


class Canonicalizer(ABC, Generic[StateT, MoveT]):
	"""Interface for identifying and canonicalizing game positions.

	Canonicalization is optional. The default implementations use
	the exact position without applying any symmetries.

	Implementations may override canonical_position_key() and
	canonicalize() to identify equivalent positions under supported
	game-specific symmetries.
	"""

	@abstractmethod
	def position_key(self, state: StateT) -> Hashable:
		"""Return a representation of the exact game position.

		The key must account for all state information relevant to
		future legal moves and position evaluation, including whose
		turn it is.

		Positions that differ in any relevant state information must
		produce different keys.
		"""
		raise NotImplementedError

	def position_hash(self, state: StateT) -> int:
		"""Return a hash of the exact game position.

		By default, this hashes the value returned by position_key().
		"""
		return hash(self.position_key(state))

	def canonical_position_key(self, state: StateT) -> Hashable:
		"""Return the canonical key of a game position.

		Equivalent positions under supported symmetries must produce
		the same key.

		The default implementation returns the exact position key,
		without applying any symmetries.
		"""
		return self.position_key(state)

	def canonical_position_hash(self, state: StateT) -> int:
		"""Return a hash of the canonical game position.

		By default, this hashes the value returned by
		canonical_position_key().
		"""
		return hash(self.canonical_position_key(state))

	def canonicalize(self, state: StateT) -> CanonicalPosition[MoveT]:
		"""Return the canonical position and its move transformations.

		The returned key identifies the canonical position.
		to_canonical_move() converts moves from the original position
		to canonical coordinates.
		from_canonical_move() converts moves from canonical coordinates
		back to the original position.

		The default implementation uses the exact position key and
		identity move transformations, without applying symmetries.
		"""
		return CanonicalPosition(
			key=self.canonical_position_key(state),
			to_canonical_move=lambda move: move,
			from_canonical_move=lambda move: move,
		)

