from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, Hashable, TypeVar

from .models import CanonicalPosition, Outcome


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
	def outcome(self, state: StateT, player: PlayerT) -> Outcome:
		"""Return the outcome of a terminal position for a player.

		The outcome is relative to the supplied player: if the
		opponent has won, the outcome for the supplied player is
		Outcome.LOSS.

		Implementations only identify the result of the game. The
		numerical value of that result is decided by the Evaluator.
		"""
		raise NotImplementedError


class Evaluator(ABC, Generic[StateT, PlayerT]):
	"""Assigns numerical values to terminal and nonterminal positions.

	The evaluator is responsible for the value of every position,
	including wins, losses and draws. Search algorithms only compare
	the values it returns; they never decide what makes a result
	valuable.

	This class is also the default evaluator. Wins are worth positive
	infinity, losses negative infinity, and draws and nonterminal
	positions 0.0. Subclasses override whichever methods they need,
	for example to prefer faster wins, to value draws differently, or
	to evaluate nonterminal positions with game-specific heuristics.

	Values must be zero-sum: a position's value for one player is the
	negation of its value for the opponent.

	Values may depend on ply and depth, but search caches reuse
	positions reached at a different ply or depth. A cached value can
	therefore be inappropriate for the current search, and keeping
	ply- and depth-dependent values consistent is the responsibility
	of the evaluator's implementer.
	"""

	def evaluate(
		self,
		state: StateT,
		player: PlayerT,
		outcome: Outcome | None,
		ply: int,
		depth: int | None,
	) -> float:
		"""Return the value of a position for the specified player.

		state is the position being evaluated and player the player
		whose perspective is used. outcome is the position's outcome
		relative to that player, or None if it is not terminal. ply is
		the number of plies from the root of the current search to this
		position, and depth is the remaining search depth at it (None
		if the search has no depth limit).

		The default implementation dispatches on the outcome.
		"""
		if outcome == Outcome.WIN:
			return self.evaluate_win(state, player, ply, depth)

		if outcome == Outcome.LOSS:
			return self.evaluate_loss(state, player, ply, depth)

		if outcome == Outcome.DRAW:
			return self.evaluate_draw(state, player, ply, depth)

		return self.evaluate_non_terminal(state, player, ply, depth)

	def evaluate_win(
		self,
		state: StateT,
		player: PlayerT,
		ply: int,
		depth: int | None,
	) -> float:
		"""Return the value of a position the player has won."""
		return float("inf")

	def evaluate_loss(
		self,
		state: StateT,
		player: PlayerT,
		ply: int,
		depth: int | None,
	) -> float:
		"""Return the value of a position the player has lost."""
		return float("-inf")

	def evaluate_draw(
		self,
		state: StateT,
		player: PlayerT,
		ply: int,
		depth: int | None,
	) -> float:
		"""Return the value of a position that ended in a draw."""
		return 0.0

	def evaluate_non_terminal(
		self,
		state: StateT,
		player: PlayerT,
		ply: int,
		depth: int | None,
	) -> float:
		"""Return the value of a position that is not terminal."""
		return 0.0


class MoveOrdering(ABC, Generic[StateT, MoveT, PlayerT]):
	"""Ranks legal moves in their preferred search order.

	Implementations must return every supplied move exactly once.
	They must not add, omit, or modify moves.
	"""

	@abstractmethod
	def order_moves(
		self,
		state: StateT,
		moves: Sequence[MoveT],
		player: PlayerT,
	) -> Sequence[MoveT]:
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

