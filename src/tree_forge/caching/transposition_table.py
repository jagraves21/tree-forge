from __future__ import annotations

from collections.abc import Hashable, Set
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Generic, TypeVar

from .search_dependency_tracker import SearchDependencyTracker


MoveT = TypeVar("MoveT")


class TranspositionBound(Enum):
	"""Type of score information stored in a transposition entry."""

	EXACT = auto()
	LOWER = auto()
	UPPER = auto()


@dataclass(frozen=True, slots=True)
class TranspositionEntry(Generic[MoveT]):
	"""Cached information about a previously searched position.

	Depth represents the remaining search depth when the position
	was evaluated.

	A depth of None indicates that the position was searched without
	a depth limit.

	The solved flag indicates that the stored score information
	is proven independently of the search horizon.

	The score_valid flag indicates whether the cached score can
	still be reused. Invalidated entries retain their best move
	for move ordering.

	The score is interpreted by the search algorithm. The table
	does not impose a particular player perspective.
	"""

	key: Hashable
	depth: int | None
	score: float
	bound: TranspositionBound
	best_move: MoveT | None = None
	solved: bool = False
	score_valid: bool = True

	def __post_init__(self) -> None:
		if self.depth is not None and self.depth < 0:
			raise ValueError("depth must be non-negative or None")

	def is_usable_for_depth(self, depth: int | None) -> bool:
		"""Return whether this entry's score can be reused."""
		if not self.score_valid:
			return False

		if self.solved:
			return True

		if depth is None:
			return self.depth is None

		if self.depth is None:
			return True

		return self.depth >= depth


class TranspositionTable(Generic[MoveT]):
	"""Stores previously searched game positions.

	Entries are indexed by their position keys. The table does not
	generate keys or perform canonicalization; those responsibilities
	belong to the Canonicalizer.

	Dependency tracking is optional. When enabled, improvements to
	cached positions invalidate the scores of dependent, unsolved
	ancestor positions.
	"""

	def __init__(
		self,
		dependency_tracking: bool = False,
	) -> None:
		self._entries: dict[Hashable, TranspositionEntry[MoveT]] = {}
		self._dependency_tracker = (
			SearchDependencyTracker() if dependency_tracking else None
		)

	@property
	def dependency_tracking(self) -> bool:
		"""Return whether dependency tracking is enabled."""
		return self._dependency_tracker is not None

	def get(self, key: Hashable) -> TranspositionEntry[MoveT] | None:
		"""Return the entry for a position, if one exists.

		The entry is returned regardless of its search depth or
		score validity. Its best move may still be useful for
		move ordering even when its score has been invalidated.
		"""
		return self._entries.get(key)

	def register_dependency(
		self,
		parent_key: Hashable,
		child_key: Hashable,
	) -> None:
		"""Register that a parent's result depends on a child's result."""
		if self._dependency_tracker is None:
			return

		self._dependency_tracker.register_dependency(
			parent_key,
			child_key,
		)

	def replace_dependencies(
		self,
		parent_key: Hashable,
		child_keys: Set[Hashable],
	) -> None:
		"""Replace the dependencies registered for a parent position."""
		if self._dependency_tracker is None:
			return

		self._dependency_tracker.replace_dependencies(
			parent_key,
			set(child_keys),
		)

	def store(self, entry: TranspositionEntry[MoveT]) -> int:
		"""Store an entry when it improves the existing information.

		Solved entries take priority over unsolved entries.

		At equal depth, exact scores are preferred over bounds.
		For bounds of the same type, stronger bounds are preferred.

		When an entry materially improves, dependent unsolved
		ancestor scores are invalidated.

		Returns:
			The number of ancestor scores invalidated.
		"""
		existing = self._entries.get(entry.key)

		if existing is not None:
			if not self._is_improvement(entry, existing):
				return 0

			# Preserve the previous move-ordering hint when the
			# improved entry does not provide a new best move.
			if entry.best_move is None and existing.best_move is not None:
				entry = replace(
					entry,
					best_move=existing.best_move,
				)

		self._entries[entry.key] = entry

		return self._invalidate_ancestors(entry.key)

	def clear(self) -> None:
		"""Remove all entries and dependency relationships."""
		self._entries.clear()

		if self._dependency_tracker is not None:
			self._dependency_tracker.clear()

	def __len__(self) -> int:
		"""Return the number of stored positions."""
		return len(self._entries)

	def _invalidate_ancestors(self, key: Hashable) -> int:
		"""Invalidate cached scores that depend on an improved position.

		Solved ancestors are preserved because their results are
		independent of the search horizon.

		Returns the number of entries invalidated.
		"""
		if self._dependency_tracker is None:
			return 0

		invalidated = 0

		for ancestor_key in self._dependency_tracker.get_ancestors(key):
			if ancestor_key == key:
				continue

			ancestor = self._entries.get(ancestor_key)

			if ancestor is None or ancestor.solved:
				continue

			if not ancestor.score_valid:
				continue

			self._entries[ancestor_key] = replace(
				ancestor,
				score_valid=False,
			)

			invalidated += 1

		return invalidated

	@classmethod
	def _is_improvement(
		cls,
		new_entry: TranspositionEntry[MoveT],
		existing: TranspositionEntry[MoveT],
	) -> bool:
		"""Return whether new information improves an existing entry."""
		if not new_entry.score_valid:
			return False

		if not existing.score_valid:
			return True

		# A proven result takes priority over a horizon-limited result.
		if new_entry.solved and not existing.solved:
			return True

		if existing.solved and not new_entry.solved:
			return False

		# A deeper search provides newer information for an
		# unsolved position.
		if not new_entry.solved:
			if cls._has_greater_depth(
				new_entry.depth,
				existing.depth,
			):
				return True

			if new_entry.depth != existing.depth:
				return False

		# Prefer exact scores over bounds.
		if new_entry.bound == TranspositionBound.EXACT:
			if existing.bound != TranspositionBound.EXACT:
				return True

			# A changed exact score is new information.
			return new_entry.score != existing.score

		if existing.bound == TranspositionBound.EXACT:
			return False

		# Prefer stronger bounds of the same type.
		if new_entry.bound != existing.bound:
			return False

		if new_entry.bound == TranspositionBound.LOWER:
			return new_entry.score > existing.score

		if new_entry.bound == TranspositionBound.UPPER:
			return new_entry.score < existing.score

		return False

	@staticmethod
	def _has_greater_depth(
		new_depth: int | None,
		existing_depth: int | None,
	) -> bool:
		"""Return whether the new entry represents a deeper search."""
		if new_depth is None:
			return existing_depth is not None

		if existing_depth is None:
			return False

		return new_depth > existing_depth

