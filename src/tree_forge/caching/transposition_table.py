from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, TypeVar
from collections.abc import Hashable


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

	A depth of None indicates that the position was searched all
	the way to a terminal state.

	The score is interpreted by the search algorithm. The table
	does not impose a particular player perspective.
	"""

	key: Hashable
	depth: int | None
	score: float
	bound: TranspositionBound
	best_move: MoveT | None = None

	def __post_init__(self) -> None:
		if self.depth is not None and self.depth < 0:
			raise ValueError("depth must be non-negative or None")

	def is_usable_for_depth(self, depth: int | None) -> bool:
		"""Return whether this entry has sufficient search depth.

		An entry searched to terminal depth (None) can satisfy any
		finite depth request.

		A finite-depth entry cannot satisfy an unlimited-depth
		request.
		"""
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

	Entries persist until explicitly replaced or the table is cleared.
	"""

	def __init__(self) -> None:
		self._entries: dict[Hashable, TranspositionEntry[MoveT]] = {}

	def get(self, key: Hashable) -> TranspositionEntry[MoveT] | None:
		"""Return the entry for a position, if one exists.

		The entry is returned regardless of its search depth. Call
		is_usable_for_depth() to determine whether its score can be
		reused. Its best move may still be useful for move ordering.
		"""
		return self._entries.get(key)

	def store(self, entry: TranspositionEntry[MoveT]) -> None:
		"""Store an entry, retaining the more useful existing entry.

		An entry replaces an existing entry when it was searched
		deeper, or when it has the same depth and provides better
		score information.

		At equal depth, an exact score is preferred over a bound.
		"""
		existing = self._entries.get(entry.key)

		if existing is None:
			self._entries[entry.key] = entry
			return

		if self._has_greater_depth(entry.depth, existing.depth):
			self._entries[entry.key] = entry
			return

		if entry.depth != existing.depth:
			return

		if (
			entry.bound == TranspositionBound.EXACT
			or existing.bound != TranspositionBound.EXACT
		):
			self._entries[entry.key] = entry

	def clear(self) -> None:
		"""Remove all entries from the table."""
		self._entries.clear()

	def __len__(self) -> int:
		"""Return the number of stored positions."""
		return len(self._entries)

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

