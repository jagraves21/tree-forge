from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Generic, TypeVar

from .search_dependency_tracker import SearchDependencyTracker


MoveT = TypeVar("MoveT")


class TranspositionBound(Enum):
	"""Kind of score information, used to describe an entry's interval."""

	EXACT = auto()
	LOWER = auto()
	UPPER = auto()


@dataclass(frozen=True, slots=True)
class TranspositionEntry(Generic[MoveT]):
	"""Cached information about a previously searched position.

	The entry describes the value of the position as an interval.
	lower and upper bound the value from below and above. A bound
	that is not known is None, so None never stands for zero or for an
	infinite bound. When both bounds are known and equal the value is
	exact. An entry without bounds carries no score information, only
	its best move.

	Scores are interpreted by the search algorithm, which stores them
	from the perspective of the player to move. The table does not
	impose a perspective, except that propagating improvements to
	dependent positions (see TranspositionTable) assumes that the
	perspective of a parent maximizes over its children.

	Depth is the remaining search depth when the position was
	evaluated, or None if it was searched without a depth limit.

	The solved flag indicates that the stored information is proven
	independently of the search horizon.

	The score_valid flag indicates whether the stored bounds can still
	be reused. Invalidated entries retain their best move for move
	ordering.
	"""

	key: Hashable
	depth: int | None
	lower: float | None = None
	upper: float | None = None
	best_move: MoveT | None = None
	solved: bool = False
	score_valid: bool = True

	def __post_init__(self) -> None:
		if self.depth is not None and self.depth < 0:
			raise ValueError("depth must be non-negative or None")

		if (
			self.lower is not None
			and self.upper is not None
			and self.lower > self.upper
		):
			raise ValueError("lower must not exceed upper")

	@classmethod
	def from_bound(
		cls,
		key: Hashable,
		depth: int | None,
		score: float,
		bound: TranspositionBound,
		best_move: MoveT | None = None,
		solved: bool = False,
		score_valid: bool = True,
	) -> TranspositionEntry[MoveT]:
		"""Create an entry from a score and the kind of bound it is."""
		return cls(
			key=key,
			depth=depth,
			lower=score if bound is not TranspositionBound.UPPER else None,
			upper=score if bound is not TranspositionBound.LOWER else None,
			best_move=best_move,
			solved=solved,
			score_valid=score_valid,
		)

	@property
	def is_exact(self) -> bool:
		"""Return whether the value of the position is known exactly."""
		return self.lower is not None and self.lower == self.upper

	@property
	def score(self) -> float | None:
		"""Return the exact value, or None if it is not known exactly."""
		return self.lower if self.is_exact else None

	def is_usable_for_depth(self, depth: int | None) -> bool:
		"""Return whether this entry's bounds can be reused."""
		if not self.score_valid:
			return False

		if self.solved:
			return True

		if depth is None:
			return self.depth is None

		if self.depth is None:
			return True

		return self.depth >= depth


@dataclass(frozen=True, slots=True)
class StoreOutcome:
	"""Result of storing an entry in a transposition table.

	stored tells whether the table changed. invalidated and refreshed
	count the previously stored positions whose entries were marked
	stale, or recomputed in place, because of the change. They are
	None when the table does not track dependencies, since nothing
	was measured.
	"""

	stored: bool
	invalidated: int | None = None
	refreshed: int | None = None


class _Derived:
	"""Information about a position derived from its continuations."""

	__slots__ = ("lower", "upper", "solved")

	def __init__(
		self,
		lower: float | None,
		upper: float | None,
		solved: bool,
	) -> None:
		self.lower = lower
		self.upper = upper
		self.solved = solved


class TranspositionTable(Generic[MoveT]):
	"""Stores previously searched game positions.

	Entries are indexed by their position keys. The table does not
	generate keys or perform canonicalization; those responsibilities
	belong to the Canonicalizer.

	Information about a position is merged, never discarded without
	reason: solved information beats unsolved information, deeper
	information beats shallower information, and information of equal
	depth is intersected. The merge is commutative and associative, so
	the stored information does not depend on the order in which
	searches discover it.

	Dependency tracking is optional. A dependency (parent, child)
	records that the parent's value is derived from the child, or may
	be improved by it. Dependencies are registered for all
	continuations of the parent, also for those that were not searched,
	and registering them recomputes the parent from the information
	that is already available. When the information stored for a child
	changes later, the change is propagated through every dependent position, directly and
	transitively, in an order that updates children before parents:

	- A solved dependent is skipped, since its value cannot change.
	- If the relations of the dependent's dependencies are known, its
	  bounds are recomputed from the current information about its
	  continuations. Information that is not available yet leaves the
	  corresponding bound unknown.
	- Otherwise the dependent's bounds are marked stale instead.
	- Propagation continues past a dependent only if its entry
	  changed, so positions that cannot be affected are not touched.

	A relation is +1 when the child's score has the same perspective
	as the parent's score and -1 when it has the opposite perspective
	and is negated. The parent's score is the maximum over its
	children of their scores, after applying the relations.
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
		relation: int | None = None,
	) -> None:
		"""Register that a parent's result depends on a child's result."""
		if self._dependency_tracker is None:
			return

		self._dependency_tracker.register_dependency(
			parent_key,
			child_key,
			relation,
		)

	def replace_dependencies(
		self,
		parent_key: Hashable,
		children: Iterable[Hashable] | Mapping[Hashable, int | None],
	) -> StoreOutcome:
		"""Replace the dependencies registered for a parent position.

		children is either the child keys, whose relations are then
		unknown, or a mapping from child keys to relations.

		Information about the children may already be stored, for
		example because they were searched before the parent. The
		parent's entry is therefore recomputed from it right away, so
		the result does not depend on whether the parent or its
		children were searched first. The returned outcome reports
		whether this changed the parent's entry (stored), and how many
		positions were refreshed or invalidated as a result.
		"""
		tracker = self._dependency_tracker

		if tracker is None:
			return StoreOutcome(False)

		if not isinstance(children, Mapping):
			children = list(children)

		tracker.replace_dependencies(parent_key, children)

		entry = self._entries.get(parent_key)

		if entry is None:
			return StoreOutcome(False, 0, 0)

		updated = self._refreshed(parent_key, entry)

		if updated == entry:
			return StoreOutcome(False, 0, 0)

		self._entries[parent_key] = updated

		invalidated, refreshed = self._propagate(parent_key)

		return StoreOutcome(True, invalidated, refreshed + 1)

	def store(self, entry: TranspositionEntry[MoveT]) -> StoreOutcome:
		"""Merge an entry into the table.

		Returns:
			Whether the table changed and, when dependencies are
			tracked, how many dependent positions were invalidated
			or refreshed as a result.
		"""
		existing = self._entries.get(entry.key)
		merged = entry if existing is None else self._merge(existing, entry)

		# Continuations may already be known, so the stored information is
		# completed with them. This way it does not matter whether the
		# position or its continuations were stored first.
		completed = self._refreshed(entry.key, merged)
		refreshes = 1 if completed != merged else 0
		merged = completed

		if merged == existing:
			return self._outcome(False, 0, 0)

		self._entries[entry.key] = merged

		# A deeper search of the same information still matters: it can make
		# the entry usable for dependents that needed more depth than it had.
		information_changed = existing is None or self._information(
			existing
		) != self._information(merged)

		invalidated, refreshed = self._propagate(
			entry.key,
			information_changed,
		)

		return self._outcome(True, invalidated, refreshed + refreshes)

	def clear(self) -> None:
		"""Remove all entries and dependency relationships."""
		self._entries.clear()

		if self._dependency_tracker is not None:
			self._dependency_tracker.clear()

	def __len__(self) -> int:
		"""Return the number of stored positions."""
		return len(self._entries)

	def _outcome(
		self,
		stored: bool,
		invalidated: int,
		refreshed: int,
	) -> StoreOutcome:
		if self._dependency_tracker is None:
			return StoreOutcome(stored)

		return StoreOutcome(stored, invalidated, refreshed)

	@staticmethod
	def _information(entry: TranspositionEntry[MoveT]) -> tuple:
		"""Return what dependent positions can observe about an entry."""
		return entry.lower, entry.upper, entry.solved, entry.score_valid

	def _propagate(
		self,
		key: Hashable,
		information_changed: bool = True,
	) -> tuple[int, int]:
		"""Update every position that depends on a changed position.

		information_changed tells whether the bounds of the changed
		position changed, as opposed to only its search depth. Positions
		whose relations are unknown can only be marked stale in the
		first case, because nothing else about the value changed.
		"""
		tracker = self._dependency_tracker

		if tracker is None:
			return 0, 0

		changed = {key}
		informed = {key} if information_changed else set()
		invalidated = 0
		refreshed = 0

		for ancestor_key in tracker.get_update_order(key):
			children = tracker.get_children(ancestor_key)

			if changed.isdisjoint(children):
				continue

			entry = self._entries.get(ancestor_key)

			# Solved values cannot change and stale values were already
			# reported, so nothing further depends on a change here.
			if entry is None or entry.solved or not entry.score_valid:
				continue

			derived = self._derive(entry, children)

			if derived is None:
				if informed.isdisjoint(children):
					continue

				self._entries[ancestor_key] = replace(
					entry,
					score_valid=False,
				)
				invalidated += 1
				changed.add(ancestor_key)
				informed.add(ancestor_key)

				continue

			updated = self._refresh(entry, derived)

			if updated != entry:
				self._entries[ancestor_key] = updated
				refreshed += 1
				changed.add(ancestor_key)
				informed.add(ancestor_key)

		return invalidated, refreshed

	def _refreshed(
		self,
		key: Hashable,
		entry: TranspositionEntry[MoveT],
	) -> TranspositionEntry[MoveT]:
		"""Complete an entry with what its continuations currently say.

		Entries that are solved or stale, positions without known
		continuations, and continuations with unknown relations are
		left as they are.
		"""
		tracker = self._dependency_tracker

		if tracker is None or entry.solved or not entry.score_valid:
			return entry

		derived = self._derive(entry, tracker.get_children(key))

		if derived is None:
			return entry

		return self._refresh(entry, derived)

	def _derive(
		self,
		entry: TranspositionEntry[MoveT],
		children: Mapping[Hashable, int | None],
	) -> _Derived | None:
		"""Derive information about a position from its continuations.

		Returns None if the relations of the continuations are not
		known, in which case nothing can be derived.
		"""
		if not children or any(
			relation not in (1, -1) for relation in children.values()
		):
			return None

		if entry.depth is not None and entry.depth < 1:
			return None

		needed_depth = None if entry.depth is None else entry.depth - 1

		lowers: list[float] = []
		uppers: list[float] = []
		complete = True
		solved = True

		for child_key, relation in children.items():
			child = self._entries.get(child_key)

			if child is None or not child.is_usable_for_depth(needed_depth):
				complete = False
				solved = False

				continue

			if relation == 1:
				lower, upper = child.lower, child.upper
			else:
				lower = None if child.upper is None else -child.upper + 0.0
				upper = None if child.lower is None else -child.lower + 0.0

			if lower is not None:
				lowers.append(lower)

			if upper is None:
				complete = False
			else:
				uppers.append(upper)

			if not (child.solved and child.is_exact):
				solved = False

		# The best continuation is at least as good as any continuation
		# whose lower bound is known, but its upper bound is only known
		# once every continuation has an upper bound.
		return _Derived(
			max(lowers) if lowers else None,
			max(uppers) if complete else None,
			solved,
		)

	@classmethod
	def _refresh(
		cls,
		entry: TranspositionEntry[MoveT],
		derived: _Derived,
	) -> TranspositionEntry[MoveT]:
		"""Combine an entry with information derived from its children."""
		if derived.solved:
			return replace(
				entry,
				lower=derived.lower,
				upper=derived.upper,
				solved=True,
			)

		if derived.lower is None and derived.upper is None:
			return entry

		intersection = cls._intersect(
			entry.lower,
			entry.upper,
			derived.lower,
			derived.upper,
		)

		# Information that contradicts the stored bounds is more recent
		# than them, since it reflects the current continuations.
		lower, upper = (
			intersection
			if intersection is not None
			else (derived.lower, derived.upper)
		)

		return replace(entry, lower=lower, upper=upper)

	@classmethod
	def _merge(
		cls,
		existing: TranspositionEntry[MoveT],
		new: TranspositionEntry[MoveT],
	) -> TranspositionEntry[MoveT]:
		"""Merge new information into the existing entry for a position.

		Returns the existing entry itself when the new entry adds
		nothing, so that rejected entries cannot alter it.
		"""
		if not new.score_valid:
			return existing

		if not existing.score_valid:
			return cls._with_move_hint(new, existing)

		# Proven information outranks horizon-limited information.
		if new.solved != existing.solved:
			return (
				cls._with_move_hint(new, existing)
				if new.solved
				else existing
			)

		depth = existing.depth

		# Deeper information about an unsolved position outranks
		# shallower information.
		if not new.solved and new.depth != existing.depth:
			if cls._is_deeper(new.depth, existing.depth):
				return cls._with_move_hint(new, existing)

			return existing

		if new.solved and new.depth != existing.depth:
			depth = (
				new.depth
				if cls._is_deeper(new.depth, existing.depth)
				else existing.depth
			)

		intersection = cls._intersect(
			existing.lower,
			existing.upper,
			new.lower,
			new.upper,
		)

		if intersection is None:
			# Contradictory information: keep the more precise one and,
			# if both are equally precise, the more recent one.
			if cls._precision(new) >= cls._precision(existing):
				return cls._with_move_hint(new, existing)

			return existing

		lower, upper = intersection

		if (lower, upper) == (existing.lower, existing.upper):
			if depth == existing.depth:
				return existing

			return replace(existing, depth=depth)

		return cls._with_move_hint(
			replace(new, depth=depth, lower=lower, upper=upper),
			existing,
		)

	@staticmethod
	def _with_move_hint(
		entry: TranspositionEntry[MoveT],
		previous: TranspositionEntry[MoveT],
	) -> TranspositionEntry[MoveT]:
		"""Keep the previous move-ordering hint if the entry has none."""
		if entry.best_move is None and previous.best_move is not None:
			return replace(entry, best_move=previous.best_move)

		return entry

	@staticmethod
	def _precision(entry: TranspositionEntry[MoveT]) -> int:
		return (entry.lower is not None) + (entry.upper is not None)

	@staticmethod
	def _intersect(
		first_lower: float | None,
		first_upper: float | None,
		second_lower: float | None,
		second_upper: float | None,
	) -> tuple[float | None, float | None] | None:
		"""Intersect two intervals, or return None if they are disjoint."""
		lowers = [v for v in (first_lower, second_lower) if v is not None]
		uppers = [v for v in (first_upper, second_upper) if v is not None]

		lower = max(lowers) if lowers else None
		upper = min(uppers) if uppers else None

		if lower is not None and upper is not None and lower > upper:
			return None

		return lower, upper

	@staticmethod
	def _is_deeper(new_depth: int | None, existing_depth: int | None) -> bool:
		"""Return whether the new depth represents a deeper search."""
		if new_depth is None:
			return existing_depth is not None

		if existing_depth is None:
			return False

		return new_depth > existing_depth
