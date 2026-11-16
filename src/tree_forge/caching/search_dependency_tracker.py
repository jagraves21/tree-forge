from collections import defaultdict
from typing import Hashable


class SearchDependencyTracker:
	"""Tracks dependencies between transposition-table positions."""

	def __init__(self) -> None:
		self._children: dict[Hashable, set[Hashable]] = defaultdict(set)
		self._parents: dict[Hashable, set[Hashable]] = defaultdict(set)

	def register_dependency(
		self,
		parent_key: Hashable,
		child_key: Hashable,
	) -> None:
		"""Register that a parent's result depends on a child's result."""
		self._children[parent_key].add(child_key)
		self._parents[child_key].add(parent_key)

	def replace_dependencies(
		self,
		parent_key: Hashable,
		child_keys: set[Hashable],
	) -> None:
		"""Replace all dependencies currently registered for a parent."""
		self.remove_dependencies(parent_key)

		for child_key in child_keys:
			self.register_dependency(parent_key, child_key)

	def remove_dependencies(self, parent_key: Hashable) -> None:
		"""Remove all child dependencies registered for a parent."""
		children = self._children.pop(parent_key, set())

		for child_key in children:
			parents = self._parents.get(child_key)

			if parents is None:
				continue

			parents.discard(parent_key)

			if not parents:
				self._parents.pop(child_key, None)

	def get_ancestors(self, key: Hashable) -> set[Hashable]:
		"""Return all positions that transitively depend on a position."""
		ancestors: set[Hashable] = set()
		pending = list(self._parents.get(key, ()))

		while pending:
			parent_key = pending.pop()

			if parent_key in ancestors:
				continue

			ancestors.add(parent_key)
			pending.extend(self._parents.get(parent_key, ()))

		return ancestors

	def clear(self) -> None:
		"""Remove all tracked dependencies."""
		self._children.clear()
		self._parents.clear()

