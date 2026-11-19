from collections.abc import Hashable, Iterable, Mapping


class SearchDependencyTracker:
	"""Tracks which positions' results depend on which other positions.

	A dependency (parent, child) means that the parent's result is
	derived from, or may be improved by, information about the child.
	It is registered for every continuation of the parent, including
	continuations that were not searched, because information about
	those continuations may only become available through later
	searches.

	A dependency may carry a relation describing how the child's
	result contributes to the parent's result. The tracker stores
	relations without interpreting them. A relation of None means
	that the relation is unknown.
	"""

	def __init__(self) -> None:
		self._children: dict[Hashable, dict[Hashable, Hashable | None]] = {}
		self._parents: dict[Hashable, set[Hashable]] = {}

	def register_dependency(
		self,
		parent_key: Hashable,
		child_key: Hashable,
		relation: Hashable | None = None,
	) -> None:
		"""Register that a parent's result depends on a child's result.

		Registering an existing dependency replaces its relation.
		"""
		self._children.setdefault(parent_key, {})[child_key] = relation
		self._parents.setdefault(child_key, set()).add(parent_key)

	def replace_dependencies(
		self,
		parent_key: Hashable,
		children: Iterable[Hashable] | Mapping[Hashable, Hashable | None],
	) -> None:
		"""Replace all dependencies currently registered for a parent.

		children is either the child keys, whose relations are then
		unknown, or a mapping from child keys to relations.
		"""
		if isinstance(children, Mapping):
			relations = dict(children)
		else:
			relations = dict.fromkeys(children)

		self.remove_dependencies(parent_key)

		for child_key, relation in relations.items():
			self.register_dependency(parent_key, child_key, relation)

	def remove_dependencies(self, parent_key: Hashable) -> None:
		"""Remove all child dependencies registered for a parent."""
		children = self._children.pop(parent_key, {})

		for child_key in children:
			parents = self._parents.get(child_key)

			if parents is None:
				continue

			parents.discard(parent_key)

			if not parents:
				del self._parents[child_key]

	def get_children(
		self,
		parent_key: Hashable,
	) -> dict[Hashable, Hashable | None]:
		"""Return the children of a parent with their relations."""
		return dict(self._children.get(parent_key, {}))

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

	def get_update_order(self, key: Hashable) -> list[Hashable]:
		"""Return the ancestors of a position in update order.

		Every ancestor appears after the ancestors it depends on, so
		a result can be recomputed once from already updated children.
		The position itself is not included. Cyclic dependencies are
		broken arbitrarily, and every position appears exactly once,
		so processing the order always terminates.
		"""
		visited: set[Hashable] = {key}
		finished: list[Hashable] = []

		# Depth-first search along parent edges, iteratively so that
		# deep dependency chains cannot overflow the call stack.
		stack = [(key, iter(self._parents.get(key, ())))]

		while stack:
			current, remaining = stack[-1]

			for parent_key in remaining:
				if parent_key not in visited:
					visited.add(parent_key)
					stack.append(
						(parent_key, iter(self._parents.get(parent_key, ())))
					)

					break
			else:
				stack.pop()
				finished.append(current)

		# The search finishes the most distant ancestors first, so the
		# reverse puts every position after the ones it depends on.
		finished.pop()  # the position itself finished last

		return finished[::-1]

	def clear(self) -> None:
		"""Remove all tracked dependencies."""
		self._children.clear()
		self._parents.clear()
