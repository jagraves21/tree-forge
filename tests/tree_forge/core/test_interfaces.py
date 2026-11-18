from collections.abc import Hashable, Sequence

import pytest

from tree_forge.core.interfaces import (
	Canonicalizer,
	Evaluator,
	Game,
	MoveOrdering,
)
from tree_forge.core.models import CanonicalPosition


def stub(self, *args, **kwargs):
	"""Placeholder implementation for an abstract method."""
	return None


def make_partial_subclass(base: type, missing: str) -> type:
	"""Create a subclass that implements every abstract method but one."""
	namespace = {
		name: stub
		for name in base.__abstractmethods__
		if name != missing
	}

	return type("PartialImplementation", (base,), namespace)


def make_complete_subclass(base: type) -> type:
	"""Create a subclass that implements every abstract method."""
	namespace = {name: stub for name in base.__abstractmethods__}

	return type("CompleteImplementation", (base,), namespace)


class ListCanonicalizer(Canonicalizer[list[int], int]):
	"""Canonicalizer that only implements the required position_key()."""

	def position_key(self, state: list[int]) -> Hashable:
		return tuple(state), "X"


class SortingCanonicalizer(ListCanonicalizer):
	"""Canonicalizer that treats permutations of a state as equivalent."""

	def canonical_position_key(self, state: list[int]) -> Hashable:
		return tuple(sorted(state))


class TestGame:
	REQUIRED_METHODS = {
		"initial_state": (),
		"current_player": ("state",),
		"legal_moves": ("state",),
		"make_move": ("state", "move"),
		"unmake_move": ("state", "move", "undo_data"),
		"is_terminal": ("state",),
		"terminal_score": ("state", "player"),
	}

	def test_declares_expected_abstract_methods(self) -> None:
		assert Game.__abstractmethods__ == frozenset(self.REQUIRED_METHODS)

	def test_cannot_be_instantiated_directly(self) -> None:
		with pytest.raises(TypeError):
			Game()

	@pytest.mark.parametrize("missing", sorted(REQUIRED_METHODS))
	def test_cannot_be_instantiated_without_required_method(
		self,
		missing: str,
	) -> None:
		partial = make_partial_subclass(Game, missing)

		with pytest.raises(TypeError, match=missing):
			partial()

	def test_can_be_instantiated_when_complete(self) -> None:
		game = make_complete_subclass(Game)()

		assert isinstance(game, Game)

	@pytest.mark.parametrize(
		"method, arguments",
		sorted(REQUIRED_METHODS.items()),
	)
	def test_base_methods_raise_not_implemented_error(
		self,
		method: str,
		arguments: tuple[str, ...],
	) -> None:
		game = make_complete_subclass(Game)()

		with pytest.raises(NotImplementedError):
			getattr(Game, method)(game, *arguments)


class TestEvaluator:
	def test_declares_expected_abstract_methods(self) -> None:
		assert Evaluator.__abstractmethods__ == frozenset({"evaluate"})

	def test_cannot_be_instantiated_directly(self) -> None:
		with pytest.raises(TypeError):
			Evaluator()

	def test_cannot_be_instantiated_without_evaluate(self) -> None:
		partial = make_partial_subclass(Evaluator, "evaluate")

		with pytest.raises(TypeError, match="evaluate"):
			partial()

	def test_can_be_instantiated_when_complete(self) -> None:
		evaluator = make_complete_subclass(Evaluator)()

		assert isinstance(evaluator, Evaluator)

	def test_base_evaluate_raises_not_implemented_error(self) -> None:
		evaluator = make_complete_subclass(Evaluator)()

		with pytest.raises(NotImplementedError):
			Evaluator.evaluate(evaluator, "state", "player")


class TestMoveOrdering:
	def test_declares_expected_abstract_methods(self) -> None:
		assert MoveOrdering.__abstractmethods__ == frozenset({"order_moves"})

	def test_cannot_be_instantiated_directly(self) -> None:
		with pytest.raises(TypeError):
			MoveOrdering()

	def test_cannot_be_instantiated_without_order_moves(self) -> None:
		partial = make_partial_subclass(MoveOrdering, "order_moves")

		with pytest.raises(TypeError, match="order_moves"):
			partial()

	def test_can_be_instantiated_when_complete(self) -> None:
		ordering = make_complete_subclass(MoveOrdering)()

		assert isinstance(ordering, MoveOrdering)

	def test_base_order_moves_raises_not_implemented_error(self) -> None:
		ordering = make_complete_subclass(MoveOrdering)()
		moves: Sequence[int] = [1, 2, 3]

		with pytest.raises(NotImplementedError):
			MoveOrdering.order_moves(ordering, "state", moves, "player")


class TestCanonicalizerAbstractInterface:
	def test_declares_only_position_key_as_abstract(self) -> None:
		assert Canonicalizer.__abstractmethods__ == frozenset({"position_key"})

	def test_cannot_be_instantiated_directly(self) -> None:
		with pytest.raises(TypeError):
			Canonicalizer()

	def test_cannot_be_instantiated_without_position_key(self) -> None:
		partial = make_partial_subclass(Canonicalizer, "position_key")

		with pytest.raises(TypeError, match="position_key"):
			partial()

	def test_can_be_instantiated_with_only_position_key(self) -> None:
		canonicalizer = ListCanonicalizer()

		assert isinstance(canonicalizer, Canonicalizer)

	def test_base_position_key_raises_not_implemented_error(self) -> None:
		canonicalizer = ListCanonicalizer()

		with pytest.raises(NotImplementedError):
			Canonicalizer.position_key(canonicalizer, [1, 2])


class TestCanonicalizerDefaults:
	@pytest.fixture
	def canonicalizer(self) -> ListCanonicalizer:
		return ListCanonicalizer()

	def test_position_hash_hashes_position_key(
		self,
		canonicalizer: ListCanonicalizer,
	) -> None:
		state = [1, 2]

		assert canonicalizer.position_hash(state) == hash(
			canonicalizer.position_key(state)
		)

	def test_position_hash_is_stable_for_equal_states(
		self,
		canonicalizer: ListCanonicalizer,
	) -> None:
		assert canonicalizer.position_hash([1, 2]) == (
			canonicalizer.position_hash([1, 2])
		)

	def test_position_hash_requires_hashable_key(self) -> None:
		class UnhashableKeyCanonicalizer(Canonicalizer[list[int], int]):
			def position_key(self, state: list[int]) -> Hashable:
				return list(state)

		with pytest.raises(TypeError):
			UnhashableKeyCanonicalizer().position_hash([1])

	def test_canonical_position_key_defaults_to_position_key(
		self,
		canonicalizer: ListCanonicalizer,
	) -> None:
		state = [1, 2]

		assert canonicalizer.canonical_position_key(state) == (
			canonicalizer.position_key(state)
		)

	def test_canonical_position_hash_hashes_canonical_key(
		self,
		canonicalizer: ListCanonicalizer,
	) -> None:
		state = [1, 2]

		assert canonicalizer.canonical_position_hash(state) == hash(
			canonicalizer.canonical_position_key(state)
		)

	def test_canonicalize_returns_canonical_position(
		self,
		canonicalizer: ListCanonicalizer,
	) -> None:
		position = canonicalizer.canonicalize([1, 2])

		assert isinstance(position, CanonicalPosition)

	def test_canonicalize_uses_canonical_position_key(
		self,
		canonicalizer: ListCanonicalizer,
	) -> None:
		state = [1, 2]

		assert canonicalizer.canonicalize(state).key == (
			canonicalizer.canonical_position_key(state)
		)

	def test_canonicalize_defaults_to_identity_move_transformations(
		self,
		canonicalizer: ListCanonicalizer,
	) -> None:
		position = canonicalizer.canonicalize([1, 2])
		move = object()

		assert position.to_canonical_move(move) is move
		assert position.from_canonical_move(move) is move


class TestCanonicalizerOverrides:
	"""Defaults must dispatch to overridden methods."""

	@pytest.fixture
	def canonicalizer(self) -> SortingCanonicalizer:
		return SortingCanonicalizer()

	def test_exact_position_key_is_unaffected_by_canonical_override(
		self,
		canonicalizer: SortingCanonicalizer,
	) -> None:
		assert canonicalizer.position_key([2, 1]) == ((2, 1), "X")

	def test_canonical_position_hash_uses_overridden_canonical_key(
		self,
		canonicalizer: SortingCanonicalizer,
	) -> None:
		assert canonicalizer.canonical_position_hash([2, 1]) == hash((1, 2))

	def test_canonicalize_uses_overridden_canonical_key(
		self,
		canonicalizer: SortingCanonicalizer,
	) -> None:
		assert canonicalizer.canonicalize([2, 1]).key == (1, 2)

	def test_equivalent_states_share_canonical_identity(
		self,
		canonicalizer: SortingCanonicalizer,
	) -> None:
		first = [1, 2, 3]
		second = [3, 1, 2]

		assert canonicalizer.position_key(first) != (
			canonicalizer.position_key(second)
		)
		assert canonicalizer.canonical_position_key(first) == (
			canonicalizer.canonical_position_key(second)
		)
		assert canonicalizer.canonical_position_hash(first) == (
			canonicalizer.canonical_position_hash(second)
		)
