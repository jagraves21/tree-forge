from collections.abc import Hashable, Sequence

import inspect
from math import inf

import pytest

from tree_forge.core.interfaces import (
	Canonicalizer,
	Evaluator,
	Game,
	MoveOrdering,
)
from tree_forge.core.models import CanonicalPosition, Outcome


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
		"outcome": ("state", "player"),
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


class RecordingEvaluator(Evaluator[str, int]):
	"""Evaluator whose terminal and nonterminal methods are all overridden."""

	def __init__(self) -> None:
		self.calls: list[tuple] = []

	def evaluate_win(self, state, player, ply, depth) -> float:
		self.calls.append(("win", state, player, ply, depth))

		return 10.0

	def evaluate_loss(self, state, player, ply, depth) -> float:
		self.calls.append(("loss", state, player, ply, depth))

		return -10.0

	def evaluate_draw(self, state, player, ply, depth) -> float:
		self.calls.append(("draw", state, player, ply, depth))

		return 1.5

	def evaluate_non_terminal(self, state, player, ply, depth) -> float:
		self.calls.append(("non_terminal", state, player, ply, depth))

		return -0.5


class TestEvaluator:
	METHODS = (
		"evaluate_win",
		"evaluate_loss",
		"evaluate_draw",
		"evaluate_non_terminal",
	)

	def test_has_no_abstract_methods(self) -> None:
		assert Evaluator.__abstractmethods__ == frozenset()

	def test_can_be_instantiated_directly_as_the_default_evaluator(
		self,
	) -> None:
		assert isinstance(Evaluator(), Evaluator)

	def test_evaluate_takes_the_state_player_outcome_ply_and_depth(
		self,
	) -> None:
		parameters = inspect.signature(Evaluator.evaluate).parameters

		assert list(parameters) == [
			"self",
			"state",
			"player",
			"outcome",
			"ply",
			"depth",
		]

	@pytest.mark.parametrize("method", METHODS)
	def test_outcome_methods_take_the_state_player_ply_and_depth(
		self,
		method: str,
	) -> None:
		parameters = inspect.signature(getattr(Evaluator, method)).parameters

		assert list(parameters) == ["self", "state", "player", "ply", "depth"]

	@pytest.mark.parametrize("outcome, expected", [
		pytest.param(Outcome.WIN, inf, id="win"),
		pytest.param(Outcome.LOSS, -inf, id="loss"),
		pytest.param(Outcome.DRAW, 0.0, id="draw"),
		pytest.param(None, 0.0, id="non-terminal"),
	])
	@pytest.mark.parametrize("ply, depth", [(0, 0), (3, 5), (9, None)])
	def test_default_values(
		self,
		outcome: Outcome | None,
		expected: float,
		ply: int,
		depth: int | None,
	) -> None:
		value = Evaluator().evaluate("state", 1, outcome, ply, depth)

		assert value == expected
		assert isinstance(value, float)

	@pytest.mark.parametrize("method, expected", [
		pytest.param("evaluate_win", inf, id="win"),
		pytest.param("evaluate_loss", -inf, id="loss"),
		pytest.param("evaluate_draw", 0.0, id="draw"),
		pytest.param("evaluate_non_terminal", 0.0, id="non-terminal"),
	])
	def test_default_methods_return_the_default_values(
		self,
		method: str,
		expected: float,
	) -> None:
		assert getattr(Evaluator(), method)("state", 1, 2, 3) == expected

	def test_default_win_and_loss_are_zero_sum(self) -> None:
		evaluator = Evaluator()

		assert evaluator.evaluate("s", 1, Outcome.WIN, 0, 1) == -(
			evaluator.evaluate("s", 2, Outcome.LOSS, 0, 1)
		)

	@pytest.mark.parametrize("outcome, label", [
		pytest.param(Outcome.WIN, "win", id="win"),
		pytest.param(Outcome.LOSS, "loss", id="loss"),
		pytest.param(Outcome.DRAW, "draw", id="draw"),
		pytest.param(None, "non_terminal", id="non-terminal"),
	])
	def test_evaluate_dispatches_on_the_outcome(
		self,
		outcome: Outcome | None,
		label: str,
	) -> None:
		evaluator = RecordingEvaluator()

		evaluator.evaluate("state", 2, outcome, 4, 6)

		assert evaluator.calls == [(label, "state", 2, 4, 6)]

	def test_evaluate_returns_the_value_of_the_overridden_method(self) -> None:
		evaluator = RecordingEvaluator()

		assert evaluator.evaluate("s", 1, Outcome.WIN, 0, 1) == 10.0
		assert evaluator.evaluate("s", 1, Outcome.LOSS, 0, 1) == -10.0
		assert evaluator.evaluate("s", 1, Outcome.DRAW, 0, 1) == 1.5
		assert evaluator.evaluate("s", 1, None, 0, 1) == -0.5

	@pytest.mark.parametrize("overridden", METHODS)
	def test_overriding_one_method_keeps_the_other_defaults(
		self,
		overridden: str,
	) -> None:
		def override(self, state, player, ply, depth) -> float:
			return 99.0

		custom = type("Custom", (Evaluator,), {overridden: override})()
		outcomes = {
			"evaluate_win": Outcome.WIN,
			"evaluate_loss": Outcome.LOSS,
			"evaluate_draw": Outcome.DRAW,
			"evaluate_non_terminal": None,
		}

		for method, outcome in outcomes.items():
			expected = (
				99.0
				if method == overridden
				else getattr(Evaluator(), method)("s", 1, 0, 1)
			)

			assert custom.evaluate("s", 1, outcome, 0, 1) == expected

	def test_values_may_depend_on_ply_and_depth(self) -> None:
		class Sooner(Evaluator[str, int]):
			def evaluate_win(self, state, player, ply, depth) -> float:
				return 100.0 - ply

			def evaluate_loss(self, state, player, ply, depth) -> float:
				return -100.0 + ply

			def evaluate_non_terminal(self, state, player, ply, depth) -> float:
				return float(depth)

		evaluator = Sooner()

		assert evaluator.evaluate("s", 1, Outcome.WIN, 3, 0) == 97.0
		assert evaluator.evaluate("s", 1, Outcome.LOSS, 3, 0) == -97.0
		assert evaluator.evaluate("s", 1, None, 3, 4) == 4.0

	def test_state_and_player_reach_the_overridden_methods(self) -> None:
		class StateValue(Evaluator[dict, str]):
			def evaluate_non_terminal(self, state, player, ply, depth) -> float:
				return state["value"] if player == "X" else -state["value"]

		evaluator = StateValue()

		assert evaluator.evaluate({"value": 3.0}, "X", None, 0, 1) == 3.0
		assert evaluator.evaluate({"value": 3.0}, "O", None, 0, 1) == -3.0


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
