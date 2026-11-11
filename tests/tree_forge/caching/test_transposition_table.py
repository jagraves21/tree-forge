import pytest

from tree_forge.caching import (
	TranspositionBound,
	TranspositionEntry,
	TranspositionTable,
)


@pytest.fixture
def table():
	return TranspositionTable()


def make_entry(
	key="position",
	depth=4,
	score=3.0,
	bound=TranspositionBound.EXACT,
	best_move="move",
):
	return TranspositionEntry(
		key=key,
		depth=depth,
		score=score,
		bound=bound,
		best_move=best_move,
	)


def test_store_and_retrieve_entry(table):
	entry = make_entry()

	table.store(entry)

	assert table.get("position") == entry
	assert len(table) == 1


def test_get_returns_none_for_missing_entry(table):
	assert table.get("missing") is None


def test_entry_is_usable_at_equal_depth():
	entry = make_entry(depth=4)

	assert entry.is_usable_for_depth(4)


def test_entry_is_usable_at_shallower_requested_depth():
	entry = make_entry(depth=5)

	assert entry.is_usable_for_depth(3)


def test_entry_is_not_usable_at_deeper_requested_depth():
	entry = make_entry(depth=3)

	assert not entry.is_usable_for_depth(5)


def test_unlimited_depth_entry_is_usable_at_finite_depth():
	entry = make_entry(depth=None)

	assert entry.is_usable_for_depth(1)
	assert entry.is_usable_for_depth(4)
	assert entry.is_usable_for_depth(10)


def test_finite_depth_entry_is_not_usable_at_unlimited_depth():
	entry = make_entry(depth=10)

	assert not entry.is_usable_for_depth(None)


def test_unlimited_depth_entry_is_usable_at_unlimited_depth():
	entry = make_entry(depth=None)

	assert entry.is_usable_for_depth(None)


def test_entry_rejects_negative_depth():
	with pytest.raises(ValueError, match="depth must be non-negative or None"):
		make_entry(depth=-1)


def test_deeper_entry_replaces_shallower_entry(table):
	shallow_entry = make_entry(depth=2, score=1.0)
	deep_entry = make_entry(depth=5, score=3.0)

	table.store(shallow_entry)
	table.store(deep_entry)

	assert table.get("position") == deep_entry


def test_shallower_entry_does_not_replace_deeper_entry(table):
	deep_entry = make_entry(depth=5, score=3.0)
	shallow_entry = make_entry(depth=2, score=1.0)

	table.store(deep_entry)
	table.store(shallow_entry)

	assert table.get("position") == deep_entry


def test_unlimited_depth_entry_replaces_finite_depth_entry(table):
	finite_entry = make_entry(depth=5, score=2.0)
	unlimited_entry = make_entry(depth=None, score=3.0)

	table.store(finite_entry)
	table.store(unlimited_entry)

	assert table.get("position") == unlimited_entry


def test_finite_depth_entry_does_not_replace_unlimited_depth_entry(table):
	unlimited_entry = make_entry(depth=None, score=3.0)
	finite_entry = make_entry(depth=10, score=2.0)

	table.store(unlimited_entry)
	table.store(finite_entry)

	assert table.get("position") == unlimited_entry


def test_exact_entry_replaces_lower_bound_at_equal_depth(table):
	lower_entry = make_entry(
		depth=4,
		bound=TranspositionBound.LOWER,
	)
	exact_entry = make_entry(
		depth=4,
		bound=TranspositionBound.EXACT,
		score=3.0,
	)

	table.store(lower_entry)
	table.store(exact_entry)

	assert table.get("position") == exact_entry


def test_exact_entry_replaces_upper_bound_at_equal_depth(table):
	upper_entry = make_entry(
		depth=4,
		bound=TranspositionBound.UPPER,
	)
	exact_entry = make_entry(
		depth=4,
		bound=TranspositionBound.EXACT,
		score=3.0,
	)

	table.store(upper_entry)
	table.store(exact_entry)

	assert table.get("position") == exact_entry


@pytest.mark.parametrize(
	"existing_bound",
	[
		TranspositionBound.LOWER,
		TranspositionBound.UPPER,
	],
)
def test_bound_entry_does_not_replace_exact_entry(
	table,
	existing_bound,
):
	exact_entry = make_entry(
		depth=4,
		bound=TranspositionBound.EXACT,
	)
	bound_entry = make_entry(
		depth=4,
		bound=existing_bound,
		score=5.0,
	)

	table.store(exact_entry)
	table.store(bound_entry)

	assert table.get("position") == exact_entry


def test_entry_with_same_depth_and_bound_replaces_existing_entry(table):
	first_entry = make_entry(
		depth=4,
		score=2.0,
		bound=TranspositionBound.LOWER,
	)
	second_entry = make_entry(
		depth=4,
		score=3.0,
		bound=TranspositionBound.LOWER,
	)

	table.store(first_entry)
	table.store(second_entry)

	assert table.get("position") == second_entry


def test_best_move_is_available_even_when_score_depth_is_insufficient(
	table,
):
	entry = make_entry(depth=2, best_move="preferred_move")

	table.store(entry)

	retrieved_entry = table.get("position")

	assert retrieved_entry is not None
	assert not retrieved_entry.is_usable_for_depth(5)
	assert retrieved_entry.best_move == "preferred_move"


def test_different_position_keys_store_separate_entries(table):
	first_entry = make_entry(key="position_a", score=1.0)
	second_entry = make_entry(key="position_b", score=2.0)

	table.store(first_entry)
	table.store(second_entry)

	assert table.get("position_a") == first_entry
	assert table.get("position_b") == second_entry
	assert len(table) == 2


def test_clear_removes_all_entries(table):
	table.store(make_entry(key="position_a"))
	table.store(make_entry(key="position_b"))

	table.clear()

	assert table.get("position_a") is None
	assert table.get("position_b") is None
	assert len(table) == 0


def test_store_after_clear_works(table):
	entry = make_entry()

	table.store(entry)
	table.clear()
	table.store(entry)

	assert table.get("position") == entry
	assert len(table) == 1

