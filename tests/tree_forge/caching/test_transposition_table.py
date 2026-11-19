from dataclasses import FrozenInstanceError, replace
from itertools import permutations

import pytest

from tree_forge.caching.transposition_table import (
	StoreOutcome,
	TranspositionBound,
	TranspositionEntry,
	TranspositionTable,
)


KEY = "position"


def make_entry(
	key="position",
	depth=4,
	lower=3.0,
	upper=3.0,
	best_move="move",
	solved=False,
	score_valid=True,
) -> TranspositionEntry:
	"""Create an entry; by default its value is exactly 3."""
	return TranspositionEntry(
		key=key,
		depth=depth,
		lower=lower,
		upper=upper,
		best_move=best_move,
		solved=solved,
		score_valid=score_valid,
	)


def exact(key, depth, score, solved=False, best_move=None):
	return make_entry(
		key=key,
		depth=depth,
		lower=score,
		upper=score,
		best_move=best_move,
		solved=solved,
	)


class TestTranspositionBound:
	def test_defines_expected_bounds(self) -> None:
		assert {bound.name for bound in TranspositionBound} == {
			"EXACT",
			"LOWER",
			"UPPER",
		}

	def test_bounds_have_distinct_values(self) -> None:
		values = [bound.value for bound in TranspositionBound]

		assert len(values) == len(set(values))


class TestTranspositionEntry:
	def test_optional_fields_have_defaults(self) -> None:
		entry = TranspositionEntry(key="position", depth=3)

		assert entry.lower is None
		assert entry.upper is None
		assert entry.best_move is None
		assert entry.solved is False
		assert entry.score_valid is True

	def test_stores_all_fields(self) -> None:
		entry = make_entry(
			key=("board", 1),
			depth=7,
			lower=-2.5,
			upper=4.0,
			best_move="e2e4",
			solved=True,
			score_valid=False,
		)

		assert entry.key == ("board", 1)
		assert entry.depth == 7
		assert entry.lower == -2.5
		assert entry.upper == 4.0
		assert entry.best_move == "e2e4"
		assert entry.solved is True
		assert entry.score_valid is False

	def test_is_immutable(self) -> None:
		entry = make_entry()

		with pytest.raises(FrozenInstanceError):
			entry.lower = 10.0

	def test_entries_with_equal_fields_are_equal(self) -> None:
		assert make_entry() == make_entry()
		assert make_entry(lower=1.0) != make_entry(lower=2.0)

	def test_replace_creates_modified_copy(self) -> None:
		original = make_entry()

		invalidated = replace(original, score_valid=False)

		assert invalidated.score_valid is False
		assert invalidated.lower == original.lower
		assert original.score_valid is True

	@pytest.mark.parametrize("depth", [0, 1, 25, None])
	def test_accepts_non_negative_or_missing_depth(self, depth) -> None:
		assert make_entry(depth=depth).depth == depth

	@pytest.mark.parametrize("depth", [-1, -10])
	def test_rejects_negative_depth(self, depth: int) -> None:
		with pytest.raises(
			ValueError,
			match="depth must be non-negative or None",
		):
			make_entry(depth=depth)

	@pytest.mark.parametrize("lower, upper", [(2.0, 1.0), (0.0, -0.5)])
	def test_rejects_a_lower_bound_above_the_upper_bound(
		self,
		lower: float,
		upper: float,
	) -> None:
		with pytest.raises(ValueError, match="lower must not exceed upper"):
			make_entry(lower=lower, upper=upper)

	def test_accepts_equal_or_missing_bounds(self) -> None:
		make_entry(lower=1.0, upper=1.0)
		make_entry(lower=None, upper=1.0)
		make_entry(lower=1.0, upper=None)
		make_entry(lower=None, upper=None)


class TestTranspositionEntryScoreInformation:
	def test_equal_bounds_are_an_exact_score(self) -> None:
		entry = make_entry(lower=2.0, upper=2.0)

		assert entry.is_exact is True
		assert entry.score == 2.0

	def test_a_zero_score_is_exact_and_not_missing(self) -> None:
		entry = make_entry(lower=0.0, upper=0.0)

		assert entry.is_exact is True
		assert entry.score == 0.0
		assert entry.score is not None

	@pytest.mark.parametrize("lower, upper", [
		pytest.param(1.0, None, id="lower-only"),
		pytest.param(None, 1.0, id="upper-only"),
		pytest.param(1.0, 2.0, id="two-sided-interval"),
		pytest.param(None, None, id="no-information"),
	])
	def test_inexact_entries_have_no_score(self, lower, upper) -> None:
		entry = make_entry(lower=lower, upper=upper)

		assert entry.is_exact is False
		assert entry.score is None

	def test_a_zero_bound_is_a_known_bound(self) -> None:
		entry = make_entry(lower=0.0, upper=None)

		assert entry.lower == 0.0
		assert entry.upper is None


class TestTranspositionEntryFromBound:
	def test_exact_bound_sets_both_sides(self) -> None:
		entry = TranspositionEntry.from_bound(
			KEY,
			3,
			2.5,
			TranspositionBound.EXACT,
		)

		assert (entry.lower, entry.upper) == (2.5, 2.5)

	def test_lower_bound_leaves_the_upper_side_unknown(self) -> None:
		entry = TranspositionEntry.from_bound(
			KEY,
			3,
			2.5,
			TranspositionBound.LOWER,
		)

		assert (entry.lower, entry.upper) == (2.5, None)

	def test_upper_bound_leaves_the_lower_side_unknown(self) -> None:
		entry = TranspositionEntry.from_bound(
			KEY,
			3,
			2.5,
			TranspositionBound.UPPER,
		)

		assert (entry.lower, entry.upper) == (None, 2.5)

	def test_passes_the_remaining_fields_through(self) -> None:
		entry = TranspositionEntry.from_bound(
			KEY,
			3,
			0.0,
			TranspositionBound.EXACT,
			best_move="move",
			solved=True,
			score_valid=False,
		)

		assert entry.key == KEY
		assert entry.depth == 3
		assert entry.best_move == "move"
		assert entry.solved is True
		assert entry.score_valid is False


class TestTranspositionEntryUsability:
	@pytest.mark.parametrize(
		"entry_depth, requested_depth, expected",
		[
			pytest.param(4, 4, True, id="equal-depth"),
			pytest.param(5, 3, True, id="deeper-than-requested"),
			pytest.param(3, 5, False, id="shallower-than-requested"),
			pytest.param(0, 0, True, id="zero-depth-equal"),
			pytest.param(0, 1, False, id="zero-depth-too-shallow"),
			pytest.param(None, 1, True, id="unlimited-entry-finite-request"),
			pytest.param(None, 99, True, id="unlimited-entry-deep-request"),
			pytest.param(None, None, True, id="unlimited-both"),
			pytest.param(10, None, False, id="finite-entry-unlimited-request"),
		],
	)
	def test_unsolved_entry_depth_rules(
		self,
		entry_depth,
		requested_depth,
		expected: bool,
	) -> None:
		entry = make_entry(depth=entry_depth)

		assert entry.is_usable_for_depth(requested_depth) is expected

	@pytest.mark.parametrize("requested_depth", [0, 5, 1000, None])
	def test_solved_entry_is_usable_at_any_depth(self, requested_depth) -> None:
		entry = make_entry(depth=0, solved=True)

		assert entry.is_usable_for_depth(requested_depth) is True

	@pytest.mark.parametrize("solved", [False, True])
	@pytest.mark.parametrize("requested_depth", [0, 1, None])
	def test_entry_with_invalid_score_is_never_usable(
		self,
		solved: bool,
		requested_depth,
	) -> None:
		entry = make_entry(depth=None, solved=solved, score_valid=False)

		assert entry.is_usable_for_depth(requested_depth) is False

	@pytest.mark.parametrize("lower, upper", [
		(3.0, 3.0),
		(3.0, None),
		(None, 3.0),
		(1.0, 3.0),
		(None, None),
	])
	def test_usability_does_not_depend_on_the_bounds(self, lower, upper) -> None:
		entry = make_entry(depth=4, lower=lower, upper=upper)

		assert entry.is_usable_for_depth(4)
		assert not entry.is_usable_for_depth(5)


class TestStoreOutcome:
	def test_dependency_counts_default_to_not_measured(self) -> None:
		outcome = StoreOutcome(stored=True)

		assert outcome.invalidated is None
		assert outcome.refreshed is None

	def test_is_immutable(self) -> None:
		with pytest.raises(FrozenInstanceError):
			StoreOutcome(stored=True).stored = False


class TestTranspositionTableBasics:
	@pytest.fixture
	def table(self) -> TranspositionTable:
		return TranspositionTable()

	def test_new_table_is_empty(self, table: TranspositionTable) -> None:
		assert len(table) == 0

	def test_dependency_tracking_is_disabled_by_default(
		self,
		table: TranspositionTable,
	) -> None:
		assert table.dependency_tracking is False

	def test_dependency_tracking_can_be_enabled(self) -> None:
		assert TranspositionTable(dependency_tracking=True).dependency_tracking

	def test_get_returns_none_for_missing_key(
		self,
		table: TranspositionTable,
	) -> None:
		assert table.get("missing") is None

	def test_stores_and_retrieves_entry(
		self,
		table: TranspositionTable,
	) -> None:
		entry = make_entry()

		table.store(entry)

		assert table.get(KEY) == entry
		assert len(table) == 1

	def test_stores_entries_for_different_keys_separately(
		self,
		table: TranspositionTable,
	) -> None:
		first = make_entry(key="first", lower=1.0, upper=1.0)
		second = make_entry(key="second", lower=2.0, upper=2.0)

		table.store(first)
		table.store(second)

		assert table.get("first") == first
		assert table.get("second") == second
		assert len(table) == 2

	def test_replacing_an_entry_does_not_change_length(
		self,
		table: TranspositionTable,
	) -> None:
		table.store(make_entry(depth=2))
		table.store(make_entry(depth=5))

		assert len(table) == 1

	def test_accepts_any_hashable_key(
		self,
		table: TranspositionTable,
	) -> None:
		for key in (("board", 1), 42, frozenset({1, 2}), "text"):
			table.store(make_entry(key=key))

		assert len(table) == 4
		assert table.get(("board", 1)) is not None

	def test_get_returns_entries_that_are_not_usable(
		self,
		table: TranspositionTable,
	) -> None:
		table.store(make_entry(depth=1, best_move="hint"))

		entry = table.get(KEY)

		assert entry is not None
		assert not entry.is_usable_for_depth(8)
		assert entry.best_move == "hint"

	def test_entry_without_bounds_is_stored_for_its_best_move(
		self,
		table: TranspositionTable,
	) -> None:
		table.store(make_entry(lower=None, upper=None, best_move="hint"))

		assert table.get(KEY).best_move == "hint"
		assert table.get(KEY).score is None

	def test_clear_removes_all_entries(
		self,
		table: TranspositionTable,
	) -> None:
		table.store(make_entry(key="first"))
		table.store(make_entry(key="second"))

		table.clear()

		assert table.get("first") is None
		assert table.get("second") is None
		assert len(table) == 0

	def test_clearing_empty_table_is_a_no_op(
		self,
		table: TranspositionTable,
	) -> None:
		table.clear()

		assert len(table) == 0

	def test_table_is_usable_after_clear(
		self,
		table: TranspositionTable,
	) -> None:
		entry = make_entry()

		table.store(entry)
		table.clear()
		table.store(entry)

		assert table.get(KEY) == entry
		assert len(table) == 1

	def test_tables_do_not_share_entries(self) -> None:
		first = TranspositionTable()
		second = TranspositionTable()

		first.store(make_entry())

		assert second.get(KEY) is None


class TestTranspositionTableStoreOutcome:
	def test_first_store_is_reported_as_stored(self) -> None:
		assert TranspositionTable().store(make_entry()).stored is True

	def test_identical_store_is_not_reported_as_stored(self) -> None:
		table = TranspositionTable()
		table.store(make_entry())

		assert table.store(make_entry()).stored is False

	def test_rejected_entry_is_not_reported_as_stored(self) -> None:
		table = TranspositionTable()
		table.store(make_entry(depth=6))

		assert table.store(make_entry(depth=2, lower=9.0, upper=9.0)).stored is False

	def test_entry_that_changes_the_table_is_reported_as_stored(self) -> None:
		table = TranspositionTable()
		table.store(make_entry(depth=2))

		assert table.store(make_entry(depth=5)).stored is True

	def test_invalid_entry_is_not_reported_as_stored(self) -> None:
		table = TranspositionTable()
		table.store(make_entry())

		assert table.store(make_entry(depth=9, score_valid=False)).stored is False

	def test_dependency_counts_are_not_measured_without_tracking(self) -> None:
		outcome = TranspositionTable().store(make_entry())

		assert outcome.invalidated is None
		assert outcome.refreshed is None

	def test_dependency_counts_are_zero_when_tracked_but_nothing_changed(
		self,
	) -> None:
		outcome = TranspositionTable(dependency_tracking=True).store(make_entry())

		assert outcome.invalidated == 0
		assert outcome.refreshed == 0

	def test_a_rejected_store_with_tracking_reports_zero_counts(self) -> None:
		table = TranspositionTable(dependency_tracking=True)
		table.store(make_entry(depth=6))

		outcome = table.store(make_entry(depth=2))

		assert outcome == StoreOutcome(False, 0, 0)

	def test_replace_dependencies_is_not_measured_without_tracking(
		self,
	) -> None:
		outcome = TranspositionTable().replace_dependencies("p", {"c": 1})

		assert outcome == StoreOutcome(False)


def entry(best_move="old", **fields) -> TranspositionEntry:
	"""Create an entry whose bounds are unknown unless given."""
	fields.setdefault("lower", None)
	fields.setdefault("upper", None)

	return make_entry(best_move=best_move, **fields)


class TestTranspositionTableMerge:
	"""Decision table for combining new information with a stored entry."""

	@pytest.mark.parametrize(
		"existing, new, expected",
		[
			# Proven information outranks horizon-limited information.
			pytest.param(
				dict(depth=6, lower=3.0, upper=3.0),
				dict(depth=1, lower=1.0, solved=True),
				dict(lower=1.0, upper=None, best_move="new", solved=True),
				id="solved-beats-unsolved-even-if-shallower-and-weaker",
			),
			pytest.param(
				dict(depth=1, lower=2.0, upper=2.0, solved=True),
				dict(depth=9, lower=5.0, upper=5.0),
				dict(lower=2.0, upper=2.0, best_move="old", solved=True),
				id="unsolved-never-replaces-solved",
			),
			pytest.param(
				dict(solved=True, lower=1.0, upper=1.0),
				dict(solved=True, lower=2.0, upper=2.0),
				dict(lower=2.0, upper=2.0, best_move="new", solved=True),
				id="solved-with-changed-score-is-new-information",
			),
			pytest.param(
				dict(solved=True, lower=1.0, upper=1.0),
				dict(solved=True, lower=1.0, upper=1.0),
				dict(lower=1.0, upper=1.0, best_move="old", solved=True),
				id="solved-with-same-score-is-not-new-information",
			),
			pytest.param(
				dict(solved=True, lower=1.0),
				dict(solved=True, lower=1.0, upper=1.0),
				dict(lower=1.0, upper=1.0, best_move="new", solved=True),
				id="solved-exact-completes-solved-bound",
			),
			pytest.param(
				dict(solved=True, lower=3.0, upper=3.0),
				dict(solved=True, lower=9.0),
				dict(lower=3.0, upper=3.0, best_move="old", solved=True),
				id="contradicting-solved-bound-does-not-replace-solved-exact",
			),
			pytest.param(
				dict(depth=0, solved=True, lower=1.0, upper=1.0),
				dict(depth=9, solved=True, lower=2.0, upper=2.0),
				dict(lower=2.0, upper=2.0, best_move="new", solved=True),
				id="depth-is-ignored-between-solved-entries",
			),
			# Search depth of unsolved entries.
			pytest.param(
				dict(depth=4, lower=3.0, upper=3.0),
				dict(depth=6, lower=3.0, upper=3.0),
				dict(depth=6, lower=3.0, upper=3.0, best_move="new"),
				id="deeper-replaces-shallower",
			),
			pytest.param(
				dict(depth=4, lower=1.0),
				dict(depth=2, lower=5.0, upper=5.0),
				dict(depth=4, lower=1.0, upper=None, best_move="old"),
				id="shallower-never-replaces-deeper-even-if-exact",
			),
			pytest.param(
				dict(depth=4, lower=3.0, upper=3.0),
				dict(depth=6, lower=1.0),
				dict(depth=6, lower=1.0, upper=None, best_move="new"),
				id="deeper-bound-replaces-shallower-exact",
			),
			pytest.param(
				dict(depth=5, lower=3.0, upper=3.0),
				dict(depth=None, lower=4.0, upper=4.0),
				dict(depth=None, lower=4.0, upper=4.0, best_move="new"),
				id="unlimited-depth-replaces-finite-depth",
			),
			pytest.param(
				dict(depth=None, lower=3.0, upper=3.0),
				dict(depth=10, lower=4.0, upper=4.0),
				dict(depth=None, lower=3.0, upper=3.0, best_move="old"),
				id="finite-depth-does-not-replace-unlimited-depth",
			),
			# Intersecting information of equal depth.
			pytest.param(
				dict(lower=1.0),
				dict(upper=5.0),
				dict(lower=1.0, upper=5.0, best_move="new"),
				id="lower-and-upper-bounds-combine-into-an-interval",
			),
			pytest.param(
				dict(upper=5.0),
				dict(lower=1.0),
				dict(lower=1.0, upper=5.0, best_move="new"),
				id="upper-and-lower-bounds-combine-into-an-interval",
			),
			pytest.param(
				dict(lower=2.0),
				dict(lower=3.0),
				dict(lower=3.0, upper=None, best_move="new"),
				id="higher-lower-bound-is-stronger",
			),
			pytest.param(
				dict(lower=3.0),
				dict(lower=2.0),
				dict(lower=3.0, upper=None, best_move="old"),
				id="lower-lower-bound-is-weaker",
			),
			pytest.param(
				dict(lower=3.0),
				dict(lower=3.0),
				dict(lower=3.0, upper=None, best_move="old"),
				id="equal-lower-bounds-are-not-new-information",
			),
			pytest.param(
				dict(upper=3.0),
				dict(upper=2.0),
				dict(lower=None, upper=2.0, best_move="new"),
				id="lower-upper-bound-is-stronger",
			),
			pytest.param(
				dict(upper=2.0),
				dict(upper=3.0),
				dict(lower=None, upper=2.0, best_move="old"),
				id="higher-upper-bound-is-weaker",
			),
			pytest.param(
				dict(upper=3.0),
				dict(upper=3.0),
				dict(lower=None, upper=3.0, best_move="old"),
				id="equal-upper-bounds-are-not-new-information",
			),
			pytest.param(
				dict(lower=1.0),
				dict(lower=4.0, upper=4.0),
				dict(lower=4.0, upper=4.0, best_move="new"),
				id="exact-score-completes-a-bound",
			),
			pytest.param(
				dict(lower=1.0, upper=9.0),
				dict(lower=4.0, upper=4.0),
				dict(lower=4.0, upper=4.0, best_move="new"),
				id="exact-score-narrows-an-interval",
			),
			pytest.param(
				dict(lower=3.0, upper=3.0),
				dict(lower=2.0),
				dict(lower=3.0, upper=3.0, best_move="old"),
				id="bound-does-not-loosen-an-exact-score",
			),
			pytest.param(
				dict(lower=3.0, upper=3.0),
				dict(upper=7.0),
				dict(lower=3.0, upper=3.0, best_move="old"),
				id="upper-bound-does-not-loosen-an-exact-score",
			),
			pytest.param(
				dict(lower=3.0, upper=3.0),
				dict(lower=3.0, upper=3.0),
				dict(lower=3.0, upper=3.0, best_move="old"),
				id="same-exact-score-is-not-new-information",
			),
			# Contradicting information of equal depth.
			pytest.param(
				dict(lower=3.0, upper=3.0),
				dict(lower=4.0, upper=4.0),
				dict(lower=4.0, upper=4.0, best_move="new"),
				id="equally-precise-contradiction-prefers-the-newer",
			),
			pytest.param(
				dict(lower=5.0),
				dict(upper=2.0),
				dict(lower=None, upper=2.0, best_move="new"),
				id="contradicting-bounds-prefer-the-newer",
			),
			pytest.param(
				dict(lower=3.0, upper=3.0),
				dict(lower=9.0),
				dict(lower=3.0, upper=3.0, best_move="old"),
				id="contradiction-prefers-the-more-precise-entry",
			),
			pytest.param(
				dict(upper=2.0),
				dict(lower=5.0),
				dict(lower=5.0, upper=None, best_move="new"),
				id="contradicting-bounds-prefer-the-newer-in-either-direction",
			),
			pytest.param(
				dict(lower=3.0, upper=3.0),
				dict(upper=1.0),
				dict(lower=3.0, upper=3.0, best_move="old"),
				id="exact-score-outweighs-a-contradicting-upper-bound",
			),
			pytest.param(
				dict(lower=9.0),
				dict(lower=3.0, upper=3.0),
				dict(lower=3.0, upper=3.0, best_move="new"),
				id="a-more-precise-new-entry-replaces-a-contradicted-bound",
			),
			pytest.param(
				dict(upper=1.0),
				dict(lower=3.0, upper=3.0),
				dict(lower=3.0, upper=3.0, best_move="new"),
				id="a-more-precise-new-entry-replaces-a-contradicted-upper-bound",
			),
			# Score validity.
			pytest.param(
				dict(depth=9, lower=3.0, upper=3.0, score_valid=False),
				dict(depth=1, upper=4.0),
				dict(depth=1, lower=None, upper=4.0, best_move="new"),
				id="valid-entry-replaces-invalidated-entry",
			),
			pytest.param(
				dict(depth=1, lower=3.0, upper=3.0),
				dict(depth=9, lower=3.0, upper=3.0, score_valid=False),
				dict(depth=1, lower=3.0, upper=3.0, best_move="old"),
				id="invalid-entry-never-replaces-valid-entry",
			),
			pytest.param(
				dict(lower=3.0, upper=3.0, score_valid=False),
				dict(lower=3.0, upper=3.0, score_valid=False),
				dict(lower=3.0, upper=3.0, best_move="old", score_valid=False),
				id="invalid-entry-does-not-change-an-invalid-entry",
			),
		],
	)
	def test_merge_policy(
		self,
		existing: dict,
		new: dict,
		expected: dict,
	) -> None:
		table = TranspositionTable()

		table.store(entry(best_move="old", **existing))
		table.store(entry(best_move="new", **new))

		stored = table.get(KEY)

		for field, value in expected.items():
			assert getattr(stored, field) == value, field

	@pytest.mark.parametrize("existing_depth, new_depth, expected_depth", [
		pytest.param(5, 2, 5, id="shallower-new-keeps-the-deeper-depth"),
		pytest.param(2, 5, 5, id="deeper-new-raises-the-depth"),
		pytest.param(None, 3, None, id="unlimited-depth-is-kept"),
		pytest.param(3, None, None, id="unlimited-depth-is-adopted"),
		pytest.param(4, 4, 4, id="equal-depths-are-unchanged"),
	])
	def test_solved_entries_keep_the_deepest_depth_they_were_proven_at(
		self,
		existing_depth,
		new_depth,
		expected_depth,
	) -> None:
		table = TranspositionTable()

		# The same proven value; the new entry also learns the upper bound.
		table.store(entry(depth=existing_depth, lower=1.0, solved=True))
		table.store(entry(depth=new_depth, lower=1.0, upper=1.0, solved=True))

		stored = table.get(KEY)

		assert stored.depth == expected_depth
		assert stored.score == 1.0

	def test_a_solved_entry_with_nothing_new_still_raises_the_depth(self) -> None:
		table = TranspositionTable()
		table.store(entry(depth=2, lower=1.0, upper=1.0, solved=True))

		outcome = table.store(entry(depth=6, lower=1.0, upper=1.0, solved=True))

		assert outcome.stored is True
		assert table.get(KEY).depth == 6

	def test_rejected_entry_leaves_existing_entry_untouched(self) -> None:
		table = TranspositionTable()
		deep = make_entry(depth=6, lower=3.0, upper=3.0)

		table.store(deep)
		table.store(make_entry(depth=2, lower=-1.0, upper=-1.0))

		assert table.get(KEY) == deep

	def test_improved_entry_replaces_all_fields(self) -> None:
		table = TranspositionTable()
		better = make_entry(
			depth=8,
			lower=5.0,
			upper=None,
			best_move="better",
			solved=False,
		)

		table.store(make_entry(depth=2, lower=1.0, upper=1.0, best_move="worse"))
		table.store(better)

		assert table.get(KEY) == better

	def test_improved_entry_inherits_best_move_when_it_has_none(self) -> None:
		table = TranspositionTable()

		table.store(make_entry(depth=2, best_move="hint"))
		table.store(make_entry(depth=5, lower=9.0, upper=9.0, best_move=None))

		stored = table.get(KEY)

		assert stored is not None
		assert stored.depth == 5
		assert stored.score == 9.0
		assert stored.best_move == "hint"

	def test_improved_entry_prefers_its_own_best_move(self) -> None:
		table = TranspositionTable()

		table.store(make_entry(depth=2, best_move="old"))
		table.store(make_entry(depth=5, best_move="new"))

		assert table.get(KEY).best_move == "new"

	def test_best_move_is_inherited_across_an_invalidated_entry(self) -> None:
		table = TranspositionTable()

		table.store(make_entry(best_move="hint", score_valid=False))
		table.store(make_entry(depth=1, best_move=None))

		stored = table.get(KEY)

		assert stored.score_valid is True
		assert stored.best_move == "hint"

	def test_rejected_entry_does_not_alter_best_move(self) -> None:
		table = TranspositionTable()

		table.store(make_entry(depth=5, best_move="kept"))
		table.store(make_entry(depth=1, best_move="ignored"))
		table.store(make_entry(depth=1, best_move=None))

		assert table.get(KEY).best_move == "kept"

	def test_first_entry_without_best_move_keeps_none(self) -> None:
		table = TranspositionTable()

		table.store(make_entry(best_move=None))

		assert table.get(KEY).best_move is None

	def test_replacement_is_scoped_to_the_entry_key(self) -> None:
		table = TranspositionTable()
		other = make_entry(key="other", depth=9, lower=1.0, upper=1.0)

		table.store(other)
		table.store(make_entry(key="shared", depth=1))
		table.store(make_entry(key="shared", depth=5))

		assert table.get("other") == other
		assert table.get("shared").depth == 5


class TestTranspositionTableMergeOrderIndependence:
	"""Consistent information must merge to the same result in any order."""

	# Everything below is consistent with a position whose true value is 3.
	SAMPLES = [
		dict(depth=1, lower=2.0, upper=None),
		dict(depth=1, lower=None, upper=5.0),
		dict(depth=2, lower=1.0, upper=4.0),
		dict(depth=2, lower=3.0, upper=None),
		dict(depth=None, lower=None, upper=3.0),
		dict(depth=None, lower=3.0, upper=3.0),
		dict(depth=0, lower=3.0, upper=3.0, solved=True),
		dict(depth=7, lower=None, upper=None),
	]

	@staticmethod
	def result(samples) -> tuple:
		table = TranspositionTable()

		for sample in samples:
			table.store(entry(**sample))

		stored = table.get(KEY)

		return stored.depth, stored.lower, stored.upper, stored.solved

	def test_every_pair_merges_to_the_same_result_in_both_orders(self) -> None:
		for first in self.SAMPLES:
			for second in self.SAMPLES:
				assert self.result([first, second]) == self.result(
					[second, first]
				)

	def test_every_ordering_of_several_entries_agrees(self) -> None:
		samples = [self.SAMPLES[1], self.SAMPLES[2], self.SAMPLES[3], self.SAMPLES[6]]
		results = {self.result(order) for order in permutations(samples)}

		assert len(results) == 1

	@pytest.mark.parametrize("group", [
		pytest.param([0, 1, 2, 3], id="finite-depths"),
		pytest.param([0, 2, 4, 5], id="unlimited-depth"),
		pytest.param([0, 5, 6, 7], id="solved-and-unsolved"),
	])
	def test_orderings_of_mixed_groups_agree(self, group) -> None:
		samples = [self.SAMPLES[index] for index in group]
		results = {self.result(order) for order in permutations(samples)}

		assert len(results) == 1

	def test_storing_an_entry_twice_changes_nothing(self) -> None:
		for sample in self.SAMPLES:
			assert self.result([sample, sample]) == self.result([sample])

	def test_merged_information_is_at_least_as_precise_as_each_input(
		self,
	) -> None:
		for first in self.SAMPLES:
			for second in self.SAMPLES:
				table = TranspositionTable()
				table.store(entry(**first))
				table.store(entry(**second))
				merged = table.get(KEY)

				if merged.depth == first["depth"] and not first.get("solved"):
					if first["lower"] is not None:
						assert merged.lower is not None
						assert merged.lower >= first["lower"]

					if first["upper"] is not None:
						assert merged.upper is not None
						assert merged.upper <= first["upper"]


def tracked() -> TranspositionTable:
	return TranspositionTable(dependency_tracking=True)


def bounds(key, depth, lower=None, upper=None, solved=False):
	"""Create an entry with only the given bounds known."""
	return TranspositionEntry(
		key=key,
		depth=depth,
		lower=lower,
		upper=upper,
		solved=solved,
	)


class TestDependencyRefresh:
	"""Information found later improves positions that were searched earlier."""

	def test_later_information_completes_a_value_that_was_unknown(self) -> None:
		table = tracked()

		# "parent" is the best of "x" and "y", seen from the opposite
		# perspective. It is only known to be at least -3.
		table.store(bounds("parent", 2, lower=-3.0))
		table.replace_dependencies("parent", {"x": -1, "y": -1})

		# "x" is worth 3, so "parent" >= -3, which was already known. The
		# upper bound stays unknown until "y" is known as well.
		outcome = table.store(exact("x", 1, 3.0))

		assert outcome.refreshed == 0
		assert table.get("parent").lower == -3.0
		assert table.get("parent").upper is None
		assert table.get("parent").score is None

		# "y" is worth 1, which completes the value: max(-3, -1) = -1.
		outcome = table.store(exact("y", 1, 1.0))

		assert outcome == StoreOutcome(True, 0, 1)
		assert table.get("parent").score == -1.0

	def test_continuations_known_before_the_position_complete_it(self) -> None:
		table = tracked()
		table.store(exact("x", 1, 3.0))
		table.store(exact("y", 1, 1.0))
		table.store(bounds("parent", 2, lower=-3.0))

		outcome = table.replace_dependencies("parent", {"x": -1, "y": -1})

		assert outcome == StoreOutcome(True, 0, 1)
		assert table.get("parent").score == -1.0

	def test_a_position_stored_after_its_dependencies_is_completed(self) -> None:
		table = tracked()
		table.replace_dependencies("parent", {"x": -1, "y": -1})
		table.store(exact("x", 1, 3.0))
		table.store(exact("y", 1, 1.0))

		outcome = table.store(bounds("parent", 2, lower=-3.0))

		assert outcome == StoreOutcome(True, 0, 1)
		assert table.get("parent").score == -1.0

	def test_unknown_continuations_leave_the_upper_bound_unknown(self) -> None:
		table = tracked()
		table.store(bounds("parent", 2))
		table.replace_dependencies("parent", {"x": 1, "y": 1})

		table.store(exact("x", 1, 4.0))

		assert table.get("parent").lower == 4.0
		assert table.get("parent").upper is None

	def test_a_known_zero_is_not_confused_with_unknown(self) -> None:
		table = tracked()
		table.store(bounds("parent", 2))
		table.replace_dependencies("parent", {"x": 1, "y": 1})

		table.store(exact("x", 1, 0.0))

		assert table.get("parent").lower == 0.0
		assert table.get("parent").upper is None

		table.store(exact("y", 1, 0.0))

		assert table.get("parent").score == 0.0

	def test_positive_relation_passes_bounds_through(self) -> None:
		table = tracked()
		table.store(bounds("parent", 2))
		table.replace_dependencies("parent", {"child": 1})

		table.store(bounds("child", 1, lower=2.0, upper=5.0))

		assert (table.get("parent").lower, table.get("parent").upper) == (2.0, 5.0)

	def test_negative_relation_negates_and_swaps_bounds(self) -> None:
		table = tracked()
		table.store(bounds("parent", 2))
		table.replace_dependencies("parent", {"child": -1})

		table.store(bounds("child", 1, lower=2.0, upper=5.0))

		assert (table.get("parent").lower, table.get("parent").upper) == (-5.0, -2.0)

	def test_negating_a_zero_does_not_produce_a_negative_zero(self) -> None:
		table = tracked()
		table.store(bounds("parent", 2))
		table.replace_dependencies("parent", {"child": -1})

		table.store(exact("child", 1, 0.0))

		assert repr(table.get("parent").lower) == "0.0"
		assert repr(table.get("parent").upper) == "0.0"

	def test_negative_relation_keeps_a_missing_side_missing(self) -> None:
		table = tracked()
		table.store(bounds("parent", 2))
		table.replace_dependencies("parent", {"child": -1})

		table.store(bounds("child", 1, lower=2.0))

		# A lower bound of 2 on the child is an upper bound of -2.
		assert table.get("parent").lower is None
		assert table.get("parent").upper == -2.0

	def test_a_position_takes_its_best_continuation(self) -> None:
		table = tracked()
		table.store(bounds("parent", 2))
		table.replace_dependencies("parent", {"first": 1, "second": 1})

		table.store(bounds("first", 1, lower=1.0, upper=2.0))
		table.store(bounds("second", 1, lower=3.0, upper=4.0))

		assert (table.get("parent").lower, table.get("parent").upper) == (3.0, 4.0)

	def test_upper_bound_needs_an_upper_bound_for_every_continuation(
		self,
	) -> None:
		table = tracked()
		table.store(bounds("parent", 2))
		table.replace_dependencies("parent", {"first": 1, "second": 1})

		table.store(exact("first", 1, 5.0))
		table.store(bounds("second", 1, lower=1.0))

		assert table.get("parent").lower == 5.0
		assert table.get("parent").upper is None

	def test_information_that_contradicts_the_stored_bounds_replaces_them(
		self,
	) -> None:
		table = tracked()
		table.store(exact("parent", 2, 5.0))
		table.replace_dependencies("parent", {"child": 1})

		# The continuation proves the position is at most 4, so the
		# earlier value of 5 can no longer be right.
		outcome = table.store(bounds("child", 1, upper=4.0))

		assert outcome.refreshed == 1
		assert (table.get("parent").lower, table.get("parent").upper) == (None, 4.0)

	def test_consistent_information_leaves_the_position_unchanged(self) -> None:
		table = tracked()
		table.store(exact("parent", 2, 5.0))
		table.replace_dependencies("parent", {"child": 1})
		before = table.get("parent")

		outcome = table.store(bounds("child", 1, upper=7.0))

		assert outcome == StoreOutcome(True, 0, 0)
		assert table.get("parent") == before

	def test_derived_information_narrows_the_stored_bounds(self) -> None:
		table = tracked()
		table.store(bounds("parent", 2, lower=1.0, upper=9.0))
		table.replace_dependencies("parent", {"child": 1})

		table.store(bounds("child", 1, lower=4.0, upper=6.0))

		assert (table.get("parent").lower, table.get("parent").upper) == (4.0, 6.0)

	def test_the_best_move_hint_survives_a_refresh(self) -> None:
		table = tracked()
		table.store(replace(bounds("parent", 2), best_move="hint"))
		table.replace_dependencies("parent", {"child": 1})

		table.store(exact("child", 1, 2.0))

		assert table.get("parent").best_move == "hint"
		assert table.get("parent").score == 2.0


class TestDependencyRegistration:
	"""Registering dependencies completes a position, but only if it may change."""

	def test_a_position_without_an_entry_is_left_alone(self) -> None:
		table = tracked()
		table.store(exact("child", 1, 4.0))

		outcome = table.replace_dependencies("parent", {"child": 1})

		assert outcome == StoreOutcome(False, 0, 0)
		assert table.get("parent") is None

	def test_a_proven_position_is_not_changed_by_registering_dependencies(
		self,
	) -> None:
		table = tracked()
		table.store(exact("child", 1, 9.0))
		table.store(exact("parent", 2, 3.0, solved=True))
		before = table.get("parent")

		outcome = table.replace_dependencies("parent", {"child": 1})

		assert outcome == StoreOutcome(False, 0, 0)
		assert table.get("parent") == before

	def test_a_proven_position_is_not_changed_when_it_is_stored_later(
		self,
	) -> None:
		table = tracked()
		table.store(exact("child", 1, 9.0))
		table.replace_dependencies("parent", {"child": 1})

		outcome = table.store(exact("parent", 2, 3.0, solved=True))

		assert outcome == StoreOutcome(True, 0, 0)
		assert table.get("parent").score == 3.0

	def test_a_stale_position_is_not_changed_by_registering_dependencies(
		self,
	) -> None:
		table = tracked()
		table.store(exact("child", 1, 9.0))
		table.store(replace(exact("parent", 2, 3.0), score_valid=False))
		before = table.get("parent")

		outcome = table.replace_dependencies("parent", {"child": 1})

		assert outcome == StoreOutcome(False, 0, 0)
		assert table.get("parent") == before

	def test_registering_changes_nothing_when_the_information_is_consistent(
		self,
	) -> None:
		table = tracked()
		table.store(exact("child", 1, 3.0))
		table.store(exact("parent", 2, 3.0))

		outcome = table.replace_dependencies("parent", {"child": 1})

		assert outcome == StoreOutcome(False, 0, 0)

	def test_registering_propagates_the_improvement_to_dependents(self) -> None:
		table = tracked()
		table.store(bounds("grand", 3))
		table.store(bounds("parent", 2))
		table.store(exact("child", 1, 4.0))
		table.replace_dependencies("grand", {"parent": 1})

		outcome = table.replace_dependencies("parent", {"child": 1})

		assert outcome == StoreOutcome(True, 0, 2)
		assert table.get("grand").score == 4.0


class TestDependencyDepth:
	def test_continuations_that_are_too_shallow_are_ignored(self) -> None:
		table = tracked()
		table.store(bounds("parent", 3))
		table.replace_dependencies("parent", {"child": 1})
		before = table.get("parent")

		# The parent needs continuations searched to depth 2.
		outcome = table.store(exact("child", 1, 4.0))

		assert outcome == StoreOutcome(True, 0, 0)
		assert table.get("parent") == before

	def test_continuations_searched_exactly_one_ply_less_are_used(self) -> None:
		table = tracked()
		table.store(bounds("parent", 3))
		table.replace_dependencies("parent", {"child": 1})

		table.store(exact("child", 2, 4.0))

		assert table.get("parent").score == 4.0

	def test_deeper_continuations_are_used(self) -> None:
		table = tracked()
		table.store(bounds("parent", 3))
		table.replace_dependencies("parent", {"child": 1})

		table.store(exact("child", 8, 4.0))

		assert table.get("parent").score == 4.0

	def test_deeper_information_arriving_later_improves_the_position(
		self,
	) -> None:
		table = tracked()
		table.store(bounds("parent", 3))
		table.replace_dependencies("parent", {"child": 1})
		table.store(exact("child", 1, 4.0))

		assert table.get("parent").score is None

		table.store(exact("child", 2, 6.0))

		assert table.get("parent").score == 6.0

	def test_unlimited_depth_needs_unlimited_depth_or_proven_continuations(
		self,
	) -> None:
		table = tracked()
		table.store(bounds("parent", None))
		table.replace_dependencies("parent", {"child": 1})

		table.store(exact("child", 5, 4.0))

		assert table.get("parent").score is None

		table.store(exact("child", None, 4.0))

		assert table.get("parent").score == 4.0

	def test_stale_continuations_are_ignored(self) -> None:
		table = tracked()
		table.store(bounds("parent", 2))
		table.replace_dependencies("parent", {"child": 1})
		table.store(replace(exact("child", 1, 4.0), score_valid=False))

		assert table.get("parent").score is None


class TestDependencySolved:
	def test_proven_continuations_prove_the_position(self) -> None:
		table = tracked()
		table.store(bounds("parent", 1))
		table.replace_dependencies("parent", {"first": 1, "second": 1})

		table.store(exact("first", 0, 1.0, solved=True))

		assert table.get("parent").solved is False
		assert table.get("parent").upper is None

		table.store(exact("second", 0, 4.0, solved=True))

		assert table.get("parent").solved is True
		assert table.get("parent").score == 4.0

	def test_a_continuation_that_is_not_proven_prevents_proof(self) -> None:
		table = tracked()
		table.store(bounds("parent", 2))
		table.replace_dependencies("parent", {"first": 1, "second": 1})

		table.store(exact("first", 0, 1.0, solved=True))
		table.store(exact("second", 1, 4.0))

		assert table.get("parent").score == 4.0
		assert table.get("parent").solved is False

	def test_proven_continuations_are_used_at_any_depth(self) -> None:
		table = tracked()
		table.store(bounds("parent", 9))
		table.replace_dependencies("parent", {"child": 1})

		table.store(exact("child", 0, 2.0, solved=True))

		assert table.get("parent").score == 2.0
		assert table.get("parent").solved is True

	def test_a_proven_position_is_not_changed_by_later_information(self) -> None:
		table = tracked()
		table.store(exact("parent", 2, 3.0, solved=True))
		table.replace_dependencies("parent", {"child": 1})
		before = table.get("parent")

		outcome = table.store(exact("child", 1, 9.0))

		assert outcome == StoreOutcome(True, 0, 0)
		assert table.get("parent") == before

	def test_a_proven_position_shields_the_positions_that_depend_on_it(
		self,
	) -> None:
		table = tracked()
		table.store(exact("grand", 3, 3.0))
		table.store(exact("parent", 2, 3.0, solved=True))
		table.replace_dependencies("grand", ["parent"])
		table.replace_dependencies("parent", {"child": 1})

		outcome = table.store(exact("child", 1, 9.0))

		assert outcome == StoreOutcome(True, 0, 0)
		assert table.get("grand").score_valid is True

	def test_a_proven_result_propagates_to_the_positions_above(self) -> None:
		table = tracked()
		table.store(bounds("top", 3))
		table.store(bounds("middle", 2))
		table.replace_dependencies("top", {"middle": 1})
		table.replace_dependencies("middle", {"leaf": 1})

		table.store(exact("leaf", 0, 5.0, solved=True))

		assert table.get("middle").solved is True
		assert table.get("top").solved is True
		assert table.get("top").score == 5.0


class TestDependencyPropagation:
	@pytest.mark.parametrize("levels", [2, 3, 5, 8])
	def test_information_reaches_every_level_above(self, levels: int) -> None:
		# n0 -> n1 -> ... -> n<levels>, each searched one ply deeper.
		table = tracked()

		for level in range(levels):
			table.store(bounds(f"n{level}", levels - level + 1))
			table.replace_dependencies(f"n{level}", {f"n{level + 1}": 1})

		outcome = table.store(exact(f"n{levels}", 1, 4.0))

		assert outcome == StoreOutcome(True, 0, levels)

		for level in range(levels):
			assert table.get(f"n{level}").score == 4.0

	def test_a_position_that_does_not_change_stops_the_propagation(self) -> None:
		table = tracked()
		table.store(exact("grand", 3, 5.0))
		table.store(exact("parent", 2, 5.0))
		table.replace_dependencies("grand", ["parent"])
		table.replace_dependencies("parent", {"child": 1})

		# Consistent with the parent, so nothing above it can be affected.
		outcome = table.store(bounds("child", 1, upper=7.0))

		assert outcome == StoreOutcome(True, 0, 0)
		assert table.get("grand").score_valid is True

	def test_a_position_that_changes_continues_the_propagation(self) -> None:
		table = tracked()
		table.store(exact("grand", 3, 5.0))
		table.store(exact("parent", 2, 5.0))
		table.replace_dependencies("grand", ["parent"])
		table.replace_dependencies("parent", {"child": 1})

		outcome = table.store(bounds("child", 1, upper=4.0))

		assert outcome == StoreOutcome(True, 1, 1)
		assert table.get("grand").score_valid is False

	def test_unrelated_positions_are_not_touched(self) -> None:
		table = tracked()
		table.store(bounds("parent", 2))
		table.store(bounds("bystander", 2))
		table.replace_dependencies("parent", {"child": 1})
		table.replace_dependencies("bystander", {"other": 1})
		before = table.get("bystander")

		table.store(exact("child", 1, 4.0))

		assert table.get("bystander") == before

	def test_descendants_are_not_touched(self) -> None:
		table = tracked()
		table.store(bounds("parent", 2))
		table.store(exact("child", 1, 4.0))
		table.replace_dependencies("parent", {"child": 1})
		before = table.get("child")

		table.store(bounds("parent", 2, lower=0.0))

		assert table.get("child") == before

	def test_diamond_dependencies_are_resolved_to_a_consistent_result(self) -> None:
		table = tracked()

		for key, depth in [("top", 3), ("left", 2), ("right", 2)]:
			table.store(bounds(key, depth))

		table.replace_dependencies("top", {"left": 1, "right": 1})
		table.replace_dependencies("left", {"bottom": 1})
		table.replace_dependencies("right", {"bottom": 1, "side": 1})

		table.store(exact("side", 1, 6.0))
		outcome = table.store(exact("bottom", 1, 2.0))

		assert table.get("left").score == 2.0
		assert table.get("right").score == 6.0
		assert table.get("top").score == 6.0
		assert outcome.refreshed == 3

	def test_a_position_is_only_updated_once_per_change(self) -> None:
		table = tracked()

		for key, depth in [("top", 3), ("left", 2), ("right", 2)]:
			table.store(bounds(key, depth))

		table.replace_dependencies("top", {"left": 1, "right": 1})
		table.replace_dependencies("left", {"bottom": 1})
		table.replace_dependencies("right", {"bottom": 1})

		outcome = table.store(exact("bottom", 1, 2.0))

		# left, right, and top: top is not refreshed once per path.
		assert outcome == StoreOutcome(True, 0, 3)

	def test_cyclic_dependencies_terminate(self) -> None:
		table = tracked()
		table.store(bounds("a", 1))
		table.store(bounds("b", 2))
		table.replace_dependencies("a", {"b": 1})
		table.replace_dependencies("b", {"a": 1})

		outcome = table.store(exact("a", 1, 5.0))

		assert outcome.stored is True
		assert table.get("b").score == 5.0

	def test_a_position_depending_on_itself_is_not_updated_by_itself(self) -> None:
		table = tracked()
		table.store(bounds("loop", 2))
		table.replace_dependencies("loop", {"loop": 1})

		outcome = table.store(exact("loop", 2, 3.0))

		assert outcome == StoreOutcome(True, 0, 0)
		assert table.get("loop").score == 3.0

	def test_a_new_set_of_dependencies_replaces_the_old_one(self) -> None:
		table = tracked()
		table.store(bounds("parent", 2))
		table.replace_dependencies("parent", {"old": 1})
		table.replace_dependencies("parent", {"new": 1})

		table.store(exact("old", 1, 9.0))

		assert table.get("parent").score is None

		table.store(exact("new", 1, 4.0))

		assert table.get("parent").score == 4.0

	def test_a_position_without_an_entry_is_skipped(self) -> None:
		table = tracked()
		table.replace_dependencies("ghost", {"child": 1})

		outcome = table.store(exact("child", 1, 4.0))

		assert outcome == StoreOutcome(True, 0, 0)
		assert table.get("ghost") is None

	def test_a_deeper_result_with_the_same_information_does_not_propagate(
		self,
	) -> None:
		table = tracked()
		table.store(exact("parent", 2, 3.0))
		table.replace_dependencies("parent", ["child"])
		table.store(exact("child", 1, 3.0))
		table.store(exact("parent", 2, 3.0))

		outcome = table.store(exact("child", 2, 3.0))

		assert outcome == StoreOutcome(True, 0, 0)
		assert table.get("parent").score_valid is True
		assert table.get("child").depth == 2

	def test_clear_discards_dependencies(self) -> None:
		table = tracked()
		table.store(bounds("parent", 2))
		table.replace_dependencies("parent", {"child": 1})

		table.clear()
		table.store(bounds("parent", 2))

		assert table.store(exact("child", 1, 4.0)) == StoreOutcome(True, 0, 0)
		assert table.get("parent").score is None

	def test_nothing_is_tracked_when_tracking_is_disabled(self) -> None:
		table = TranspositionTable(dependency_tracking=False)
		table.store(bounds("parent", 2))
		table.register_dependency("parent", "child", 1)
		table.replace_dependencies("parent", {"child": 1})

		outcome = table.store(exact("child", 1, 4.0))

		assert outcome == StoreOutcome(True)
		assert table.get("parent").score is None


class TestDependencyInvalidation:
	"""Dependencies without a known relation cannot be recomputed."""

	def test_unknown_relation_marks_the_position_stale(self) -> None:
		table = tracked()
		table.store(exact("parent", 3, 3.0))
		table.replace_dependencies("parent", ["child"])

		outcome = table.store(exact("child", 1, 1.0))

		assert outcome == StoreOutcome(True, 1, 0)
		assert table.get("parent").score_valid is False

	def test_registering_dependencies_does_not_invalidate(self) -> None:
		table = tracked()
		table.store(exact("parent", 3, 3.0))
		table.store(exact("child", 1, 9.0))

		outcome = table.replace_dependencies("parent", ["child"])

		assert outcome == StoreOutcome(False, 0, 0)
		assert table.get("parent").score_valid is True

	def test_the_rest_of_a_stale_entry_is_preserved(self) -> None:
		parent = replace(exact("parent", 3, 4.5), best_move="hint")
		table = tracked()
		table.store(parent)
		table.replace_dependencies("parent", ["child"])

		table.store(exact("child", 1, 1.0))

		assert table.get("parent") == replace(parent, score_valid=False)

	def test_a_stale_position_is_not_usable(self) -> None:
		table = tracked()
		table.store(exact("parent", 9, 3.0))
		table.register_dependency("parent", "child")

		table.store(exact("child", 1, 1.0))

		assert not table.get("parent").is_usable_for_depth(0)

	def test_one_unknown_relation_is_enough(self) -> None:
		table = tracked()
		table.store(exact("parent", 2, 3.0))
		table.replace_dependencies("parent", {"known": 1, "unknown": None})

		outcome = table.store(exact("known", 1, 9.0))

		assert outcome == StoreOutcome(True, 1, 0)

	@pytest.mark.parametrize("relation", [0, 2, -2, "same"])
	def test_a_relation_other_than_plus_or_minus_one_is_unknown(
		self,
		relation,
	) -> None:
		table = tracked()
		table.store(exact("parent", 2, 3.0))
		table.replace_dependencies("parent", {"child": relation})

		outcome = table.store(exact("child", 1, 9.0))

		assert outcome == StoreOutcome(True, 1, 0)

	def test_staleness_propagates_to_transitive_dependents(self) -> None:
		table = tracked()

		for key in ("grandparent", "parent"):
			table.store(exact(key, 3, 3.0))

		table.replace_dependencies("grandparent", ["parent"])
		table.replace_dependencies("parent", ["child"])

		outcome = table.store(exact("child", 1, 1.0))

		assert outcome == StoreOutcome(True, 2, 0)
		assert table.get("grandparent").score_valid is False
		assert table.get("parent").score_valid is False

	def test_every_parent_of_a_child_is_invalidated(self) -> None:
		table = tracked()

		for key in ("first", "second"):
			table.store(exact(key, 2, 3.0))
			table.replace_dependencies(key, ["child"])

		assert table.store(exact("child", 1, 1.0)).invalidated == 2

	def test_a_stale_position_is_not_counted_again(self) -> None:
		table = tracked()
		table.store(exact("parent", 3, 3.0))
		table.replace_dependencies("parent", ["child"])

		assert table.store(exact("child", 1, 1.0)).invalidated == 1
		assert table.store(exact("child", 2, 9.0)).invalidated == 0

	def test_a_stale_position_can_be_replaced_by_a_fresh_result(self) -> None:
		table = tracked()
		table.store(exact("parent", 9, 3.0))
		table.register_dependency("parent", "child")
		table.store(exact("child", 1, 1.0))

		table.store(exact("parent", 1, 7.0))

		assert table.get("parent").score_valid is True
		assert table.get("parent").score == 7.0


class TestDependencyOrderIndependence:
	EVENTS = [
		"link top",
		"link left",
		"link right",
		"store top",
		"store left",
		"store right",
		"store bottom",
	]

	@staticmethod
	def play(order) -> dict:
		table = tracked()

		actions = {
			"link top": lambda: table.replace_dependencies(
				"top",
				{"left": 1, "right": 1},
			),
			"link left": lambda: table.replace_dependencies("left", {"bottom": 1}),
			"link right": lambda: table.replace_dependencies("right", {"bottom": 1}),
			"store top": lambda: table.store(bounds("top", 3, lower=1.0)),
			"store left": lambda: table.store(bounds("left", 2)),
			"store right": lambda: table.store(bounds("right", 2, upper=9.0)),
			"store bottom": lambda: table.store(exact("bottom", 1, 2.0)),
		}

		for event in order:
			actions[event]()

		return {
			key: (entry.lower, entry.upper, entry.score_valid)
			for key in ("top", "left", "right", "bottom")
			if (entry := table.get(key)) is not None
		}

	def test_every_arrival_order_gives_the_same_result(self) -> None:
		expected = {
			"top": (2.0, 2.0, True),
			"left": (2.0, 2.0, True),
			"right": (2.0, 2.0, True),
			"bottom": (2.0, 2.0, True),
		}

		for order in permutations(self.EVENTS):
			assert self.play(order) == expected, order
