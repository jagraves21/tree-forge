from dataclasses import FrozenInstanceError, replace

import pytest

from tree_forge.caching.transposition_table import (
	TranspositionBound,
	TranspositionEntry,
	TranspositionTable,
)


EXACT = TranspositionBound.EXACT
LOWER = TranspositionBound.LOWER
UPPER = TranspositionBound.UPPER

KEY = "position"


def make_entry(
	key="position",
	depth=4,
	score=3.0,
	bound=EXACT,
	best_move="move",
	solved=False,
	score_valid=True,
) -> TranspositionEntry:
	return TranspositionEntry(
		key=key,
		depth=depth,
		score=score,
		bound=bound,
		best_move=best_move,
		solved=solved,
		score_valid=score_valid,
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
		entry = TranspositionEntry(
			key="position",
			depth=3,
			score=1.5,
			bound=EXACT,
		)

		assert entry.best_move is None
		assert entry.solved is False
		assert entry.score_valid is True

	def test_stores_all_fields(self) -> None:
		entry = make_entry(
			key=("board", 1),
			depth=7,
			score=-2.5,
			bound=LOWER,
			best_move="e2e4",
			solved=True,
			score_valid=False,
		)

		assert entry.key == ("board", 1)
		assert entry.depth == 7
		assert entry.score == -2.5
		assert entry.bound is LOWER
		assert entry.best_move == "e2e4"
		assert entry.solved is True
		assert entry.score_valid is False

	def test_is_immutable(self) -> None:
		entry = make_entry()

		with pytest.raises(FrozenInstanceError):
			entry.score = 10.0

	def test_entries_with_equal_fields_are_equal(self) -> None:
		assert make_entry() == make_entry()
		assert make_entry(score=1.0) != make_entry(score=2.0)

	def test_replace_creates_modified_copy(self) -> None:
		original = make_entry()

		invalidated = replace(original, score_valid=False)

		assert invalidated.score_valid is False
		assert invalidated.score == original.score
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

	@pytest.mark.parametrize("bound", list(TranspositionBound))
	def test_usability_does_not_depend_on_bound(self, bound) -> None:
		assert make_entry(depth=4, bound=bound).is_usable_for_depth(4)
		assert not make_entry(depth=4, bound=bound).is_usable_for_depth(5)


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
		first = make_entry(key="first", score=1.0)
		second = make_entry(key="second", score=2.0)

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

	def test_store_reports_no_invalidations_without_tracking(
		self,
		table: TranspositionTable,
	) -> None:
		assert table.store(make_entry(depth=1)) == 0
		assert table.store(make_entry(depth=5)) == 0

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


class TestTranspositionTableReplacement:
	"""Decision table for when a stored entry replaces an existing one."""

	@pytest.mark.parametrize(
		"existing, new, replaced",
		[
			# A proven (solved) result outranks a horizon-limited one.
			pytest.param(
				dict(depth=6, bound=EXACT),
				dict(depth=1, bound=LOWER, solved=True),
				True,
				id="solved-replaces-unsolved-even-if-shallower-and-weaker",
			),
			pytest.param(
				dict(depth=1, bound=LOWER, solved=True),
				dict(depth=9, bound=EXACT),
				False,
				id="unsolved-never-replaces-solved",
			),
			pytest.param(
				dict(solved=True, score=1.0),
				dict(solved=True, score=2.0),
				True,
				id="solved-exact-with-changed-score-replaces",
			),
			pytest.param(
				dict(solved=True, score=1.0),
				dict(solved=True, score=1.0),
				False,
				id="solved-exact-with-same-score-is-not-new-information",
			),
			pytest.param(
				dict(solved=True, bound=LOWER),
				dict(solved=True, bound=EXACT),
				True,
				id="solved-exact-replaces-solved-bound",
			),
			pytest.param(
				dict(solved=True, bound=EXACT),
				dict(solved=True, bound=LOWER, score=99.0),
				False,
				id="solved-bound-does-not-replace-solved-exact",
			),
			pytest.param(
				dict(depth=0, solved=True, score=1.0),
				dict(depth=9, solved=True, score=2.0),
				True,
				id="depth-is-ignored-between-solved-entries",
			),
			# Search depth for unsolved entries.
			pytest.param(
				dict(depth=4, score=3.0),
				dict(depth=6, score=3.0),
				True,
				id="deeper-replaces-shallower",
			),
			pytest.param(
				dict(depth=4, bound=LOWER),
				dict(depth=2, bound=EXACT),
				False,
				id="shallower-never-replaces-deeper-even-if-exact",
			),
			pytest.param(
				dict(depth=4, bound=EXACT),
				dict(depth=6, bound=LOWER),
				True,
				id="deeper-bound-replaces-shallower-exact",
			),
			pytest.param(
				dict(depth=5),
				dict(depth=None),
				True,
				id="unlimited-depth-replaces-finite-depth",
			),
			pytest.param(
				dict(depth=None),
				dict(depth=10),
				False,
				id="finite-depth-does-not-replace-unlimited-depth",
			),
			pytest.param(
				dict(depth=None, score=1.0),
				dict(depth=None, score=2.0),
				True,
				id="unlimited-depths-fall-through-to-exact-score-change",
			),
			pytest.param(
				dict(depth=None, score=1.0),
				dict(depth=None, score=1.0),
				False,
				id="unlimited-depths-with-identical-exact-score",
			),
			# Bound strength at equal depth.
			pytest.param(
				dict(bound=LOWER),
				dict(bound=EXACT),
				True,
				id="exact-replaces-lower-at-equal-depth",
			),
			pytest.param(
				dict(bound=UPPER),
				dict(bound=EXACT),
				True,
				id="exact-replaces-upper-at-equal-depth",
			),
			pytest.param(
				dict(bound=EXACT, score=3.0),
				dict(bound=EXACT, score=4.0),
				True,
				id="exact-with-changed-score-replaces-exact",
			),
			pytest.param(
				dict(bound=EXACT, score=3.0),
				dict(bound=EXACT, score=3.0),
				False,
				id="exact-with-same-score-does-not-replace",
			),
			pytest.param(
				dict(bound=EXACT),
				dict(bound=LOWER, score=9.0),
				False,
				id="lower-does-not-replace-exact",
			),
			pytest.param(
				dict(bound=EXACT),
				dict(bound=UPPER, score=-9.0),
				False,
				id="upper-does-not-replace-exact",
			),
			pytest.param(
				dict(bound=LOWER, score=2.0),
				dict(bound=LOWER, score=3.0),
				True,
				id="higher-lower-bound-is-stronger",
			),
			pytest.param(
				dict(bound=LOWER, score=3.0),
				dict(bound=LOWER, score=2.0),
				False,
				id="lower-lower-bound-is-weaker",
			),
			pytest.param(
				dict(bound=LOWER, score=3.0),
				dict(bound=LOWER, score=3.0),
				False,
				id="equal-lower-bounds-do-not-replace",
			),
			pytest.param(
				dict(bound=UPPER, score=3.0),
				dict(bound=UPPER, score=2.0),
				True,
				id="lower-upper-bound-is-stronger",
			),
			pytest.param(
				dict(bound=UPPER, score=2.0),
				dict(bound=UPPER, score=3.0),
				False,
				id="higher-upper-bound-is-weaker",
			),
			pytest.param(
				dict(bound=UPPER, score=3.0),
				dict(bound=UPPER, score=3.0),
				False,
				id="equal-upper-bounds-do-not-replace",
			),
			pytest.param(
				dict(bound=LOWER, score=1.0),
				dict(bound=UPPER, score=0.0),
				False,
				id="upper-does-not-replace-lower",
			),
			pytest.param(
				dict(bound=UPPER, score=1.0),
				dict(bound=LOWER, score=9.0),
				False,
				id="lower-does-not-replace-upper",
			),
			# Score validity.
			pytest.param(
				dict(depth=9, bound=EXACT, solved=True, score_valid=False),
				dict(depth=1, bound=UPPER),
				True,
				id="valid-entry-replaces-invalidated-entry",
			),
			pytest.param(
				dict(depth=1),
				dict(depth=9, solved=True, score_valid=False),
				False,
				id="invalid-entry-never-replaces-valid-entry",
			),
		],
	)
	def test_replacement_policy(
		self,
		existing: dict,
		new: dict,
		replaced: bool,
	) -> None:
		table = TranspositionTable()

		table.store(make_entry(best_move="old", **existing))
		table.store(make_entry(best_move="new", **{"score": 7.5, **new}))

		stored = table.get(KEY)

		assert stored is not None
		assert stored.best_move == ("new" if replaced else "old")

	def test_rejected_entry_leaves_existing_entry_untouched(self) -> None:
		table = TranspositionTable()
		deep = make_entry(depth=6, score=3.0)

		table.store(deep)
		table.store(make_entry(depth=2, score=-1.0))

		assert table.get(KEY) == deep

	def test_improved_entry_replaces_all_fields(self) -> None:
		table = TranspositionTable()
		better = make_entry(
			depth=8,
			score=5.0,
			bound=LOWER,
			best_move="better",
			solved=False,
		)

		table.store(make_entry(depth=2, score=1.0, best_move="worse"))
		table.store(better)

		assert table.get(KEY) == better

	def test_improved_entry_inherits_best_move_when_it_has_none(self) -> None:
		table = TranspositionTable()

		table.store(make_entry(depth=2, best_move="hint"))
		table.store(make_entry(depth=5, score=9.0, best_move=None))

		stored = table.get(KEY)

		assert stored is not None
		assert stored.depth == 5
		assert stored.score == 9.0
		assert stored.best_move == "hint"

	def test_improved_entry_prefers_its_own_best_move(self) -> None:
		table = TranspositionTable()

		table.store(make_entry(depth=2, best_move="old"))
		table.store(make_entry(depth=5, best_move="new"))

		stored = table.get(KEY)

		assert stored is not None
		assert stored.best_move == "new"

	def test_best_move_is_inherited_across_an_invalidated_entry(self) -> None:
		table = TranspositionTable()

		table.store(make_entry(best_move="hint", score_valid=False))
		table.store(make_entry(depth=1, best_move=None))

		stored = table.get(KEY)

		assert stored is not None
		assert stored.score_valid is True
		assert stored.best_move == "hint"

	def test_rejected_entry_does_not_alter_best_move(self) -> None:
		table = TranspositionTable()

		table.store(make_entry(depth=5, best_move="kept"))
		table.store(make_entry(depth=1, best_move="ignored"))
		table.store(make_entry(depth=1, best_move=None))

		stored = table.get(KEY)

		assert stored is not None
		assert stored.best_move == "kept"

	def test_first_entry_without_best_move_keeps_none(self) -> None:
		table = TranspositionTable()

		table.store(make_entry(best_move=None))

		stored = table.get(KEY)

		assert stored is not None
		assert stored.best_move is None

	def test_replacement_is_scoped_to_the_entry_key(self) -> None:
		table = TranspositionTable()
		other = make_entry(key="other", depth=9, score=1.0)

		table.store(other)
		table.store(make_entry(key="shared", depth=1))
		table.store(make_entry(key="shared", depth=5))

		assert table.get("other") == other
		assert table.get("shared").depth == 5


class TestTranspositionTableDependencies:
	@pytest.fixture
	def table(self) -> TranspositionTable:
		return TranspositionTable(dependency_tracking=True)

	@staticmethod
	def improve(table: TranspositionTable, key: str) -> int:
		"""Store a strictly deeper entry for a key and return the result."""
		current = table.get(key)
		depth = (current.depth if current is not None else 0) + 1

		return table.store(make_entry(key=key, depth=depth))

	def test_improved_child_invalidates_unsolved_parent(
		self,
		table: TranspositionTable,
	) -> None:
		table.store(make_entry(key="parent", depth=3))
		table.store(make_entry(key="child", depth=1))
		table.register_dependency("parent", "child")

		invalidated = self.improve(table, "child")

		assert invalidated == 1
		assert table.get("parent").score_valid is False

	def test_invalidation_preserves_the_rest_of_the_parent_entry(
		self,
		table: TranspositionTable,
	) -> None:
		parent = make_entry(
			key="parent",
			depth=3,
			score=4.5,
			bound=LOWER,
			best_move="hint",
		)

		table.store(parent)
		table.store(make_entry(key="child", depth=1))
		table.register_dependency("parent", "child")

		self.improve(table, "child")

		assert table.get("parent") == replace(parent, score_valid=False)

	def test_invalidated_parent_is_no_longer_usable(
		self,
		table: TranspositionTable,
	) -> None:
		table.store(make_entry(key="parent", depth=9))
		table.store(make_entry(key="child", depth=1))
		table.register_dependency("parent", "child")

		self.improve(table, "child")

		assert not table.get("parent").is_usable_for_depth(0)

	def test_invalidation_propagates_to_transitive_ancestors(
		self,
		table: TranspositionTable,
	) -> None:
		for key in ("grandparent", "parent", "child"):
			table.store(make_entry(key=key, depth=1))

		table.register_dependency("grandparent", "parent")
		table.register_dependency("parent", "child")

		invalidated = self.improve(table, "child")

		assert invalidated == 2
		assert table.get("parent").score_valid is False
		assert table.get("grandparent").score_valid is False

	def test_invalidates_every_parent_of_the_child(
		self,
		table: TranspositionTable,
	) -> None:
		for key in ("first", "second", "child"):
			table.store(make_entry(key=key, depth=1))

		table.register_dependency("first", "child")
		table.register_dependency("second", "child")

		assert self.improve(table, "child") == 2

	def test_solved_ancestors_are_preserved(
		self,
		table: TranspositionTable,
	) -> None:
		parent = make_entry(key="parent", depth=3, solved=True)

		table.store(parent)
		table.store(make_entry(key="child", depth=1))
		table.register_dependency("parent", "child")

		assert self.improve(table, "child") == 0
		assert table.get("parent") == parent

	def test_already_invalid_ancestors_are_not_counted_again(
		self,
		table: TranspositionTable,
	) -> None:
		table.store(make_entry(key="parent", depth=3))
		table.store(make_entry(key="child", depth=1))
		table.register_dependency("parent", "child")

		assert self.improve(table, "child") == 1
		assert self.improve(table, "child") == 0

	def test_ancestors_without_entries_are_ignored(
		self,
		table: TranspositionTable,
	) -> None:
		table.store(make_entry(key="child", depth=1))
		table.register_dependency("ghost", "child")

		assert self.improve(table, "child") == 0
		assert table.get("ghost") is None
		assert len(table) == 1

	def test_rejected_entry_does_not_invalidate_ancestors(
		self,
		table: TranspositionTable,
	) -> None:
		table.store(make_entry(key="parent", depth=3))
		table.store(make_entry(key="child", depth=5))
		table.register_dependency("parent", "child")

		invalidated = table.store(make_entry(key="child", depth=2))

		assert invalidated == 0
		assert table.get("parent").score_valid is True

	def test_unrelated_entries_are_not_invalidated(
		self,
		table: TranspositionTable,
	) -> None:
		table.store(make_entry(key="parent", depth=3))
		table.store(make_entry(key="bystander", depth=3))
		table.store(make_entry(key="child", depth=1))
		table.register_dependency("parent", "child")

		self.improve(table, "child")

		assert table.get("bystander").score_valid is True

	def test_descendants_are_not_invalidated(
		self,
		table: TranspositionTable,
	) -> None:
		table.store(make_entry(key="parent", depth=1))
		table.store(make_entry(key="child", depth=3))
		table.register_dependency("parent", "child")

		self.improve(table, "parent")

		assert table.get("child").score_valid is True

	def test_replace_dependencies_swaps_the_tracked_children(
		self,
		table: TranspositionTable,
	) -> None:
		for key in ("parent", "old", "new"):
			table.store(make_entry(key=key, depth=1))

		table.register_dependency("parent", "old")
		table.replace_dependencies("parent", {"new"})

		assert self.improve(table, "old") == 0
		assert table.get("parent").score_valid is True

		assert self.improve(table, "new") == 1
		assert table.get("parent").score_valid is False

	def test_replace_dependencies_accepts_any_set_type(
		self,
		table: TranspositionTable,
	) -> None:
		table.store(make_entry(key="parent", depth=1))
		table.store(make_entry(key="child", depth=1))

		table.replace_dependencies("parent", frozenset({"child"}))

		assert self.improve(table, "child") == 1

	def test_cyclic_dependencies_invalidate_the_other_member(
		self,
		table: TranspositionTable,
	) -> None:
		table.store(make_entry(key="a", depth=1))
		table.store(make_entry(key="b", depth=1))
		table.register_dependency("a", "b")
		table.register_dependency("b", "a")

		assert self.improve(table, "a") == 1
		assert table.get("b").score_valid is False
		assert table.get("a").score_valid is True

	def test_self_dependency_does_not_invalidate_the_improved_entry(
		self,
		table: TranspositionTable,
	) -> None:
		table.store(make_entry(key="loop", depth=1))
		table.register_dependency("loop", "loop")

		assert self.improve(table, "loop") == 0
		assert table.get("loop").score_valid is True

	def test_clear_discards_dependencies(
		self,
		table: TranspositionTable,
	) -> None:
		table.store(make_entry(key="parent", depth=1))
		table.store(make_entry(key="child", depth=1))
		table.register_dependency("parent", "child")

		table.clear()

		table.store(make_entry(key="parent", depth=1))
		table.store(make_entry(key="child", depth=1))

		assert self.improve(table, "child") == 0
		assert table.get("parent").score_valid is True

	def test_dependencies_are_ignored_when_tracking_is_disabled(self) -> None:
		table = TranspositionTable(dependency_tracking=False)

		table.store(make_entry(key="parent", depth=3))
		table.store(make_entry(key="child", depth=1))
		table.register_dependency("parent", "child")
		table.replace_dependencies("parent", {"child"})

		assert self.improve(table, "child") == 0
		assert table.get("parent").score_valid is True
