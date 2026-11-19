import pytest

from tree_forge.caching.search_dependency_tracker import (
	SearchDependencyTracker,
)


@pytest.fixture
def tracker() -> SearchDependencyTracker:
	return SearchDependencyTracker()


class TestRegisterDependency:
	def test_registered_parent_is_ancestor_of_child(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "child")

		assert tracker.get_ancestors("child") == {"parent"}

	def test_child_is_not_ancestor_of_parent(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "child")

		assert tracker.get_ancestors("parent") == set()

	def test_child_may_have_several_parents(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("first", "child")
		tracker.register_dependency("second", "child")

		assert tracker.get_ancestors("child") == {"first", "second"}

	def test_parent_may_have_several_children(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "first")
		tracker.register_dependency("parent", "second")

		assert tracker.get_ancestors("first") == {"parent"}
		assert tracker.get_ancestors("second") == {"parent"}

	def test_registering_same_dependency_twice_has_no_extra_effect(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "child")
		tracker.register_dependency("parent", "child")

		assert tracker.get_ancestors("child") == {"parent"}

		tracker.remove_dependencies("parent")

		assert tracker.get_ancestors("child") == set()

	def test_accepts_any_hashable_key_type(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency(("board", 1), 42)
		tracker.register_dependency(42, "leaf")

		assert tracker.get_ancestors("leaf") == {42, ("board", 1)}

	def test_rejects_unhashable_keys(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		with pytest.raises(TypeError):
			tracker.register_dependency(["unhashable"], "child")


class TestGetAncestors:
	def test_unknown_key_has_no_ancestors(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		assert tracker.get_ancestors("missing") == set()

	def test_returns_every_transitive_ancestor(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("root", "middle")
		tracker.register_dependency("middle", "leaf")

		assert tracker.get_ancestors("leaf") == {"root", "middle"}
		assert tracker.get_ancestors("middle") == {"root"}

	def test_excludes_unrelated_branches(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("root", "left")
		tracker.register_dependency("root", "right")
		tracker.register_dependency("left", "left_leaf")

		assert tracker.get_ancestors("left_leaf") == {"root", "left"}

	def test_excludes_descendants(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("root", "middle")
		tracker.register_dependency("middle", "leaf")

		assert "leaf" not in tracker.get_ancestors("middle")

	def test_diamond_ancestors_are_reported_once(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("top", "left")
		tracker.register_dependency("top", "right")
		tracker.register_dependency("left", "bottom")
		tracker.register_dependency("right", "bottom")

		assert tracker.get_ancestors("bottom") == {"top", "left", "right"}

	def test_terminates_on_cyclic_dependencies(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("a", "b")
		tracker.register_dependency("b", "c")
		tracker.register_dependency("c", "a")

		assert tracker.get_ancestors("a") == {"a", "b", "c"}

	def test_self_dependency_makes_key_its_own_ancestor(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("loop", "loop")

		assert tracker.get_ancestors("loop") == {"loop"}

	def test_handles_deep_chains_without_recursion_errors(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		depth = 5_000

		for index in range(depth):
			tracker.register_dependency(index, index + 1)

		assert tracker.get_ancestors(depth) == set(range(depth))

	def test_returned_set_is_a_copy(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "child")

		tracker.get_ancestors("child").add("intruder")

		assert tracker.get_ancestors("child") == {"parent"}

	def test_querying_unknown_key_does_not_register_it(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.get_ancestors("ghost")
		tracker.register_dependency("ghost", "child")

		assert tracker.get_ancestors("child") == {"ghost"}
		assert tracker.get_ancestors("ghost") == set()


class TestRemoveDependencies:
	def test_removes_all_children_of_parent(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "first")
		tracker.register_dependency("parent", "second")

		tracker.remove_dependencies("parent")

		assert tracker.get_ancestors("first") == set()
		assert tracker.get_ancestors("second") == set()

	def test_preserves_other_parents_of_the_same_child(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("removed", "child")
		tracker.register_dependency("kept", "child")

		tracker.remove_dependencies("removed")

		assert tracker.get_ancestors("child") == {"kept"}

	def test_preserves_dependencies_of_other_parents(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("removed", "first")
		tracker.register_dependency("kept", "second")

		tracker.remove_dependencies("removed")

		assert tracker.get_ancestors("second") == {"kept"}

	def test_does_not_remove_links_to_the_parents_own_ancestors(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("grandparent", "parent")
		tracker.register_dependency("parent", "child")

		tracker.remove_dependencies("parent")

		assert tracker.get_ancestors("parent") == {"grandparent"}
		assert tracker.get_ancestors("child") == set()

	def test_removing_unknown_parent_is_a_no_op(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "child")

		tracker.remove_dependencies("missing")

		assert tracker.get_ancestors("child") == {"parent"}

	def test_removed_dependency_can_be_registered_again(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "child")
		tracker.remove_dependencies("parent")
		tracker.register_dependency("parent", "child")

		assert tracker.get_ancestors("child") == {"parent"}


class TestReplaceDependencies:
	def test_replaces_existing_children(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "old")

		tracker.replace_dependencies("parent", {"new"})

		assert tracker.get_ancestors("old") == set()
		assert tracker.get_ancestors("new") == {"parent"}

	def test_registers_children_for_new_parent(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.replace_dependencies("parent", {"first", "second"})

		assert tracker.get_ancestors("first") == {"parent"}
		assert tracker.get_ancestors("second") == {"parent"}

	def test_empty_replacement_clears_the_parents_dependencies(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "child")

		tracker.replace_dependencies("parent", set())

		assert tracker.get_ancestors("child") == set()

	def test_keeps_children_present_in_both_sets(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "shared")
		tracker.register_dependency("parent", "old")

		tracker.replace_dependencies("parent", {"shared", "new"})

		assert tracker.get_ancestors("shared") == {"parent"}
		assert tracker.get_ancestors("old") == set()
		assert tracker.get_ancestors("new") == {"parent"}

	def test_does_not_affect_other_parents(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "child")
		tracker.register_dependency("other", "child")

		tracker.replace_dependencies("parent", set())

		assert tracker.get_ancestors("child") == {"other"}

	def test_does_not_retain_a_reference_to_the_given_set(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		children = {"child"}

		tracker.replace_dependencies("parent", children)
		children.add("later")

		assert tracker.get_ancestors("later") == set()


class TestClear:
	def test_removes_every_dependency(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("root", "middle")
		tracker.register_dependency("middle", "leaf")

		tracker.clear()

		assert tracker.get_ancestors("leaf") == set()
		assert tracker.get_ancestors("middle") == set()

	def test_clearing_empty_tracker_is_a_no_op(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.clear()

		assert tracker.get_ancestors("anything") == set()

	def test_tracker_is_usable_after_clear(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "child")
		tracker.clear()
		tracker.register_dependency("other", "child")

		assert tracker.get_ancestors("child") == {"other"}


class TestRelations:
	def test_dependency_has_no_relation_by_default(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "child")

		assert tracker.get_children("parent") == {"child": None}

	def test_stores_the_relation_of_a_dependency(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "child", -1)

		assert tracker.get_children("parent") == {"child": -1}

	def test_registering_again_replaces_the_relation(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "child", 1)
		tracker.register_dependency("parent", "child", -1)

		assert tracker.get_children("parent") == {"child": -1}

	def test_replace_dependencies_accepts_a_mapping_with_relations(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.replace_dependencies("parent", {"first": 1, "second": -1})

		assert tracker.get_children("parent") == {"first": 1, "second": -1}
		assert tracker.get_ancestors("second") == {"parent"}

	def test_replace_dependencies_with_keys_leaves_relations_unknown(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.replace_dependencies("parent", ["first", "second"])

		assert tracker.get_children("parent") == {"first": None, "second": None}

	def test_replacing_drops_old_relations(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.replace_dependencies("parent", {"old": 1})
		tracker.replace_dependencies("parent", {"new": -1})

		assert tracker.get_children("parent") == {"new": -1}


class TestGetChildren:
	def test_unknown_parent_has_no_children(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		assert tracker.get_children("missing") == {}

	def test_returns_only_direct_children(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("root", "middle")
		tracker.register_dependency("middle", "leaf")

		assert set(tracker.get_children("root")) == {"middle"}

	def test_returned_mapping_is_a_copy(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "child", 1)

		tracker.get_children("parent")["intruder"] = 1

		assert tracker.get_children("parent") == {"child": 1}

	def test_removed_children_are_not_reported(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "child")

		tracker.remove_dependencies("parent")

		assert tracker.get_children("parent") == {}

	def test_cleared_children_are_not_reported(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "child")

		tracker.clear()

		assert tracker.get_children("parent") == {}


class TestGetUpdateOrder:
	def test_unknown_key_has_nothing_to_update(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		assert tracker.get_update_order("missing") == []

	def test_excludes_the_changed_key(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "child")

		assert tracker.get_update_order("child") == ["parent"]

	def test_chain_is_ordered_from_nearest_to_furthest_ancestor(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("great", "grand")
		tracker.register_dependency("grand", "parent")
		tracker.register_dependency("parent", "child")

		assert tracker.get_update_order("child") == [
			"parent",
			"grand",
			"great",
		]

	def test_contains_exactly_the_ancestors(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("top", "left")
		tracker.register_dependency("top", "right")
		tracker.register_dependency("left", "bottom")
		tracker.register_dependency("right", "bottom")
		tracker.register_dependency("bottom", "leaf")
		tracker.register_dependency("unrelated", "other")

		order = tracker.get_update_order("leaf")

		assert sorted(order) == sorted(tracker.get_ancestors("leaf"))
		assert len(order) == len(set(order))

	def test_every_position_follows_the_positions_it_depends_on(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		edges = [
			("top", "left"),
			("top", "right"),
			("top", "bottom"),
			("left", "bottom"),
			("right", "bottom"),
			("right", "side"),
			("side", "bottom"),
			("bottom", "leaf"),
		]

		for parent, child in edges:
			tracker.register_dependency(parent, child)

		order = tracker.get_update_order("leaf")
		position = {key: index for index, key in enumerate(order)}

		for parent, child in edges:
			if child in position:
				assert position[child] < position[parent]

	def test_diamond_is_ordered_before_its_top(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("top", "left")
		tracker.register_dependency("top", "right")
		tracker.register_dependency("left", "bottom")
		tracker.register_dependency("right", "bottom")

		order = tracker.get_update_order("bottom")

		assert order[-1] == "top"
		assert set(order[:2]) == {"left", "right"}

	def test_terminates_on_cycles_and_lists_each_position_once(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("a", "b")
		tracker.register_dependency("b", "c")
		tracker.register_dependency("c", "a")

		order = tracker.get_update_order("a")

		assert sorted(order) == ["b", "c"]

	def test_self_dependency_does_not_list_the_key(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("loop", "loop")

		assert tracker.get_update_order("loop") == []

	def test_handles_deep_chains_without_recursion_errors(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		depth = 20_000

		for index in range(depth):
			tracker.register_dependency(index, index + 1)

		order = tracker.get_update_order(depth)

		assert order == list(range(depth - 1, -1, -1))

	def test_does_not_modify_the_tracker(
		self,
		tracker: SearchDependencyTracker,
	) -> None:
		tracker.register_dependency("parent", "child", 1)

		tracker.get_update_order("child")

		assert tracker.get_children("parent") == {"child": 1}
		assert tracker.get_ancestors("child") == {"parent"}
