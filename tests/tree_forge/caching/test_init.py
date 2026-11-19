import pytest

import tree_forge.caching as caching
from tree_forge.caching import search_dependency_tracker, transposition_table


EXPECTED_EXPORTS = {
	"TranspositionBound": transposition_table,
	"TranspositionEntry": transposition_table,
	"TranspositionTable": transposition_table,
	"StoreOutcome": transposition_table,
	"SearchDependencyTracker": search_dependency_tracker,
}


class TestCachingPackageExports:
	def test_all_declares_expected_public_names(self) -> None:
		assert set(caching.__all__) == set(EXPECTED_EXPORTS)

	def test_all_contains_no_duplicates(self) -> None:
		assert len(caching.__all__) == len(set(caching.__all__))

	@pytest.mark.parametrize("name", sorted(EXPECTED_EXPORTS))
	def test_exports_object_from_defining_module(self, name: str) -> None:
		defining_module = EXPECTED_EXPORTS[name]

		assert getattr(caching, name) is getattr(defining_module, name)
