import pytest

import tree_forge.algorithms as algorithms
from tree_forge.algorithms import minimax, random_search


EXPECTED_EXPORTS = {
	"Minimax": minimax,
	"RandomSearch": random_search,
}


class TestAlgorithmsPackageExports:
	def test_all_declares_expected_public_names(self) -> None:
		assert set(algorithms.__all__) == set(EXPECTED_EXPORTS)

	def test_all_contains_no_duplicates(self) -> None:
		assert len(algorithms.__all__) == len(set(algorithms.__all__))

	@pytest.mark.parametrize("name", sorted(EXPECTED_EXPORTS))
	def test_exports_object_from_defining_module(self, name: str) -> None:
		defining_module = EXPECTED_EXPORTS[name]

		assert getattr(algorithms, name) is getattr(defining_module, name)
