import pytest

import tree_forge.core as core
from tree_forge.core import interfaces, models


EXPECTED_EXPORTS = {
	"Canonicalizer": interfaces,
	"Evaluator": interfaces,
	"Game": interfaces,
	"MoveOrdering": interfaces,
	"CanonicalPosition": models,
	"SearchAlgorithm": models,
	"SearchConfig": models,
	"SearchLimits": models,
	"SearchResult": models,
	"SearchStatistics": models,
	"SearchTerminationReason": models,
}


class TestCorePackageExports:
	def test_all_declares_expected_public_names(self) -> None:
		assert set(core.__all__) == set(EXPECTED_EXPORTS)

	def test_all_contains_no_duplicates(self) -> None:
		assert len(core.__all__) == len(set(core.__all__))

	@pytest.mark.parametrize("name", sorted(EXPECTED_EXPORTS))
	def test_exports_object_from_defining_module(self, name: str) -> None:
		defining_module = EXPECTED_EXPORTS[name]

		assert getattr(core, name) is getattr(defining_module, name)
