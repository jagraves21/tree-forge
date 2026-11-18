import pytest

import tree_forge
import tree_forge.algorithms
import tree_forge.caching
import tree_forge.core


SUBPACKAGES = [
	tree_forge.core,
	tree_forge.algorithms,
	tree_forge.caching,
]


class TestTopLevelPackageExports:
	def test_all_contains_no_duplicates(self) -> None:
		assert len(tree_forge.__all__) == len(set(tree_forge.__all__))

	def test_all_is_union_of_subpackage_exports(self) -> None:
		expected = {
			name
			for package in SUBPACKAGES
			for name in package.__all__
		}

		assert set(tree_forge.__all__) == expected

	@pytest.mark.parametrize("name", sorted(tree_forge.__all__))
	def test_every_exported_name_is_importable(self, name: str) -> None:
		assert hasattr(tree_forge, name)

	@pytest.mark.parametrize("package", SUBPACKAGES)
	def test_reexports_same_objects_as_subpackages(self, package) -> None:
		for name in package.__all__:
			assert getattr(tree_forge, name) is getattr(package, name)

	def test_star_import_exposes_public_api(self) -> None:
		namespace: dict = {}

		exec("from tree_forge import *", namespace)

		for name in tree_forge.__all__:
			assert name in namespace
