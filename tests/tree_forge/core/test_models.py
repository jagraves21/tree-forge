import pytest

from tree_forge.models import SearchConfig


def test_default_search_config():
	config = SearchConfig()

	assert config.max_depth == 1
	assert config.time_limit_seconds is None
	assert config.node_limit is None


def test_search_config_accepts_custom_values():
	config = SearchConfig(
		max_depth=5,
		time_limit_seconds=10.0,
		node_limit=1000,
	)

	assert config.max_depth == 5
	assert config.time_limit_seconds == 10.0
	assert config.node_limit == 1000


@pytest.mark.parametrize(
	"kwargs",
	[
		{"max_depth": 0},
		{"time_limit_seconds": 0},
		{"time_limit_seconds": -1},
		{"node_limit": 0},
		{"node_limit": -1},
	],
)
def test_search_config_rejects_invalid_values(kwargs):
	with pytest.raises(ValueError):
		SearchConfig(**kwargs)

