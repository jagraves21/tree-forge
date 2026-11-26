import sys

import pytest

from tree_forge import Minimax, TranspositionTable

from connect_four import (
	ConnectFourCanonicalizer,
	ConnectFourEvaluator,
	ConnectFourGame,
	ConnectFourMoveOrdering,
	Player,
	RandomConnectFourMoveOrdering,
	RandomizedConnectFourMoveOrdering,
)
from connect_four import main as main_module


ORDERINGS = {
	"tactical": ConnectFourMoveOrdering,
	"randomized": RandomizedConnectFourMoveOrdering,
	"random": RandomConnectFourMoveOrdering,
}

CONFIG_KEYS = (
	"depth",
	"alpha_beta",
	"move_ordering",
	"evaluator",
	"canonicalization",
	"transposition_table",
	"dependency_tracking",
)


def make_config(**overrides):
	config = {
		"depth": 2,
		"alpha_beta": False,
		"move_ordering": "none",
		"evaluator": False,
		"canonicalization": False,
		"transposition_table": False,
		"dependency_tracking": False,
	}
	config.update(overrides)

	return config


def parse(monkeypatch, *arguments):
	monkeypatch.setattr(sys, "argv", ["main", *arguments])

	return main_module.parse_arguments()


def run_main(**overrides):
	"""Run main() with two fast computer players and no delay."""
	options = {
		"red_player": "minimax",
		"yellow_player": "minimax",
		"depth": 2,
		"delay": 0,
		"quiet": True,
	}
	options.update(overrides)

	main_module.main(**options)


class TestResolvePlayerConfig:
	def test_uses_the_global_settings_when_the_player_sets_nothing(
		self,
	) -> None:
		player = {key: None for key in CONFIG_KEYS}

		config = main_module.resolve_player_config(
			player,
			5,
			True,
			"random",
			True,
			True,
			True,
			True,
		)

		assert config == {
			"depth": 5,
			"alpha_beta": True,
			"move_ordering": "random",
			"evaluator": True,
			"canonicalization": True,
			"transposition_table": True,
			"dependency_tracking": True,
		}

	def test_player_settings_override_the_global_settings(self) -> None:
		player = {
			"depth": 1,
			"alpha_beta": False,
			"move_ordering": "none",
			"evaluator": False,
			"canonicalization": False,
			"transposition_table": False,
			"dependency_tracking": False,
		}

		config = main_module.resolve_player_config(
			player,
			5,
			True,
			"random",
			True,
			True,
			True,
			True,
		)

		# A player's False must win over a global True.
		assert config == player

	def test_each_setting_is_resolved_independently(self) -> None:
		player = {key: None for key in CONFIG_KEYS}
		player["depth"] = 3
		player["move_ordering"] = "random"

		config = main_module.resolve_player_config(
			player,
			7,
			True,
			"none",
			False,
			False,
			False,
			False,
		)

		assert config["depth"] == 3
		assert config["move_ordering"] == "random"
		assert config["alpha_beta"] is True
		assert config["evaluator"] is False


class TestCreateMinimax:
	@pytest.fixture
	def game(self) -> ConnectFourGame:
		return ConnectFourGame()

	def test_without_enhancements_it_uses_the_defaults(
		self,
		game: ConnectFourGame,
	) -> None:
		minimax, table = main_module.create_minimax(game, make_config())

		assert isinstance(minimax, Minimax)
		assert minimax.move_ordering is None
		assert minimax.canonicalizer is None
		assert minimax.alpha_beta is False
		assert not isinstance(minimax.evaluator, ConnectFourEvaluator)
		assert table is None

	def test_applies_depth_and_alpha_beta(self, game: ConnectFourGame) -> None:
		minimax, _ = main_module.create_minimax(
			game,
			make_config(depth=3, alpha_beta=True),
		)

		assert minimax.limits.max_depth == 3
		assert minimax.alpha_beta is True

	def test_enables_the_game_evaluator(self, game: ConnectFourGame) -> None:
		minimax, _ = main_module.create_minimax(
			game,
			make_config(evaluator=True),
		)

		assert isinstance(minimax.evaluator, ConnectFourEvaluator)

	def test_enables_the_canonicalizer(self, game: ConnectFourGame) -> None:
		minimax, _ = main_module.create_minimax(
			game,
			make_config(canonicalization=True),
		)

		assert isinstance(minimax.canonicalizer, ConnectFourCanonicalizer)

	def test_creates_the_requested_transposition_table(
		self,
		game: ConnectFourGame,
	) -> None:
		minimax, table = main_module.create_minimax(
			game,
			make_config(canonicalization=True, transposition_table=True),
		)

		assert isinstance(table, TranspositionTable)
		assert minimax.transposition_table is table
		assert table.dependency_tracking is False

	def test_enables_dependency_tracking_on_the_table(
		self,
		game: ConnectFourGame,
	) -> None:
		_, table = main_module.create_minimax(
			game,
			make_config(
				canonicalization=True,
				transposition_table=True,
				dependency_tracking=True,
			),
		)

		assert table.dependency_tracking is True

	@pytest.mark.parametrize("name", sorted(ORDERINGS))
	def test_creates_the_requested_move_ordering(
		self,
		game: ConnectFourGame,
		name: str,
	) -> None:
		minimax, _ = main_module.create_minimax(
			game,
			make_config(move_ordering=name),
		)

		assert type(minimax.move_ordering) is ORDERINGS[name]

	def test_move_ordering_choices_match_the_available_orderings(
		self,
	) -> None:
		assert main_module.MOVE_ORDERING_TYPES == ORDERINGS
		assert set(main_module.MOVE_ORDERING_CHOICES) == {
			"none",
			*ORDERINGS,
		}


class TestCreatePlayers:
	def test_only_minimax_players_get_a_search(self) -> None:
		game = ConnectFourGame()
		configs = {
			Player.RED: make_config(),
			Player.YELLOW: make_config(),
		}

		players, ais, tables = main_module.create_players(
			game,
			"minimax",
			"random",
			configs,
		)

		assert players == {
			Player.RED: "minimax",
			Player.YELLOW: "random",
		}
		assert isinstance(ais[Player.RED], Minimax)
		assert ais[Player.YELLOW] is None
		assert tables == {Player.RED: None, Player.YELLOW: None}

	def test_each_minimax_player_gets_its_own_search_and_table(
		self,
	) -> None:
		game = ConnectFourGame()
		config = make_config(
			canonicalization=True,
			transposition_table=True,
		)
		configs = {
			Player.RED: config,
			Player.YELLOW: dict(config),
		}

		_, ais, tables = main_module.create_players(
			game,
			"minimax",
			"minimax",
			configs,
		)

		assert ais[Player.RED] is not ais[Player.YELLOW]
		assert tables[Player.RED] is not tables[Player.YELLOW]
		assert tables[Player.RED] is not None


class TestPlayGame:
	@pytest.mark.parametrize("attempt", range(5))
	def test_a_game_between_random_players_ends_with_a_result(
		self,
		capsys,
		attempt: int,
	) -> None:
		game = ConnectFourGame()
		players = {
			Player.RED: "random",
			Player.YELLOW: "random",
		}
		ais = {Player.RED: None, Player.YELLOW: None}

		winner = main_module.play_game(
			game,
			players,
			ais,
			delay=0,
			quiet=True,
		)

		output = capsys.readouterr().out.strip()

		assert winner in (None, Player.RED, Player.YELLOW)
		assert output.endswith("wins!") or output.endswith("draw!")

	def test_quiet_mode_prints_no_boards(self, capsys) -> None:
		game = ConnectFourGame()
		players = {
			Player.RED: "random",
			Player.YELLOW: "random",
		}
		ais = {Player.RED: None, Player.YELLOW: None}

		main_module.play_game(game, players, ais, delay=0, quiet=True)

		assert len(capsys.readouterr().out.strip().splitlines()) == 1

	def test_collects_statistics_for_searching_players_only(self) -> None:
		game = ConnectFourGame()
		config = make_config()
		players, ais, _ = main_module.create_players(
			game,
			"minimax",
			"random",
			{Player.RED: config, Player.YELLOW: config},
		)
		totals = {
			Player.RED: main_module.defaultdict(list),
			Player.YELLOW: main_module.defaultdict(list),
		}

		main_module.play_game(
			game,
			players,
			ais,
			delay=0,
			statistics_totals=totals,
			quiet=True,
		)

		assert len(totals[Player.RED]["search_time"]) > 0
		assert "nodes_visited" in totals[Player.RED]
		assert not totals[Player.YELLOW]


class TestParseArguments:
	def test_defaults(self, monkeypatch) -> None:
		args = parse(monkeypatch)

		assert args["red_player"] == "human"
		assert args["yellow_player"] == "human"
		assert args["games"] == 1
		assert args["quiet"] is False
		assert args["move_ordering"] == "none"
		assert args["red_depth"] is None
		assert args["red_alpha_beta"] is None

	def test_player_specific_options_are_parsed(self, monkeypatch) -> None:
		args = parse(
			monkeypatch,
			"--red-player",
			"minimax",
			"--red-depth",
			"3",
			"--yellow-move-ordering",
			"tactical",
			"--no-red-alpha-beta",
		)

		assert args["red_player"] == "minimax"
		assert args["red_depth"] == 3
		assert args["yellow_move_ordering"] == "tactical"
		assert args["red_alpha_beta"] is False

	@pytest.mark.parametrize("name", [*ORDERINGS, "none"])
	def test_accepts_every_move_ordering(self, monkeypatch, name) -> None:
		args = parse(monkeypatch, "--move-ordering", name)

		assert args["move_ordering"] == name

	def test_rejects_an_unknown_move_ordering(
		self,
		monkeypatch,
		capsys,
	) -> None:
		with pytest.raises(SystemExit) as error:
			parse(monkeypatch, "--move-ordering", "bogus")

		assert error.value.code == 2
		assert "invalid choice" in capsys.readouterr().err

	def test_accepts_a_valid_table_configuration(self, monkeypatch) -> None:
		args = parse(
			monkeypatch,
			"--red-player",
			"minimax",
			"--canonicalization",
			"--transposition-table",
			"--dependency-tracking",
		)

		assert args["transposition_table"] is True

	@pytest.mark.parametrize("arguments, message", [
		pytest.param(
			("--games", "0"),
			"--games must be at least 1",
			id="no-games",
		),
		pytest.param(
			("--delay", "-1"),
			"--delay cannot be negative",
			id="negative-delay",
		),
		pytest.param(
			("--games", "2"),
			"requires both players to be computer-controlled",
			id="several-games-with-humans",
		),
		pytest.param(
			("--red-player", "minimax", "--dependency-tracking"),
			"RED dependency tracking requires the transposition table",
			id="tracking-without-table",
		),
		pytest.param(
			("--yellow-player", "minimax", "--transposition-table"),
			"YELLOW currently requires canonicalization",
			id="table-without-canonicalization",
		),
		pytest.param(
			(
				"--red-player",
				"minimax",
				"--canonicalization",
				"--no-red-transposition-table",
				"--red-dependency-tracking",
			),
			"RED dependency tracking requires the transposition table",
			id="player-setting-disables-the-table",
		),
	])
	def test_rejects_invalid_combinations(
		self,
		monkeypatch,
		capsys,
		arguments,
		message,
	) -> None:
		with pytest.raises(SystemExit) as error:
			parse(monkeypatch, *arguments)

		assert error.value.code == 2
		assert message in capsys.readouterr().err

	def test_settings_of_non_minimax_players_are_not_validated(
		self,
		monkeypatch,
	) -> None:
		args = parse(
			monkeypatch,
			"--red-player",
			"random",
			"--yellow-player",
			"human",
			"--dependency-tracking",
			"--transposition-table",
		)

		assert args["dependency_tracking"] is True

	def test_several_games_are_allowed_for_computer_players(
		self,
		monkeypatch,
	) -> None:
		args = parse(
			monkeypatch,
			"--red-player",
			"random",
			"--yellow-player",
			"minimax",
			"--games",
			"3",
		)

		assert args["games"] == 3


class TestMainValidation:
	@pytest.mark.parametrize("keyword", ["red_player", "yellow_player"])
	def test_rejects_an_unknown_player_type(self, keyword: str) -> None:
		with pytest.raises(ValueError, match="Invalid"):
			run_main(**{keyword: "robot"})

	def test_rejects_no_games(self) -> None:
		with pytest.raises(ValueError, match="--games"):
			run_main(games=0)

	def test_rejects_a_negative_delay(self) -> None:
		with pytest.raises(ValueError, match="--delay"):
			run_main(delay=-1)

	def test_rejects_dependency_tracking_without_a_table(self) -> None:
		with pytest.raises(ValueError, match="dependency tracking"):
			run_main(dependency_tracking=True)

	def test_rejects_a_table_without_canonicalization(self) -> None:
		with pytest.raises(ValueError, match="canonicalization"):
			run_main(transposition_table=True)

	def test_rejects_several_games_with_a_human_player(self) -> None:
		with pytest.raises(ValueError, match="--games"):
			run_main(red_player="human", games=2)


class TestMainSmoke:
	"""Run whole games to check that every feature works together.

	These only check that the games finish without raising. They do not
	check who wins or what is printed.
	"""

	def test_minimax_against_minimax(self) -> None:
		run_main()

	def test_minimax_against_random(self) -> None:
		run_main(yellow_player="random")

	def test_random_against_random(self) -> None:
		run_main(red_player="random", yellow_player="random")

	@pytest.mark.parametrize("ordering", ["none", *ORDERINGS])
	def test_alpha_beta_with_evaluator_and_move_ordering(
		self,
		ordering: str,
	) -> None:
		run_main(
			alpha_beta=True,
			evaluator=True,
			move_ordering=ordering,
		)

	def test_canonicalization_alone(self) -> None:
		run_main(canonicalization=True)

	def test_transposition_table(self) -> None:
		run_main(canonicalization=True, transposition_table=True)

	def test_transposition_table_with_dependency_tracking(self) -> None:
		run_main(
			alpha_beta=True,
			canonicalization=True,
			transposition_table=True,
			dependency_tracking=True,
		)

	@pytest.mark.parametrize("ordering", sorted(ORDERINGS))
	def test_players_with_different_settings(self, ordering: str) -> None:
		run_main(
			red_depth=3,
			yellow_depth=1,
			red_alpha_beta=True,
			yellow_evaluator=True,
			yellow_move_ordering=ordering,
		)

	def test_several_games_print_the_match_summary(self, capsys) -> None:
		run_main(
			games=2,
			canonicalization=True,
			transposition_table=True,
		)

		output = capsys.readouterr().out

		assert "MATCH RESULTS" in output
		assert "Games played:" in output
		assert "TRANSPOSITION TABLE SIZES" in output

	def test_a_single_game_prints_no_match_summary(self, capsys) -> None:
		run_main()

		assert "MATCH RESULTS" not in capsys.readouterr().out

	def test_games_share_the_transposition_table(self, capsys) -> None:
		run_main(
			games=2,
			canonicalization=True,
			transposition_table=True,
		)

		sizes = capsys.readouterr().out.split(
			"TRANSPOSITION TABLE SIZES",
		)[1]

		assert "RED: 0" not in sizes
		assert "YELLOW: 0" not in sizes
