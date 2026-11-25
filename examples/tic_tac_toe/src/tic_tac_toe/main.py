import argparse
import random
from collections import defaultdict
from statistics import mean, stdev
from time import perf_counter, sleep

from tree_forge import Minimax, SearchLimits, TranspositionTable

from . import (
	TicTacToeCanonicalizer,
	TicTacToeEvaluator,
	TicTacToeGame,
)

from .move_ordering import (
	RandomTicTacToeMoveOrdering,
	RandomizedTicTacToeMoveOrdering,
	TicTacToeMoveOrdering,
)

from .game import Player


MOVE_ORDERING_TYPES = {
	"positional": TicTacToeMoveOrdering,
	"randomized": RandomizedTicTacToeMoveOrdering,
	"random": RandomTicTacToeMoveOrdering,
}

MOVE_ORDERING_CHOICES = (
	"none",
	"positional",
	"randomized",
	"random",
)


STATISTIC_FIELDS = (
	"nodes_visited",
	"leaf_nodes",
	"terminal_nodes",
	"cutoffs",
	"transposition_table_hits",
	"transposition_table_stores",
	"transposition_table_invalidations",
	"transposition_table_refreshes",
)


class ArgumentParser(argparse.ArgumentParser):
	def error(self, message):
		self.print_usage()
		self.exit(2, f"{self.prog}: error: {message}\n")


def resolve_player_config(
	player,
	depth,
	alpha_beta,
	move_ordering,
	evaluator,
	canonicalization,
	transposition_table,
	dependency_tracking,
):
	return {
		"depth": (
			depth
			if player["depth"] is None
			else player["depth"]
		),
		"alpha_beta": (
			alpha_beta
			if player["alpha_beta"] is None
			else player["alpha_beta"]
		),
		"move_ordering": (
			move_ordering
			if player["move_ordering"] is None
			else player["move_ordering"]
		),
		"evaluator": (
			evaluator
			if player["evaluator"] is None
			else player["evaluator"]
		),
		"canonicalization": (
			canonicalization
			if player["canonicalization"] is None
			else player["canonicalization"]
		),
		"transposition_table": (
			transposition_table
			if player["transposition_table"] is None
			else player["transposition_table"]
		),
		"dependency_tracking": (
			dependency_tracking
			if player["dependency_tracking"] is None
			else player["dependency_tracking"]
		),
	}


def create_minimax(game, config):
	canonicalizer = (
		TicTacToeCanonicalizer()
		if config["canonicalization"]
		else None
	)

	transposition_table = (
		TranspositionTable(
			dependency_tracking=config["dependency_tracking"]
		)
		if config["transposition_table"]
		else None
	)

	move_ordering_type = MOVE_ORDERING_TYPES.get(
		config["move_ordering"]
	)

	move_ordering = (
		move_ordering_type()
		if move_ordering_type is not None
		else None
	)

	minimax = Minimax(
		game,
		evaluator=(
			TicTacToeEvaluator()
			if config["evaluator"]
			else None
		),
		limits=SearchLimits(max_depth=config["depth"]),
		alpha_beta=config["alpha_beta"],
		canonicalizer=canonicalizer,
		transposition_table=transposition_table,
		move_ordering=move_ordering,
	)

	return minimax, transposition_table


def create_players(game, x_player, o_player, configs):
	players = {}
	ais = {}
	transposition_tables = {}

	player_types = {
		Player.X: x_player,
		Player.O: o_player,
	}

	for player, player_type in player_types.items():
		if player_type == "minimax":
			ais[player], transposition_tables[player] = create_minimax(
				game,
				configs[player],
			)
		else:
			ais[player] = None
			transposition_tables[player] = None

		players[player] = player_type

	return players, ais, transposition_tables


def get_human_move(game, state):
	legal_moves = list(game.legal_moves(state))

	if not legal_moves:
		return None

	print("\nAvailable moves:")

	for index, move in enumerate(legal_moves, start=1):
		print(f"  {index}. {move}")

	while True:
		choice = input("\nChoose a move: ").strip()

		try:
			index = int(choice) - 1

			if 0 <= index < len(legal_moves):
				return legal_moves[index]

		except ValueError:
			pass

		print("Invalid choice. Enter one of the listed move numbers.")


def print_search_statistics(result):
	print(result)


def update_statistics(totals, player, statistics, duration):
	if statistics is None:
		return

	# Each search contributes one sample per statistic. A statistic
	# that was not measured (None) contributes no sample.
	for field in STATISTIC_FIELDS:
		value = getattr(statistics, field, None)

		if value is not None:
			totals[player][field].append(value)

	totals[player]["search_time"].append(duration)


def summarize(values, precision=1):
	"""Return the min, max and mean +- std of a list of samples."""
	if not values:
		return "n/a"

	spread = stdev(values) if len(values) > 1 else 0.0

	return (
		f"min {min(values):>10,.{precision}f}  "
		f"max {max(values):>10,.{precision}f}  "
		f"mean {mean(values):>10,.{precision}f} "
		f"± {spread:,.{precision}f}"
	)


def play_game(
	game,
	players,
	ais,
	player_names=None,
	delay=0.0,
	game_number=1,
	statistics_totals=None,
	quiet=False,
):
	state = game.initial_state()
	turn_number = 1

	if player_names is None:
		player_names = {
			Player.X: "AI 1",
			Player.O: "AI 2",
		}

	if not quiet:
		print(f"\nStarting Game {game_number}:")
		print(f"Turn: {turn_number}")
		print()
		print(state)
		print()

	while not game.is_terminal(state):
		player = game.current_player(state)
		player_type = players[player]

		search_duration = 0.0
		result = None

		if player_type == "human":
			move = get_human_move(game, state)

		elif player_type == "random":
			legal_moves = game.legal_moves(state)
			move = random.choice(legal_moves) if legal_moves else None

		else:
			search = ais[player]

			search_start = perf_counter()
			result = search.search(state)
			search_duration = perf_counter() - search_start

			move = result.best_move

		if move is None:
			break

		game.make_move(state, move)

		if result is not None:
			if not quiet:
				print_search_statistics(result)

			if statistics_totals is not None:
				update_statistics(
					statistics_totals,
					player,
					result.statistics,
					search_duration,
				)

		if not quiet:
			print()
			print("Current Game:")

			turn_number += 1

			if not game.is_terminal(state):
				next_player = game.current_player(state)
				print(f"Turn: {turn_number}")
				print(
					f"Current Player: {player_names[next_player]} "
					f"({next_player.name})"
				)
			else:
				print("Game Over")

			print()
			print(state)
			print()

		remaining_delay = delay - search_duration

		if remaining_delay > 0:
			sleep(remaining_delay)

	winner = game.winner(state)

	if winner is None:
		print("Game ended in a draw!")
	else:
		print(f"{player_names[winner]} ({winner.name}) wins!")

	return winner


def print_aggregate_statistics(
	results,
	statistics_totals,
	transposition_tables,
):
	total_games = sum(results.values())

	print("\n" + "=" * 50)
	print("MATCH RESULTS")
	print("=" * 50)

	print(f"{'Games played:':<32} {total_games}")
	print(f"{'X wins:':<32} {results[Player.X]}")
	print(f"{'O wins:':<32} {results[Player.O]}")
	print(f"{'Draws:':<32} {results[None]}")

	print("\nSEARCH STATISTICS (per search: min, max, mean ± std)")

	for player in (Player.X, Player.O):
		totals = statistics_totals[player]
		search_times = totals["search_time"]

		print(f"\n{player.name}:")

		if not search_times:
			print("  No searches performed.")
			continue

		print(f"  Searches: {len(search_times)}")
		print(f"  Total search time: {sum(search_times):.6f}s")
		print(f"  {'Search time (s)':<36} {summarize(search_times, 6)}")

		for field in STATISTIC_FIELDS:
			label = field.replace("_", " ").title()
			print(f"  {label:<36} {summarize(totals[field])}")

	print("\nTRANSPOSITION TABLE SIZES")

	for player in (Player.X, Player.O):
		table = transposition_tables[player]
		size = len(table) if table is not None else 0

		print(f"  {player.name}: {size:,}")

	print("\n" + "=" * 50)


def main(
	x_player="human",
	o_player="human",
	depth=9,
	x_depth=None,
	o_depth=None,
	alpha_beta=False,
	x_alpha_beta=None,
	o_alpha_beta=None,
	move_ordering="none",
	x_move_ordering=None,
	o_move_ordering=None,
	evaluator=False,
	x_evaluator=None,
	o_evaluator=None,
	canonicalization=False,
	x_canonicalization=None,
	o_canonicalization=None,
	transposition_table=False,
	x_transposition_table=None,
	o_transposition_table=None,
	dependency_tracking=False,
	x_dependency_tracking=None,
	o_dependency_tracking=None,
	games=1,
	delay=0.5,
	quiet=False,
):
	if x_player not in ("human", "minimax", "random"):
		raise ValueError(f"Invalid X player: {x_player}")

	if o_player not in ("human", "minimax", "random"):
		raise ValueError(f"Invalid O player: {o_player}")

	if games < 1:
		raise ValueError("--games must be at least 1")

	if delay < 0:
		raise ValueError("--delay cannot be negative")

	x_config = resolve_player_config(
		{
			"depth": x_depth,
			"alpha_beta": x_alpha_beta,
			"move_ordering": x_move_ordering,
			"evaluator": x_evaluator,
			"canonicalization": x_canonicalization,
			"transposition_table": x_transposition_table,
			"dependency_tracking": x_dependency_tracking,
		},
		depth,
		alpha_beta,
		move_ordering,
		evaluator,
		canonicalization,
		transposition_table,
		dependency_tracking,
	)

	o_config = resolve_player_config(
		{
			"depth": o_depth,
			"alpha_beta": o_alpha_beta,
			"move_ordering": o_move_ordering,
			"evaluator": o_evaluator,
			"canonicalization": o_canonicalization,
			"transposition_table": o_transposition_table,
			"dependency_tracking": o_dependency_tracking,
		},
		depth,
		alpha_beta,
		move_ordering,
		evaluator,
		canonicalization,
		transposition_table,
		dependency_tracking,
	)

	configs = {
		Player.X: x_config,
		Player.O: o_config,
	}

	player_types = {
		Player.X: x_player,
		Player.O: o_player,
	}

	for player, player_type in player_types.items():
		if player_type != "minimax":
			continue

		config = configs[player]

		if config["dependency_tracking"] and not config["transposition_table"]:
			raise ValueError(
				f"{player.name} dependency tracking requires "
				"the transposition table."
			)

		if config["transposition_table"] and not config["canonicalization"]:
			raise ValueError(
				f"{player.name} currently requires canonicalization "
				"to use the transposition table."
			)

	if games > 1 and "human" in player_types.values():
		raise ValueError(
			"--games greater than 1 requires both players "
			"to be computer-controlled."
		)

	game = TicTacToeGame()

	player_names = {
		Player.X: f"X ({x_player})",
		Player.O: f"O ({o_player})",
	}

	results = defaultdict(int)

	# Each player's statistics are collected as one list of samples
	# per statistic, with one sample per search.
	statistics_totals = {
		Player.X: defaultdict(list),
		Player.O: defaultdict(list),
	}

	# Create the AI instances and transposition tables only once.
	# Reuse them across all games.
	players, ais, transposition_tables = create_players(
		game,
		x_player,
		o_player,
		configs,
	)

	for game_number in range(1, games + 1):
		winner = play_game(
			game,
			players,
			ais,
			player_names=player_names,
			delay=delay,
			game_number=game_number,
			statistics_totals=statistics_totals,
			quiet=quiet,
		)

		results[winner] += 1

	# Print aggregate results after all games.
	if games > 1:
		print_aggregate_statistics(
			results,
			statistics_totals,
			transposition_tables,
		)


def validate_arguments(args, parser):
	if args["games"] < 1:
		parser.error("--games must be at least 1")

	if args["delay"] < 0:
		parser.error("--delay cannot be negative")

	player_types = {
		Player.X: args["x_player"],
		Player.O: args["o_player"],
	}

	configs = {
		Player.X: resolve_player_config(
			{
				"depth": args["x_depth"],
				"alpha_beta": args["x_alpha_beta"],
				"move_ordering": args["x_move_ordering"],
				"evaluator": args["x_evaluator"],
				"canonicalization": args["x_canonicalization"],
				"transposition_table": args["x_transposition_table"],
				"dependency_tracking": args["x_dependency_tracking"],
			},
			args["depth"],
			args["alpha_beta"],
			args["move_ordering"],
			args["evaluator"],
			args["canonicalization"],
			args["transposition_table"],
			args["dependency_tracking"],
		),
		Player.O: resolve_player_config(
			{
				"depth": args["o_depth"],
				"alpha_beta": args["o_alpha_beta"],
				"move_ordering": args["o_move_ordering"],
				"evaluator": args["o_evaluator"],
				"canonicalization": args["o_canonicalization"],
				"transposition_table": args["o_transposition_table"],
				"dependency_tracking": args["o_dependency_tracking"],
			},
			args["depth"],
			args["alpha_beta"],
			args["move_ordering"],
			args["evaluator"],
			args["canonicalization"],
			args["transposition_table"],
			args["dependency_tracking"],
		),
	}

	for player, player_type in player_types.items():
		if player_type != "minimax":
			continue

		config = configs[player]

		if config["dependency_tracking"] and not config["transposition_table"]:
			parser.error(
				f"{player.name} dependency tracking requires "
				"the transposition table."
			)

		if config["transposition_table"] and not config["canonicalization"]:
			parser.error(
				f"{player.name} currently requires canonicalization "
				"to use the transposition table."
			)

	if args["games"] > 1 and "human" in player_types.values():
		parser.error(
			"--games greater than 1 requires both players "
			"to be computer-controlled."
		)


def parse_arguments():
	parser = ArgumentParser(
		description="Play Tic-Tac-Toe against a human or AI opponent."
	)

	parser.add_argument(
		"--x-player",
		choices=("human", "minimax", "random"),
		default="human",
	)

	parser.add_argument(
		"--o-player",
		choices=("human", "minimax", "random"),
		default="human",
	)

	parser.add_argument("--games", type=int, default=1)
	parser.add_argument("--delay", type=float, default=0.5)

	parser.add_argument(
		"--quiet",
		action="store_true",
		default=False,
		help="Only print the final board for each game and the match summary.",
	)

	parser.add_argument("--depth", type=int, default=9)
	parser.add_argument("--x-depth", type=int, default=None)
	parser.add_argument("--o-depth", type=int, default=None)

	parser.add_argument(
		"--alpha-beta",
		action=argparse.BooleanOptionalAction,
		default=False,
	)
	parser.add_argument(
		"--x-alpha-beta",
		action=argparse.BooleanOptionalAction,
		default=None,
	)
	parser.add_argument(
		"--o-alpha-beta",
		action=argparse.BooleanOptionalAction,
		default=None,
	)

	parser.add_argument(
		"--move-ordering",
		choices=MOVE_ORDERING_CHOICES,
		default="none",
	)
	parser.add_argument(
		"--x-move-ordering",
		choices=MOVE_ORDERING_CHOICES,
		default=None,
	)
	parser.add_argument(
		"--o-move-ordering",
		choices=MOVE_ORDERING_CHOICES,
		default=None,
	)

	parser.add_argument(
		"--evaluator",
		action=argparse.BooleanOptionalAction,
		default=False,
	)
	parser.add_argument(
		"--x-evaluator",
		action=argparse.BooleanOptionalAction,
		default=None,
	)
	parser.add_argument(
		"--o-evaluator",
		action=argparse.BooleanOptionalAction,
		default=None,
	)

	parser.add_argument(
		"--canonicalization",
		action=argparse.BooleanOptionalAction,
		default=False,
	)
	parser.add_argument(
		"--x-canonicalization",
		action=argparse.BooleanOptionalAction,
		default=None,
	)
	parser.add_argument(
		"--o-canonicalization",
		action=argparse.BooleanOptionalAction,
		default=None,
	)

	parser.add_argument(
		"--transposition-table",
		action=argparse.BooleanOptionalAction,
		default=False,
	)
	parser.add_argument(
		"--x-transposition-table",
		action=argparse.BooleanOptionalAction,
		default=None,
	)
	parser.add_argument(
		"--o-transposition-table",
		action=argparse.BooleanOptionalAction,
		default=None,
	)

	parser.add_argument(
		"--dependency-tracking",
		action=argparse.BooleanOptionalAction,
		default=False,
	)
	parser.add_argument(
		"--x-dependency-tracking",
		action=argparse.BooleanOptionalAction,
		default=None,
	)
	parser.add_argument(
		"--o-dependency-tracking",
		action=argparse.BooleanOptionalAction,
		default=None,
	)

	args = vars(parser.parse_args())
	validate_arguments(args, parser)

	return args


if __name__ == "__main__":
	main(**parse_arguments())

