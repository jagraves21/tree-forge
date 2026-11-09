from tree_forge import Minimax, SearchLimits


def test_minimax_finds_optimal_move_without_alpha_beta(game):
	state = game.initial_state()

	search = Minimax(
		game,
		limits=SearchLimits(max_depth=2),
		alpha_beta=False,
	)

	result = search.search(state)

	assert result.best_move.destination == "a"
	assert result.score == 3.0
	assert result.statistics.cutoffs == 0


def test_minimax_finds_optimal_move_with_alpha_beta(game):
	state = game.initial_state()

	search = Minimax(
		game,
		limits=SearchLimits(max_depth=2),
		alpha_beta=True,
	)

	result = search.search(state)

	assert result.best_move.destination == "a"
	assert result.score == 3.0
	assert result.statistics.cutoffs > 0


def test_alpha_beta_visits_fewer_leaves(game):
	state = game.initial_state()

	plain_result = Minimax(
		game,
		limits=SearchLimits(max_depth=2),
		alpha_beta=False,
	).search(state)

	pruned_result = Minimax(
		game,
		limits=SearchLimits(max_depth=2),
		alpha_beta=True,
	).search(state)

	assert pruned_result.best_move == plain_result.best_move
	assert pruned_result.score == plain_result.score
	assert (
		pruned_result.statistics.leaf_nodes
		< plain_result.statistics.leaf_nodes
	)


def test_minimax_restores_original_state(game):
	state = game.initial_state()

	original_node = state.node
	original_player = state.current_player

	search = Minimax(
		game,
		limits=SearchLimits(max_depth=2),
	)

	search.search(state)

	assert state.node == original_node
	assert state.current_player == original_player


def test_two_minimax_instances_play_complete_game(game):
	minimax_p1 = Minimax(
		game,
		limits=SearchLimits(max_depth=2),
	)

	minimax_p2 = Minimax(
		game,
		limits=SearchLimits(max_depth=2),
	)

	state = game.initial_state()
	move_count = 0

	while not game.is_terminal(state):
		player = game.current_player(state)

		search = minimax_p1 if player == 1 else minimax_p2
		result = search.search(state)

		assert result.best_move is not None
		assert result.best_move in game.legal_moves(state)

		game.make_move(state, result.best_move)
		move_count += 1

	assert game.is_terminal(state)
	assert move_count == 2
	assert state.node == "a1"

