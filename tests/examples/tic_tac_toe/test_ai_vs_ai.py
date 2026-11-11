from tree_forge import Minimax, SearchLimits

from tic_tac_toe import Player, TicTacToeGame


class TestAIVsAI:
	def test_two_minimax_instances_play_complete_game(self):
		game = TicTacToeGame()
		state = game.initial_state()

		limits = SearchLimits(max_depth=9)

		players = {
			Player.X: Minimax(game, limits=limits),
			Player.O: Minimax(game, limits=limits),
		}

		move_count = 0

		while not game.is_terminal(state):
			player = game.current_player(state)
			search = players[player]

			result = search.search(state)

			assert result.best_move is not None
			assert result.best_move in game.legal_moves(state)

			game.make_move(state, result.best_move)
			move_count += 1

		assert game.is_terminal(state)
		assert move_count <= 9

		# Perfect play by both AIs should result in a draw.
		assert game.winner(state) is None

