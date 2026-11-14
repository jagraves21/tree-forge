from tree_forge import Evaluator

from .game import GameState, Player


class TicTacToeEvaluator(Evaluator[GameState, Player]):
	"""Evaluate Tic-Tac-Toe positions from a player's perspective."""

	WINNING_LINES = (
		# Rows
		((0, 0), (0, 1), (0, 2)),
		((1, 0), (1, 1), (1, 2)),
		((2, 0), (2, 1), (2, 2)),
		# Columns
		((0, 0), (1, 0), (2, 0)),
		((0, 1), (1, 1), (2, 1)),
		((0, 2), (1, 2), (2, 2)),
		# Diagonals
		((0, 0), (1, 1), (2, 2)),
		((0, 2), (1, 1), (2, 0)),
	)

	LINE_SCORES = {
		1: 1.0,
		2: 10.0,
	}

	CENTER_SCORE = 3.0
	CORNER_SCORE = 1.0

	def evaluate(self, state: GameState, player: Player) -> float:
		opponent = player.opponent
		score = 0.0

		# Evaluate open winning lines.
		for line in self.WINNING_LINES:
			cells = [state.board[row][column] for row, column in line]

			player_count = cells.count(player)
			opponent_count = cells.count(opponent)

			if opponent_count == 0 and player_count:
				score += self.LINE_SCORES[player_count]

			elif player_count == 0 and opponent_count:
				score -= self.LINE_SCORES[opponent_count]

		# Reward occupying the center.
		if state.board[1][1] is player:
			score += self.CENTER_SCORE
		elif state.board[1][1] is opponent:
			score -= self.CENTER_SCORE

		# Reward occupying corners.
		for row, column in (
			(0, 0),
			(0, 2),
			(2, 0),
			(2, 2),
		):
			if state.board[row][column] is player:
				score += self.CORNER_SCORE
			elif state.board[row][column] is opponent:
				score -= self.CORNER_SCORE

		return score

