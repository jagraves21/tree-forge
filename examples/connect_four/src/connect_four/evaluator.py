from tree_forge import Evaluator

from .game import (
	BOARD_COLUMNS,
	BOARD_ROWS,
	CONNECT_LENGTH,
	ConnectFourGame,
	GameState,
	Player,
)


class ConnectFourEvaluator(Evaluator[GameState, Player]):
	"""Evaluate Connect Four positions from a player's perspective.

	A win is worth WIN_SCORE minus a penalty for every move played so
	far, so faster wins are preferred. A loss is its negation, so
	slower losses are preferred. Draws are worth zero. Nonterminal
	positions are scored with a line-counting heuristic.
	"""

	WIN_SCORE = 1_000_000.0
	MOVE_PENALTY = 1.0

	LINE_SCORES = {
		1: 1.0,
		2: 10.0,
		3: 100.0,
	}

	CENTER_SCORE = 3.0

	def evaluate_win(
		self,
		state: GameState,
		player: Player,
		ply: int,
		depth: int | None,
	) -> float:
		return self.WIN_SCORE - state.move_count * self.MOVE_PENALTY

	def evaluate_loss(
		self,
		state: GameState,
		player: Player,
		ply: int,
		depth: int | None,
	) -> float:
		return -self.WIN_SCORE + state.move_count * self.MOVE_PENALTY

	def evaluate_non_terminal(
		self,
		state: GameState,
		player: Player,
		ply: int,
		depth: int | None,
	) -> float:
		opponent = player.opponent
		score = 0.0

		# Evaluate control of the center column.
		center_column = BOARD_COLUMNS // 2

		for row in range(BOARD_ROWS):
			piece = state.board[row][center_column]

			if piece is player:
				score += self.CENTER_SCORE
			elif piece is opponent:
				score -= self.CENTER_SCORE

		# Evaluate every possible winning window.
		for row in range(BOARD_ROWS):
			for column in range(BOARD_COLUMNS):
				for row_delta, column_delta in (
					(0, 1),   # Horizontal
					(1, 0),   # Vertical
					(1, 1),   # Diagonal down-right
					(1, -1),  # Diagonal down-left
				):
					end_row = row + (CONNECT_LENGTH - 1) * row_delta
					end_column = column + (CONNECT_LENGTH - 1) * column_delta

					if not (
						0 <= end_row < BOARD_ROWS
						and 0 <= end_column < BOARD_COLUMNS
					):
						continue

					window = [
						state.board[
							row + offset * row_delta
						][
							column + offset * column_delta
						]
						for offset in range(CONNECT_LENGTH)
					]

					player_count = window.count(player)
					opponent_count = window.count(opponent)

					if opponent_count == 0 and player_count:
						score += self.LINE_SCORES[player_count]

					elif player_count == 0 and opponent_count:
						score -= self.LINE_SCORES[opponent_count]

		return score

