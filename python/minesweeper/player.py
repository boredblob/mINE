import minesweeper
import numpy as np
from timeit import timeit

MAX_SOLVER_ITERATIONS = 100

# easy mode accuracy: 88.5%, expert mode: 34%
def solve_game(game: minesweeper.minesweeper_game):
  game_status = game.clear_cell(game.grid_width // 2, game.grid_height // 2)
  for _ in range(MAX_SOLVER_ITERATIONS):
    if (game_status == 0) :
      game_status = solver_iteration(game)
    else:
      break
  return game_status

def solver_iteration(game: minesweeper.minesweeper_game):
  num_clears_before = game.cleared_grid.count(True)
  num_flags_before = game.flags_grid.count(True)

  game_status = clear_all_safe_cells(game)
  if game_status != 0: return game_status
  game_status = flag_all_mines(game)
  if game_status != 0: return game_status

  num_clears_diff = game.cleared_grid.count(True) - num_clears_before
  num_flags_diff = game.flags_grid.count(True) - num_flags_before
  if (num_clears_diff == 0 and num_flags_diff == 0): game_status = make_guess_flag(game)
  return game_status

def make_guess_flag(game: minesweeper.minesweeper_game):
  mine_likelihood_grid = [0]*game.grid_width*game.grid_width

  for y in range(game.grid_height):
    for x in range(game.grid_width):
      idx = game.to_idx(x, y)
      if game.cleared_grid[idx] and (game.mines_grid[idx] > 0):
        n = game.mines_grid[idx]
        neighbours = game.get_neighbouring_cells(x, y)
        uncleared_neighbours = [x for x in neighbours if not game.cleared_grid[x]]
        num_flagged_neighbours = len([x for x in neighbours if game.flags_grid[x]])
        if (len(uncleared_neighbours) > num_flagged_neighbours):
          potential_mines = [x for x in uncleared_neighbours if not game.flags_grid[x]]
          for potential_mine in potential_mines:
            mine_likelihood_grid[potential_mine] += n/len(potential_mines)

  most_likely_mine = np.argmax(mine_likelihood_grid)
  game_status = game.flag_cell(*game.to_coords(most_likely_mine))
  return game_status

def clear_all_safe_cells(game: minesweeper.minesweeper_game):
  for y in range(game.grid_height):
    for x in range(game.grid_width):
      idx = game.to_idx(x, y)
      if game.cleared_grid[idx] and (game.mines_grid[idx] > 0):
        n = game.mines_grid[idx]
        neighbours = game.get_neighbouring_cells(x, y)
        uncleared_neighbours = [x for x in neighbours if not game.cleared_grid[x]]
        num_flagged_neighbours = len([x for x in neighbours if game.flags_grid[x]])
        if (n == num_flagged_neighbours):
          for neighbour in [x for x in uncleared_neighbours if not game.flags_grid[x]]:
            game_status = game.clear_cell(*game.to_coords(neighbour))
            if (game_status != 0): return game_status
  return 0

def flag_all_mines(game: minesweeper.minesweeper_game):
  for y in range(game.grid_height):
    for x in range(game.grid_width):
      idx = game.to_idx(x, y)
      if game.cleared_grid[idx] and (game.mines_grid[idx] > 0):
        n = game.mines_grid[idx]
        neighbours = game.get_neighbouring_cells(x, y)
        uncleared_neighbours = [x for x in neighbours if not game.cleared_grid[x]]
        if (n == len(uncleared_neighbours)):
          for neighbour in [x for x in uncleared_neighbours if not game.flags_grid[x]]:
            game_status = game.flag_cell(*game.to_coords(neighbour))
            if (game_status != 0): return game_status
  return 0


def test_solver(solver_func, num_games=10):
  wins = 0
  for _ in range(num_games):
    game_instance = minesweeper.minesweeper_game(9, 9, 10)
    result = solver_func(game_instance)
    if result == 1: wins += 1

  print(str(round((wins / num_games)*100, 2)) + "%")
  return wins / num_games

print(timeit("test_solver(solve_game, 1000)", globals=locals(), number=1))