import random
import numpy as np
from collections import deque

class minesweeper_game:
  def __init__(self, grid_width, grid_height, num_mines):
    self.grid_width = grid_width
    self.grid_height = grid_height
    self.num_mines = num_mines
    self.mines_grid = self.init_mines_grid()
    self.flags_grid = np.zeros((grid_width, grid_height), dtype=bool)
    self.cleared_grid = np.zeros((grid_width, grid_height), dtype=bool)
    self.first_clear = True
    self.MAX_REGEN_ATTEMPTS = 500
    self.num_cleared_cells = 0
    self.num_flagged_cells = 0

  def __str__(self) -> str:
    grid_str = f""
    for y in range(self.grid_height):
      for x in range(self.grid_width):
        chr = "M" if self.mines_grid[x][y] == -1 else str(self.mines_grid[x][y])
        if not self.cleared_grid[x][y]: chr = '▯'
        if self.flags_grid[x][y]: chr = 'F'
        grid_str += chr
      if (y < self.grid_height-1): grid_str += '\n'
    return grid_str

  def get_neighbouring_cells(self, x, y):
    neighbours = deque(maxlen=8)
    for dy in range(-1 if y > 0 else 0, 2 if y < self.grid_height-1 else 1):
      for dx in range(-1 if x > 0 else 0, 2 if x < self.grid_width-1 else 1):
        if not (dy == 0 and dx == 0):
          neighbours.append((x+dx, y+dy))
    return neighbours
  
  def count_neighbouring_mines(self, mines_grid, x, y):
    num_neighbouring_mines = 0
    for dy in range(-1 if y > 0 else 0, 2 if y < self.grid_height-1 else 1):
      for dx in range(-1 if x > 0 else 0, 2 if x < self.grid_width-1 else 1):
        if (not (dy == 0 and dx == 0)):
          if mines_grid[x+dx][y+dy] == -1:
            num_neighbouring_mines += 1
    return num_neighbouring_mines

  def init_mines_grid(self):
    grid = np.zeros((self.grid_width, self.grid_height), dtype=np.int8)
    mines_placed = 0
    while mines_placed < self.num_mines:
      rand_x = random.randint(0, self.grid_width-1)
      rand_y = random.randint(0, self.grid_height-1)
      if grid[rand_x][rand_y] != -1:
        grid[rand_x][rand_y] = -1
        mines_placed += 1
    for y in range(self.grid_height):
      for x in range(self.grid_width):
        if grid[x][y] != -1:
          grid[x][y] = self.count_neighbouring_mines(grid, x, y)
    return grid
  
  def clear_cell(self, x, y) -> int: # returns 1 if action won game, -1 for loss, 0 otherwise
    if self.flags_grid[x][y]: return 0 # can't clear a flagged cell
    if self.cleared_grid[x][y]: return 0 # can't clear a cleared cell

    if self.first_clear: # if first click of the game, make sure you don't insta-lose or have to guess blindly
      for i in range(self.MAX_REGEN_ATTEMPTS):
        if self.mines_grid[x][y] != 0:
          self.mines_grid = self.init_mines_grid()
      if self.mines_grid[x][y] != 0:
        return False
      self.first_clear = False

    # game over if you try to clear a mine
    if self.mines_grid[x][y] == -1: return -1

    # check for empty adjacent cells and recursively clear them and their neighbours
    cleared_cells = np.zeros((self.grid_width, self.grid_height), dtype=bool)
    self.clear_neighbours(x, y, cleared_cells)
    self.num_cleared_cells = self.cleared_grid.sum()
    return int(self.check_for_win())
  
  def clear_neighbours(self, x, y, cleared_cells):
    self.cleared_grid[x][y] = True
    cleared_cells[x][y] = True
    if self.mines_grid[x][y] == 0:
      neighbours = self.get_neighbouring_cells(x, y)
      for n in neighbours:
        if (not cleared_cells[n[0]][n[1]]) and (self.mines_grid[n[0]][n[1]] != -1): self.clear_neighbours(n[0], n[1], cleared_cells)
        cleared_cells[n[0]][n[1]] = True
        self.cleared_grid[n] = True

  def flag_cell(self, x, y) -> int: # returns 1 if action won game
    if self.cleared_grid[x][y]: return 0 # can't flag a cleared cell
    is_flagged = self.flags_grid[x][y]
    self.num_flagged_cells += (-1 if is_flagged else 1)
    self.flags_grid[x][y] = not is_flagged
    return int(self.check_for_win())

  def check_for_win(self) -> bool:
    all_mines_cleared = True
    all_mines_flagged = self.flags_grid.sum() <= self.num_mines # can't place more flags than mines and win
    for y in range(self.grid_height):
      for x in range(self.grid_width):
        if self.mines_grid[x][y] != -1 and not self.cleared_grid[x][y]: all_mines_cleared = False
        if self.mines_grid[x][y] == -1 and not self.flags_grid[x][y]: all_mines_flagged = False
    return (all_mines_cleared or all_mines_flagged)