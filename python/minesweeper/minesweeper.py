import random
from collections import deque

class minesweeper_game:
  def __init__(self, grid_width, grid_height, num_mines):
    self.grid_width = grid_width
    self.grid_height = grid_height
    self.num_mines = num_mines
    self.mines_grid = self.init_mines_grid()
    self.flags_grid = [False]*grid_width*grid_height
    self.cleared_grid = [False]*grid_width*grid_height
    self.first_clear = True
    self.MAX_REGEN_ATTEMPTS = 500
    self.num_cleared_cells = 0
    self.num_flagged_cells = 0

  def __str__(self) -> str:
    grid_str = f""
    for y in range(self.grid_height):
      for x in range(self.grid_width):
        idx = self.to_idx(x, y)
        chr = "M" if self.mines_grid[idx] == -1 else str(self.mines_grid[idx])
        if not self.cleared_grid[idx]: chr = '▯'
        if self.flags_grid[idx]: chr = 'F'
        grid_str += chr
      if (y < self.grid_height-1): grid_str += '\n'
    return grid_str

  def get_neighbouring_cells(self, x, y):
    neighbours = deque(maxlen=9)
    for dy in range(-1 if y > 0 else 0, 2 if y < self.grid_height-1 else 1):
      for dx in range(-1 if x > 0 else 0, 2 if x < self.grid_width-1 else 1):
        if not (dy == 0 and dx == 0):
          neighbours.append(self.to_idx(x+dx, y+dy))
    return neighbours


  def init_mines_grid(self):
    grid = [0]*self.grid_width*self.grid_height
    mines_placed = 0
    while mines_placed < self.num_mines:
      random_idx = random.randint(0, len(grid)-1)
      if grid[random_idx] != -1:
        grid[random_idx] = -1
        mines_placed += 1
    for y in range(self.grid_height):
      for x in range(self.grid_width):
        idx = self.to_idx(x,y)
        if grid[idx] != -1:
          neighbours = self.get_neighbouring_cells(x, y)
          surrounding_mines = 0
          for n in neighbours:
            if grid[n] == -1:
              surrounding_mines += 1
          grid[idx] = surrounding_mines
    return grid


  def to_idx(self, x, y):
    return y*self.grid_width+x
  
  def to_coords(self, idx):
    y = idx // self.grid_width
    x = idx-(y*self.grid_width)
    return (x, y)
  
  def clear_cell(self, x, y) -> int: # returns 1 if action won game, -1 for loss, 0 otherwise
    idx = self.to_idx(x, y)
    if self.flags_grid[idx]: return 0 # can't clear a flagged cell
    if self.cleared_grid[idx]: return 0 # can't clear a cleared cell

    if self.first_clear: # if first click of the game, make sure you don't insta-lose or have to guess blindly
      for i in range(self.MAX_REGEN_ATTEMPTS):
        if self.mines_grid[idx] != 0:
          self.mines_grid = self.init_mines_grid()
      if self.mines_grid[idx] != 0:
        return False
      self.first_clear = False

    # game over if you try to clear a mine
    if self.mines_grid[idx] == -1: return -1

    # check for empty adjacent cells and recursively clear them and their neighbours
    cleared_cells = set()
    self.clear_neighbours(idx, cleared_cells)
    self.num_cleared_cells = self.cleared_grid.count(True)
    return int(self.check_for_win())
  
  def clear_neighbours(self, idx, cleared_cells):
    self.cleared_grid[idx] = True
    cleared_cells.add(idx)
    if self.mines_grid[idx] == 0:
      neighbours = self.get_neighbouring_cells(*self.to_coords(idx))
      for n in neighbours:
        if (n not in cleared_cells) and (self.mines_grid[n] != -1): self.clear_neighbours(n, cleared_cells)
        cleared_cells.add(n)
        self.cleared_grid[n] = True

  
  def flag_cell(self, x, y) -> int: # returns 1 if action won game
    idx = self.to_idx(x, y)
    if self.cleared_grid[idx]: return 0 # can't flag a cleared cell
    is_flagged = self.flags_grid[idx]
    self.num_flagged_cells += (-1 if is_flagged else 1)
    self.flags_grid[idx] = not is_flagged
    return int(self.check_for_win())

  def check_for_win(self) -> bool:
    all_mines_cleared = True
    all_mines_flagged = self.flags_grid.count(True) <= self.num_mines # can't place more flags than mines and win
    for i in range(len(self.mines_grid)):
      if self.mines_grid[i] != -1 and not self.cleared_grid[i]: all_mines_cleared = False
      if self.mines_grid[i] == -1 and not self.flags_grid[i]: all_mines_flagged = False
    return (all_mines_cleared or all_mines_flagged)