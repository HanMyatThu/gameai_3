import pygame
from os import walk

# ========== Game Screen ==========
WIDTH, HEIGHT = 900, 650
GROUND_HEIGHT = 60
FPS = 120

# ========== Theme ==========
DEFAULT_THEME = "day"
THEMES = ["day", "night", "hell", "space"]

# ========== Pipes ==========
PIPE_ROWS = HEIGHT // 10
PIPE_SIZE = HEIGHT // 10
PIPE_GAP = (PIPE_SIZE * 2) + (PIPE_SIZE // 2)

# Pipe Patterns for randomized PCG
DEFAULT_PIPE_PAIR = [(2, 6), (3, 5), (4, 4), (5, 3)]
PIPE_PATTERNS = {
  "zigzag": {
    "pairs": [(2, 6), (3, 5), (4, 4), (5, 3)],
    "gap_multiplier": 1.0
  },
  "ascending": {
    "pairs": [(2, 6), (2, 6), (3, 5), (4, 4)],
    "gap_multiplier": 1.0
  },
  "descending": {
    "pairs": [(6, 2), (5, 3), (4, 4), (3, 5)],
    "gap_multiplier": 0.9
  },
  "random": {
    "pairs": None,
    "gap_multiplier": 1.0
  }
}

def import_sprite(path):
  print('path', path)
  surface_list = []
  for _, __, image_files in walk(path):
    for image in sorted(image_files): 
      full_path = f"{path}/{image}"
      img_surface = pygame.image.load(full_path).convert_alpha()
      surface_list.append(img_surface)
  return surface_list