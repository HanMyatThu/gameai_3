import pygame
from settings import DEFAULT_THEME

class ThemeManager:
  def __init__(self):
    self.themes = {
      "day": {
        "background": pygame.image.load("assets/layout/day/bg.png").convert(),
        "pipe": pygame.image.load("assets/layout/day/pipe.png").convert_alpha(),
        "ground": pygame.image.load("assets/layout/day/ground.png").convert(),
      },
      "night": {
        "background": pygame.image.load("assets/layout/night/bg_night.png").convert(),
        "pipe": pygame.image.load("assets/layout/night/pipe_night.png").convert_alpha(),
        "ground": pygame.image.load("assets/layout/night/ground_night.png").convert(),
      }
    }
    self.current_theme = DEFAULT_THEME

  def set_theme(self, theme):
    self.current_theme = theme
  
  def get_current_theme(self):
    return self.current_theme

  def get(self, asset_name):
    return self.themes[self.current_theme][asset_name]
