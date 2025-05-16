import pygame
from settings import WIDTH

pygame.font.init()

class GameIndicator:
  def __init__(self, screen, theme):
    self.screen = screen
    self.theme = theme
    self.font = pygame.font.SysFont('jokerman', 50) 
    self.info_font = pygame.font.SysFont('magneto', 28) 
    self.color =  pygame.Color("white")
    self.info_color_night = pygame.Color("white")
    self.info_color = pygame.Color("black")

  def show_levels(self, level):
    game_level = self.font.render(f"STAGE: {str(level)}", True, self.color)
    self.screen.blit(game_level, (390, 50))

  def show_score(self, int_score):
    bird_score = str(int_score)
    score = self.font.render(bird_score, True, self.color)
    self.screen.blit(score, (WIDTH // 2, 90))

  def instructions(self):
    info1 = "Press SPACE button to Jump!"
    info2 = "Press \"R\"to restart."
    currenttheme = self.theme.get_current_theme()
    if currenttheme == "night":
        self.info_color = self.info_color_night
    else:
        self.info_color = pygame.Color("black")
    ins1 = self.info_font.render(info1, True, self.info_color)
    ins2 = self.info_font.render(info2, True, self.info_color)
    self.screen.blit(ins1, (95, 400))
    self.screen.blit(ins2, (70, 450))