import pygame
import sys
from settings import WIDTH, HEIGHT, GROUND_HEIGHT
from world import World
from theme import ThemeManager
from sound import play as soundPlay
from game import GameIndicator

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT + GROUND_HEIGHT))
pygame.display.set_caption("Super Flappy Bird")
theme = ThemeManager()

class Main:
    def __init__(self, screen):
      self.screen = screen
      self.ground_scroll = 0
      self.scroll_speed = -6
      self.FPS = pygame.time.Clock()
      self.stop_ground_scroll = False

    def main(self):
      world = World(screen, theme, isMulti=True)

      while True:
        self.stop_ground_scroll = world.game_over

        # --- Draw background ---
        bg_img = pygame.transform.scale(theme.get('background'), (WIDTH, HEIGHT))
        self.screen.blit(bg_img, (0, 0))

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if not world.playing and not world.game_over:
                  world.playing = True           
                if event.key == pygame.K_SPACE:
                  soundPlay("jump")
                  world.update("jump")
                if event.key == pygame.K_r:
                  world.update("restart")

        # --- Update game world ---
        world.update()
        world.draw()

        if world.game_over:
          game = GameIndicator(screen, theme)
          game.instructions()

        # --- Draw ground ---
        ground_img = theme.get('ground')
        self.screen.blit(ground_img, (self.ground_scroll, HEIGHT))

        if not self.stop_ground_scroll:
            self.ground_scroll += self.scroll_speed
            if abs(self.ground_scroll) > 35:
                self.ground_scroll = 0

        pygame.display.update()
        self.FPS.tick(60)

if __name__ == "__main__":
    play = Main(screen)
    soundPlay("background")
    play.main()
