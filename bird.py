import pygame
from settings import import_sprite, DEFAULT_THEME

class Bird(pygame.sprite.Sprite):
    def __init__(self, pos, size):
        super().__init__()
        # bird basic info
        self.frame_index = 0
        self.animation_delay = 3
        self.jump_move = -8
        self.size = size
        # bird animation
        self.game_mode = DEFAULT_THEME
        self.bird_img = import_sprite(f'assets/birds/{self.game_mode}')
        self.image = self.bird_img[self.frame_index]
        self.image = pygame.transform.scale(self.image, (size, size))
        self.rect = self.image.get_rect(topleft = pos)
        self.mask = pygame.mask.from_surface(self.image)
        # bird status
        self.direction = pygame.math.Vector2(0, 0)
        self.score = 0
        self.level = 1

    # for bird's flying animation
    def _animate(self):
        sprites = self.bird_img
        sprite_index = (self.frame_index // self.animation_delay) % len(sprites)
        self.image = sprites[sprite_index]
        self.frame_index += 1
        self.rect = self.image.get_rect(topleft=(self.rect.x, self.rect.y))
        self.mask = pygame.mask.from_surface(self.image)
        if self.frame_index // self.animation_delay > len(sprites):
            self.frame_index = 0

    def _jump(self):
        self.direction.y = self.jump_move

    def update(self, is_jump, game_mode):
        if (self.game_mode != game_mode):
          self.game_mode = game_mode
          self.update_bird()
        if is_jump:
            self._jump()
        self._animate()

    def update_bird(self):
      print("Bird image updated")
      print(self.game_mode)
      self.bird_img = import_sprite(f'assets/birds/{self.game_mode}')
      print(self.bird_img, 'bird image')
      self.frame_index = self.frame_index % len(self.bird_img)  # Avoid out-of-range error
      self.image = self.bird_img[self.frame_index]
      self.image = pygame.transform.scale(self.image, (self.size, self.size))
