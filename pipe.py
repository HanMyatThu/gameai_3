import pygame

class Pipe(pygame.sprite.Sprite):
    def __init__(self, pos, width, height, flip, theme, assets):
        super().__init__()
        self.width = width
        # Need to access the theme assets correctly - passing ThemeManager might be better
        # Assuming assets is the ThemeManager instance for now
        # assets.set_theme(theme) # Theme set in World, Pipe just uses current
        self.image = assets.get("pipe")
        self.image = pygame.transform.scale(self.image, (width, height))
        self.is_flipped = flip # Store flip status
        if self.is_flipped:
            flipped_image = pygame.transform.flip(self.image, False, True)
            self.image = flipped_image
        self.rect = self.image.get_rect(topleft = pos)
        self.mask = pygame.mask.from_surface(self.image) # Add mask for collision

    def get_is_flipped(self):
        return self.is_flipped

    # update object position due to world scroll
    def update(self, x_shift):
        self.rect.x += x_shift
        # removes the pipe in the game screen once it is not shown in the screen anymore
        if self.rect.right < 0: # Check if fully off screen left
            self.kill()

    def update_theme(self, assets):
      self.image = assets.get("pipe")  
      self.image = pygame.transform.scale(self.image, (self.width, self.rect.height))
      
      if self.is_flipped:
          self.image = pygame.transform.flip(self.image, False, True)

      self.mask = pygame.mask.from_surface(self.image)