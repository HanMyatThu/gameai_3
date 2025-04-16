import pygame

class Pipe(pygame.sprite.Sprite):
    def __init__(self, pos, width, height, flip, is_night, assets):
        super().__init__()
        self.width = width
        theme = "night" if is_night else "day"
        assets.set_theme(theme)
        self.image = assets.get("pipe")
        self.image = pygame.transform.scale(self.image, (width, height))
        if flip:
            flipped_image = pygame.transform.flip(self.image, False, True)
            self.image = flipped_image
        self.rect = self.image.get_rect(topleft = pos)

    # update object position due to world scroll
    def update(self, x_shift):
        self.rect.x += x_shift
        # removes the pipe in the game screen once it is not shown in the screen anymore
        if self.rect.right < (-self.width):
            self.kill()