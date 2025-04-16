# world.py
import pygame
from pipe import Pipe
from bird import Bird
from game import GameIndicator
from settings import WIDTH, HEIGHT, PIPE_SIZE, PIPE_GAP, DEFAULT_PIPE_PAIR, PIPE_PATTERNS
import random
from sound import play, stop

class World:
  def __init__(self, screen, theme):
    self.screen = screen
    self.theme = theme
    self.world_shift = 0
    self.current_x = 0
    self.gravity = 0.5

    # pattern for pipes
    self.current_pipe = 'zigzag'
    self.pattern_index = 0
    self.pattern_switch_count = 5 # change pipe for every 5 pipes

    self.current_pipe = None
    self.pipes = pygame.sprite.Group()
    self.player = pygame.sprite.GroupSingle()
    self.pipe_count = 0
    self.is_night = False
    self.theme_switch_count = 15  # change theme every 15 pipes
    self._generate_world()
    self.playing = False
    self.game_over = False
    self.game_over_sound = False
    self.passed = True
    self.game = GameIndicator(screen, theme)

  def _add_pipe(self):
    # Switch pattern every 5 pipes
    if self.pipe_count % self.pattern_switch_count == 0:
        self.current_pattern = random.choice(list(PIPE_PATTERNS.keys()))
        self.pattern_index = 0

    current_pattern_info = PIPE_PATTERNS[self.current_pattern]
    pipe_pairs = current_pattern_info.get("pairs")
    gap_multiplier = current_pattern_info.get("gap_multiplier", 1.0)
 
    # Get next pipe pair randomly
    if pipe_pairs:
        pipe_pair = pipe_pairs[self.pattern_index % len(pipe_pairs)]
        self.pattern_index += 1
    else:
        pipe_pair = random.choice(DEFAULT_PIPE_PAIR)

    pipe_gap = int(PIPE_GAP * gap_multiplier)

    print(f"[{self.current_pattern.upper()}] Pipe Pair: {pipe_pair}, Gap: {pipe_gap}")

    # switch theme after 15 pipes
    if (self.pipe_count > 0) and (self.pipe_count % self.theme_switch_count == 0):
        self.is_night = not self.is_night
        if self.is_night:
            play("night")
            self.theme.set_theme("night")
        else:
            play("day")
            self.theme.set_theme("night""day")

    top_pipe_height = pipe_pair[0] * PIPE_SIZE
    bottom_pipe_height = pipe_pair[1] * PIPE_SIZE

    pipe_top = Pipe((WIDTH, 0 - (bottom_pipe_height + pipe_gap)), PIPE_SIZE, HEIGHT, True, self.is_night, self.theme)
    pipe_bottom = Pipe((WIDTH, top_pipe_height + pipe_gap), PIPE_SIZE, HEIGHT, False, self.is_night, self.theme)

    self.pipes.add(pipe_top)
    self.pipes.add(pipe_bottom)
    self.pipe_count += 1
    self.current_pipe = pipe_top


  def _generate_world(self):
    self._add_pipe()
    bird = Bird((WIDTH // 2 - PIPE_SIZE, HEIGHT // 2 - PIPE_SIZE), 45)
    self.player.add(bird)

  def scrollX(self):
    self.world_shift = -6 if self.playing else 0

  def apply(self, player):
    if self.playing or self.game_over:
      player.direction.y += self.gravity
      player.rect.y += player.direction.y

  def handle_collision(self):
    bird = self.player.sprite
    if pygame.sprite.groupcollide(self.player, self.pipes, False, False) or bird.rect.bottom >= HEIGHT or bird.rect.top <= 0:
      self.playing = False
      self.game_over = True
      if not self.game_over_sound:
        play("hit") 
        stop("background")
        stop("day")
        stop("night")
        self.game_over_sound = True
    else:
      if bird.rect.x >= self.current_pipe.rect.centerx:
        play("score")
        bird.score += 1
        self.passed = True

  def update(self, player_event=None):
    if self.current_pipe.rect.centerx <= (WIDTH // 2) - PIPE_SIZE:
        self._add_pipe()

    self.pipes.update(self.world_shift)
    self.pipes.draw(self.screen)
    self.apply(self.player.sprite)
    self.scrollX()
    self.handle_collision()

    if player_event == "jump" and not self.game_over:
      player_event = True
    elif player_event == "restart":
      self.game_over = False
      self.pipes.empty()
      self.player.empty()
      self.player.score = 0
      self.game_over_sound = False
      play("background")
      stop("hit")
      self._generate_world()
    else:
      player_event = False

    if not self.playing:
      self.game.instructions()

    self.player.update(player_event)
    self.player.draw(self.screen)
    self.game.show_score(self.player.sprite.score)
