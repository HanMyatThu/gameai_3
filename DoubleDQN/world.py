# import pygame
# from pipe import Pipe
# from bird import Bird
# from game import GameIndicator
# from settings import WIDTH, HEIGHT, PIPE_SIZE, PIPE_GAP, DEFAULT_PIPE_PAIR, PIPE_PATTERNS, GROUND_HEIGHT
# import random
# from sound import play, stop
# import numpy as np

# class World:
#   def __init__(self, screen, theme):
#     self.screen = screen
#     self.theme = theme
#     self.world_shift = 0
#     self.current_x = 0
#     self.gravity = 0.5

#     # pattern for pipes
#     self.current_pattern_name = 'zigzag'
#     self.pattern_index = 0
#     self.pattern_switch_count = 5

#     self.upcoming_pipes = pygame.sprite.Group()
#     self.player = pygame.sprite.GroupSingle()
#     self.pipe_count = 0
#     self.is_night = False
#     self.theme_switch_count = 15
#     self._generate_world()

#     self.playing = False
#     self.game_over = False
#     self.game_over_sound = False
#     self.passed = True
#     self.game = GameIndicator(screen, theme)

#     # RL specific attributes
#     self.current_reward = 0.0
#     self.last_score = 0

#   def _find_next_pipe(self):
#       """Finds the next upcoming pipe pair relative to the bird."""
#       bird = self.player.sprite
#       closest_pipe_bottom = None
#       min_dist = float('inf')

#       # Iterate through bottom pipes only to find the pair
#       for pipe in self.upcoming_pipes:
#           if not pipe.get_is_flipped():
#             dist = pipe.rect.left - bird.rect.right
#             if dist >= 0 and dist < min_dist:
#                 min_dist = dist
#                 closest_pipe_bottom = pipe

#       if closest_pipe_bottom:
#           for pipe in self.upcoming_pipes:
#               if pipe.get_is_flipped() and pipe.rect.left == closest_pipe_bottom.rect.left:
#                   return pipe, closest_pipe_bottom
#       return None, None


#   def get_state(self):
#       """Returns the current state representation for the RL agent."""
#       bird = self.player.sprite
#       if not bird:
#           return np.zeros(4, dtype=np.float32)

#       top_pipe, bottom_pipe = self._find_next_pipe()

#       if top_pipe and bottom_pipe:
#           # State calculation
#           bird_y_norm = bird.rect.centery / HEIGHT
#           bird_vel = bird.direction.y / 10.0
#           h_dist = (bottom_pipe.rect.left - bird.rect.right)
#           h_dist_norm = h_dist / WIDTH

#           gap_center_y = top_pipe.rect.bottom + (bottom_pipe.rect.top - top_pipe.rect.bottom) / 2
#           v_dist = (gap_center_y - bird.rect.centery)
#           v_dist_norm = v_dist / HEIGHT

#           h_dist_norm = np.clip(h_dist_norm, -1, 1)
#           v_dist_norm = np.clip(v_dist_norm, -1, 1)
#           bird_y_norm = np.clip(bird_y_norm, 0, 1)
#           bird_vel = np.clip(bird_vel, -1, 1)


#           state = np.array([bird_y_norm, bird_vel, h_dist_norm, v_dist_norm], dtype=np.float32)
#           #print(f"State: bird_y={state[0]:.2f}, bird_vel={state[1]:.2f}, h_dist={state[2]:.2f}, v_dist={state[3]:.2f}")
#       else:
#           bird_y_norm = bird.rect.centery / HEIGHT
#           bird_vel = bird.direction.y / 10.0
#           state = np.array([bird_y_norm, bird_vel, 0.5, 0.0], dtype=np.float32)

#       return state


#   def _add_pipe(self):
#     if self.pipe_count % self.pattern_switch_count == 0:
#         self.current_pattern_name = random.choice(list(PIPE_PATTERNS.keys()))
#         self.pattern_index = 0
#         print(f"Switched to pattern: {self.current_pattern_name.upper()}")

#     current_pattern_info = PIPE_PATTERNS[self.current_pattern_name]
#     pipe_pairs = current_pattern_info.get("pairs")
#     gap_multiplier = current_pattern_info.get("gap_multiplier", 1.0)

#     if pipe_pairs:
#         pipe_pair = pipe_pairs[self.pattern_index % len(pipe_pairs)]
#         self.pattern_index += 1
#     else:
#         pipe_pair = random.choice(DEFAULT_PIPE_PAIR)

#     pipe_gap = int(PIPE_GAP * gap_multiplier)

#     if (self.pipe_count > 0) and (self.pipe_count % self.theme_switch_count == 0):
#         self.is_night = not self.is_night
#         new_theme = "night" if self.is_night else "day"
#         if self.theme.get_current_theme() != new_theme:
#             play(new_theme)
#             self.theme.set_theme(new_theme)
#             print(f"Switched theme to {new_theme.upper()}")

#     top_pipe_height = pipe_pair[0] * PIPE_SIZE
#     bottom_pipe_y = top_pipe_height + pipe_gap
#     bottom_pipe_height_needed = HEIGHT - bottom_pipe_y

#     pipe_top = Pipe((WIDTH, top_pipe_height - HEIGHT), PIPE_SIZE, HEIGHT, True, self.is_night, self.theme)

#     pipe_bottom = Pipe((WIDTH, bottom_pipe_y), PIPE_SIZE, HEIGHT, False, self.is_night, self.theme)

#     self.upcoming_pipes.add(pipe_top)
#     self.upcoming_pipes.add(pipe_bottom)
#     self.pipe_count += 1


#   def _generate_world(self):
#     # Reset everything before starting new state
#     self.upcoming_pipes.empty()
#     self.player.empty()

#     # Reset world state
#     self.world_shift = 0
#     self.current_x = 0
#     self.pipe_count = 0
#     self.pattern_index = 0
#     self.last_score = 0
#     self.is_night = False
#     self.theme.set_theme("day")
#     stop("night")
#     stop("day")
#     stop("hit")
#     stop("background")

#     self._add_pipe()
#     # Find the first pipe added and shift it slightly right so it's not immediate collision
#     first_pipe_x = WIDTH + 150
#     for pipe in self.upcoming_pipes:
#         pipe.rect.x = first_pipe_x

#     # Add player
#     bird_start_pos = (WIDTH // 4, HEIGHT // 2 - 50)
#     bird = Bird(bird_start_pos, 45)
#     self.player.add(bird)

#     # Reset game state flags
#     self.playing = True
#     self.game_over = False
#     self.game_over_sound = False
#     self.current_reward = 0.0


#   def scrollX(self):
#     self.world_shift = -6


#   def apply_physics(self):
#     """Applies gravity to the bird."""
#     if self.player.sprite:
#       if not self.game_over:
#           self.player.sprite.direction.y += self.gravity

#       # Update position based on velocity
#       self.player.sprite.rect.y += self.player.sprite.direction.y


#   def handle_collision(self):
#       """Checks for collisions and sets game over state."""
#       if self.game_over:
#           return True

#       bird = self.player.sprite
#       if not bird:
#           return True

#       # Pipe collision
#       collided_pipes = pygame.sprite.spritecollide(bird, self.upcoming_pipes, False, pygame.sprite.collide_mask)

#       # Ground/Ceiling collision
#       hit_ground = bird.rect.bottom >= HEIGHT
#       hit_ceiling = bird.rect.top <= 0

#       if collided_pipes or hit_ground or hit_ceiling:
#           if not self.game_over:
#               self.playing = False
#               self.game_over = True
#               if not self.game_over_sound:
#                   #play("hit")
#                   current_theme = self.theme.get_current_theme()
#                   stop(current_theme)
#                   stop("background")
#                   self.game_over_sound = True
#               # Large negative reward for dying
#               self.current_reward = -10.0
#           return True
#       else:
#           for pipe in self.upcoming_pipes:
#               if  bird.rect.x >= pipe.rect.centerx:
#                   #play("score")
#                   bird.score += 1
#           return False


#   def update_score_and_reward(self):
#       """Updates the score and calculates intermediate rewards."""
#       if self.game_over:
#           return

#       bird = self.player.sprite
#       if not bird:
#           return

#       # Reward for surviving
#       self.current_reward = 0.1

#       passed_pipe = False
#       _, next_bottom_pipe = self._find_next_pipe() # Find the pipe we are aiming for


#       potential_score = 0
#       for pipe in self.upcoming_pipes:
#           if not pipe.get_is_flipped() and bird.rect.centerx > pipe.rect.centerx:
#                potential_score += 1

#       current_score = potential_score // 2

#       if current_score > self.last_score:
#           play("score")
#           bird.score += (current_score - self.last_score)
#           self.current_reward += 5.0 # Reward for passing pipe
#           self.last_score = current_score
#           print(f"Passed Pipe! Score: {bird.score}, Reward: {self.current_reward}")

#   def step(self, action):
#       self.current_reward = 0.0

#       if self.game_over:
#           bird = self.player.sprite
#           if bird: bird.score = self.last_score
#           return self.get_state(), self.current_reward, True

#       bird = self.player.sprite

#       # --- Apply Action ---
#       if action == 1:
#           if bird:
#               bird.update(is_jump=True)
#               # play("jump")


#       self.apply_physics()


#       self.scrollX()
#       self.upcoming_pipes.update(self.world_shift)

#       rightmost_x = -float('inf')
#       for pipe in self.upcoming_pipes:
#           rightmost_x = max(rightmost_x, pipe.rect.right)
#       if rightmost_x < WIDTH - (PIPE_SIZE * 4):
#            self._add_pipe()

#       done = self.handle_collision()

#       self.update_score_and_reward()

#       # --- Get Next State ---
#       next_state = self.get_state()

#       if bird:
#           bird.score = self.last_score

#       return next_state, self.current_reward, done


#   # --- Rendering method ---
#   def draw(self):
#       """Draws all game elements."""
#       self.upcoming_pipes.draw(self.screen)

#       self.player.draw(self.screen)

#       if self.player.sprite: # Check if bird exists
#         self.game.show_score(self.player.sprite.score)
#       else:
#         self.game.show_score(0)



#   # --- Reset method for our agent ---
#   def reset(self):
#       """Resets the game to the initial state for a new episode."""
#       self._generate_world()
#       return self.get_state()

#   def update(self, player_event=None):


import pygame
from pipe import Pipe
from bird import Bird
from game import GameIndicator
from settings import WIDTH, HEIGHT, PIPE_SIZE, PIPE_GAP, DEFAULT_PIPE_PAIR, PIPE_PATTERNS, GROUND_HEIGHT
import random
from sound import play, stop
import numpy as np

class World:
  def __init__(self, screen, theme):
    self.screen = screen
    self.theme = theme
    self.world_shift = 0
    self.current_x = 0
    self.gravity = 0.5

    # pattern for pipes
    self.current_pattern_name = 'zigzag'
    self.pattern_index = 0
    self.pattern_switch_count = 5

    self.upcoming_pipes = pygame.sprite.Group()
    self.player = pygame.sprite.GroupSingle()
    self.pipe_count = 0
    self.is_night = False
    self.theme_switch_count = 15
    self.scored_pipes = set()
    self._generate_world()

    self.playing = False
    self.game_over = False
    self.game_over_sound = False
    self.passed = True
    self.game = GameIndicator(screen, theme)

    # RL specific attributes
    self.current_reward = 0.0
    self.last_score = 0

  def _find_next_pipe(self):
      """Finds the next upcoming pipe pair relative to the bird."""
      bird = self.player.sprite
      closest_pipe_bottom = None
      min_dist = float('inf')

      # Iterate through bottom pipes only to find the pair
      for pipe in self.upcoming_pipes:
          if not pipe.get_is_flipped():
            dist = pipe.rect.left - bird.rect.right
            if dist >= 0 and dist < min_dist:
                min_dist = dist
                closest_pipe_bottom = pipe

      if closest_pipe_bottom:
          for pipe in self.upcoming_pipes:
              if pipe.get_is_flipped() and pipe.rect.left == closest_pipe_bottom.rect.left:
                  return pipe, closest_pipe_bottom
      return None, None


  def get_state(self):
      """Returns the current state representation for the RL agent."""
      bird = self.player.sprite
      if not bird:
          return np.zeros(4, dtype=np.float32)

      top_pipe, bottom_pipe = self._find_next_pipe()

      if top_pipe and bottom_pipe:
          # State calculation
          bird_y_norm = bird.rect.centery / HEIGHT
          bird_vel = bird.direction.y / 10.0
          h_dist = (bottom_pipe.rect.left - bird.rect.right)
          h_dist_norm = h_dist / WIDTH

          gap_center_y = top_pipe.rect.bottom + (bottom_pipe.rect.top - top_pipe.rect.bottom) / 2
          v_dist = (gap_center_y - bird.rect.centery)
          v_dist_norm = v_dist / HEIGHT

          h_dist_norm = np.clip(h_dist_norm, -1, 1)
          v_dist_norm = np.clip(v_dist_norm, -1, 1)
          bird_y_norm = np.clip(bird_y_norm, 0, 1)
          bird_vel = np.clip(bird_vel, -1, 1)


          state = np.array([bird_y_norm, bird_vel, h_dist_norm, v_dist_norm], dtype=np.float32)
          #print(f"State: bird_y={state[0]:.2f}, bird_vel={state[1]:.2f}, h_dist={state[2]:.2f}, v_dist={state[3]:.2f}")
      else:
          bird_y_norm = bird.rect.centery / HEIGHT
          bird_vel = bird.direction.y / 10.0
          state = np.array([bird_y_norm, bird_vel, 0.5, 0.0], dtype=np.float32)

      return state


  def _add_pipe(self):
    if self.pipe_count % self.pattern_switch_count == 0:
        self.current_pattern_name = random.choice(list(PIPE_PATTERNS.keys()))
        self.pattern_index = 0
        print(f"Switched to pattern: {self.current_pattern_name.upper()}")

    current_pattern_info = PIPE_PATTERNS[self.current_pattern_name]
    pipe_pairs = current_pattern_info.get("pairs")
    gap_multiplier = current_pattern_info.get("gap_multiplier", 1.0)

    if pipe_pairs:
        pipe_pair = pipe_pairs[self.pattern_index % len(pipe_pairs)]
        self.pattern_index += 1
    else:
        pipe_pair = random.choice(DEFAULT_PIPE_PAIR)

    pipe_gap = int(PIPE_GAP * gap_multiplier)

    if (self.pipe_count > 0) and (self.pipe_count % self.theme_switch_count == 0):
        self.is_night = not self.is_night
        new_theme = "night" if self.is_night else "day"
        if self.theme.get_current_theme() != new_theme:
            play(new_theme)
            self.theme.set_theme(new_theme)
            print(f"Switched theme to {new_theme.upper()}")

    top_pipe_height = pipe_pair[0] * PIPE_SIZE
    bottom_pipe_y = top_pipe_height + pipe_gap
    bottom_pipe_height_needed = HEIGHT - bottom_pipe_y

    pipe_top = Pipe((WIDTH, top_pipe_height - HEIGHT), PIPE_SIZE, HEIGHT, True, self.is_night, self.theme)

    pipe_bottom = Pipe((WIDTH, bottom_pipe_y), PIPE_SIZE, HEIGHT, False, self.is_night, self.theme)

    self.upcoming_pipes.add(pipe_top)
    self.upcoming_pipes.add(pipe_bottom)
    self.pipe_count += 1


  def _generate_world(self):
    # Reset everything before starting new state
    self.upcoming_pipes.empty()
    self.player.empty()

    # Reset world state
    self.world_shift = 0
    self.current_x = 0
    self.pipe_count = 0
    self.pattern_index = 0
    self.last_score = 0
    self.is_night = False
    self.theme.set_theme("day")
    stop("night")
    stop("day")
    stop("hit")
    stop("background")

    self._add_pipe()
    # Find the first pipe added and shift it slightly right so it's not immediate collision
    first_pipe_x = WIDTH + 150
    for pipe in self.upcoming_pipes:
        pipe.rect.x = first_pipe_x

    # Add player
    bird_start_pos = (WIDTH // 4, HEIGHT // 2 - 50)
    bird = Bird(bird_start_pos, 45)
    self.player.add(bird)

    # Reset game state flags
    self.playing = True
    self.game_over = False
    self.game_over_sound = False
    self.current_reward = 0.0


  def scrollX(self):
    self.world_shift = -6


  def apply_physics(self):
    """Applies gravity to the bird."""
    if self.player.sprite:
      if not self.game_over:
          self.player.sprite.direction.y += self.gravity

      # Update position based on velocity
      self.player.sprite.rect.y += self.player.sprite.direction.y


  def handle_collision(self):
      """Checks for collisions and sets game over state."""
      if self.game_over:
          return True

      bird = self.player.sprite
      if not bird:
          return True

      # Pipe collision
      collided_pipes = pygame.sprite.spritecollide(bird, self.upcoming_pipes, False, pygame.sprite.collide_mask)

      # Ground/Ceiling collision
      hit_ground = bird.rect.bottom >= HEIGHT
      hit_ceiling = bird.rect.top <= 0

      if collided_pipes or hit_ground or hit_ceiling:
          if not self.game_over:
              self.playing = False
              self.game_over = True
              if not self.game_over_sound:
                  #play("hit")
                  current_theme = self.theme.get_current_theme()
                  stop(current_theme)
                  stop("background")
                  self.game_over_sound = True
              # Large negative reward for dying
              self.current_reward = -10.0
          return True
      else:
          for pipe in self.upcoming_pipes:
              if  bird.rect.x >= pipe.rect.centerx:
                  #play("score")
                  bird.score += 1
          return False


  def update_score_and_reward(self):
      """Updates the score and calculates intermediate rewards."""
      if self.game_over:
          return

      bird = self.player.sprite
      if not bird:
          return

      # Reward for surviving
      self.current_reward = 0.1

      passed_pipe = False
      _, next_bottom_pipe = self._find_next_pipe() # Find the pipe we are aiming for

      for pipe in self.upcoming_pipes:
          if not pipe.get_is_flipped() and pipe not in self.scored_pipes and bird.rect.centerx > pipe.rect.centerx:
              self.last_score += 1
              self.scored_pipes.add(pipe)
              # play("score")
              self.current_reward += 5.0
              print(f"Passed Pipe! Score: {self.last_score}, Reward: {self.current_reward}")

  def step(self, action):
      self.current_reward = 0.0

      if self.game_over:
          bird = self.player.sprite
          if bird: bird.score = self.last_score
          return self.get_state(), self.current_reward, True

      bird = self.player.sprite

      # --- Apply Action ---
      if action == 1:
          if bird:
              bird.update(is_jump=True)
              # play("jump")


      self.apply_physics()


      self.scrollX()
      self.upcoming_pipes.update(self.world_shift)

      rightmost_x = -float('inf')
      for pipe in self.upcoming_pipes:
          rightmost_x = max(rightmost_x, pipe.rect.right)
      if rightmost_x < WIDTH - (PIPE_SIZE * 4):
           self._add_pipe()

      done = self.handle_collision()

      self.update_score_and_reward()

      # --- Get Next State ---
      next_state = self.get_state()

      if bird:
          bird.score = self.last_score

      return next_state, self.current_reward, done


  # --- Rendering method ---
  def draw(self):
      """Draws all game elements."""
      self.upcoming_pipes.draw(self.screen)

      self.player.draw(self.screen)

      if self.player.sprite: # Check if bird exists
        self.game.show_score(self.player.sprite.score)
      else:
        self.game.show_score(0)



  # --- Reset method for our agent ---
  def reset(self):
      """Resets the game to the initial state for a new episode."""
      self._generate_world()
      return self.get_state()

  def update(self, player_event=None):
      pass