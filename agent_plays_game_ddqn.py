import pygame
import sys
import torch
import numpy as np
import time

from settings import WIDTH, HEIGHT, GROUND_HEIGHT
from world import World
from theme import ThemeManager
from Ddqn_agent import DoubleDQNAgent
from sound import stop as soundStop, play as soundPlay

# --- Configuration ---
MODEL_TAG    = "best"              # or "latest", or an episode number
MODEL_DIR    = 'DDQN/models_ddqn'
FPS          = 60
NUM_GAMES    = 10

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT + GROUND_HEIGHT))
pygame.display.set_caption(f"Flappy Bird - DDQN Playing (Model: {MODEL_TAG})")
theme = ThemeManager()
clock = pygame.time.Clock()

# --- DDQN Agent Setup ---
STATE_DIM   = 4
ACTION_DIM  = 2

agent = DoubleDQNAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    chkpt_dir=MODEL_DIR
)

# --- Load Pre-trained Models ---
try:
    agent.load_models(MODEL_TAG)
    agent.q_online.eval()
    print(f"Successfully loaded DDQN model '{MODEL_TAG}' from {MODEL_DIR}")
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    pygame.quit()
    sys.exit()
except Exception as e:
    print(f"Unexpected error: {e}")
    pygame.quit()
    sys.exit()

# --- Game World Setup ---
world = World(screen, theme)

def play_game():
    for game_idx in range(1, NUM_GAMES + 1):
        print(f"\n--- Starting Game {game_idx}/{NUM_GAMES} ---")
        state = world.reset()
        state = np.array(state, dtype=np.float32)
        done = False
        total_reward = 0.0
        steps = 0
        start_time = time.time()

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Playback stopped by user.")
                    return

            action = agent.get_action(state)
            next_state, reward, done = world.step(action)
            state = np.array(next_state, dtype=np.float32)
            total_reward += reward
            steps += 1

            # --- Render ---
            screen.fill((0, 0, 0))
            bg = pygame.transform.scale(theme.get('background'), (WIDTH, HEIGHT))
            screen.blit(bg, (0, 0))
            world.draw()
            ground = theme.get('ground')
            screen.blit(ground, (0, HEIGHT))
            pygame.display.flip()
            clock.tick(FPS)

        duration = time.time() - start_time
        print(f"Game {game_idx} Over!")
        print(f" Score: {world.last_score}")
        print(f" Total Reward: {total_reward:.2f}")
        print(f" Steps: {steps}")
        print(f" Duration: {duration:.2f} sec")
        time.sleep(1)

    print("\nFinished all games.")

if __name__ == "__main__":
    try:
        play_game()
    except Exception as e:
        print(f"\nError during playback: {e}")
        import traceback; traceback.print_exc()
    finally:
        # Clean up sounds & pygame
        soundStop("background")
        soundStop("day")
        soundStop("night")
        soundStop("hit")
        pygame.quit()
        sys.exit()