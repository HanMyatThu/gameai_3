import pygame
import sys
import torch
import numpy as np
import time

from settings import WIDTH, HEIGHT, GROUND_HEIGHT
from world import World
from theme import ThemeManager
from ppo_agent import PPOAgent
from sound import stop as soundStop, play as soundPlay

# --- Configuration ---
MODEL_TAG = "best_1"
MODEL_DIR = 'models/ppo_1/best'
FPS = 60
NUM_GAMES = 10

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT + GROUND_HEIGHT))
pygame.display.set_caption(f"Flappy Bird - PPO Playing (Model: {MODEL_TAG})")
theme = ThemeManager()
FPS_CLOCK = pygame.time.Clock()

# --- PPO Agent Setup ---
STATE_DIM = 4
ACTION_DIM = 2

agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, chkpt_dir=MODEL_DIR)

# --- Load Pre-trained Model ---
try:
    agent.load_models(MODEL_TAG)
    agent.set_eval_mode()
    print(f"Successfully loaded PPO model '{MODEL_TAG}' from {MODEL_DIR}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the model files exist and MODEL_TAG/MODEL_DIR are correct.")
    pygame.quit()
    sys.exit()
except Exception as e:
    print(f"An unexpected error occurred loading the model: {e}")
    pygame.quit()
    sys.exit()


# --- Game World Setup ---
world = World(screen, theme)

# --- Main Play Loop ---
def play_game():
    prev_score = 0
    for i_game in range(1, NUM_GAMES + 1):
        print(f"\n--- Starting Game {i_game}/{NUM_GAMES} ---")
        state = world.reset()
        state = np.array(state, dtype=np.float32)
        done = False
        total_reward = 0
        game_steps = 0
        start_time = time.time()

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Playback stopped by user.")
                    return

            action = agent.get_greedy_action(state)

            next_state, reward, done = world.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            total_reward += reward
            game_steps += 1

            state = next_state

            # --- Render ---
            screen.fill((0, 0, 0))
            bg_img = pygame.transform.scale(theme.get('background'), (WIDTH, HEIGHT))
            screen.blit(bg_img, (0, 0))
            world.draw()
            ground_img = theme.get('ground')
            screen.blit(ground_img, (0, HEIGHT))

            pygame.display.flip()
            FPS_CLOCK.tick(FPS)

        # --- End of Game ---
        end_time = time.time()
        duration = end_time - start_time
        print(f"Game {i_game} Over!")
        print(f" Score: {world.last_score}")
        print(f" Last Best Score: {prev_score}")
        curr_score = world.last_score
        if prev_score < curr_score:
            prev_score = curr_score
            print(f" New Best Score: {prev_score}")
        print(f" Total Reward: {total_reward:.2f}")
        print(f" Steps: {game_steps}")
        print(f" Duration: {duration:.2f} seconds")
        time.sleep(1)

    print(f"Best Score: {prev_score}")
    print("\nFinished playing all games.")


if __name__ == "__main__":
    try:
        play_game()
    except Exception as e:
        print(f"\nAn error occurred during playback: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up sounds and pygame
        soundStop("background")
        soundStop("day")
        soundStop("night")
        soundStop("hit")
        pygame.quit()
        sys.exit()