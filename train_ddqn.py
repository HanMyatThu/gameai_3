import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pygame
import sys
import numpy as np
from collections import deque
from settings import WIDTH, HEIGHT, GROUND_HEIGHT
from theme import ThemeManager
from world import World
from sound import stop as soundStop
from duel_ddqn import NewAgent

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT + GROUND_HEIGHT))
pygame.display.set_caption("Super Flappy Bird - New Dueling DDQN Training")
clock = pygame.time.Clock()

theme = ThemeManager()

# --- Agent and Environment Setup ---
# State and action dimensions (matching existing models)
STATE_DIM = 4
ACTION_DIM = 2

agent = NewAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM
)

world = World(screen, theme)

# --- Training Parameters ---
MAX_EPISODES = 2000
LOG_INTERVAL = 50
SAVE_MODEL_EVERY_N_EPISODES = 100
TARGET_SCORE = 300

# Track rewards
recent_rewards = deque(maxlen=100)

# --- Main Training Loop ---
for episode in range(1, MAX_EPISODES + 1):
    state = world.reset()
    state = np.array(state, dtype=np.float32)
    done = False
    episode_reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Training interrupted.")
                agent.save_models(episode)
                pygame.quit()
                sys.exit()

        # Agent selects action
        action = agent.get_action(state)

        # Environment responds
        next_state, reward, done = world.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        if done:
            next_state = np.zeros_like(state)

        # Store and learn
        agent.store_transition(state, action, reward, next_state, done)
        agent.train()

        # Move to next state
        state = next_state
        episode_reward += reward

        # Optional rendering (uncomment to visualize occasionally)
        # if episode % 100 == 0:
        #     screen.fill((0, 0, 0))
        #     bg = pygame.transform.scale(theme.get('background'), (WIDTH, HEIGHT))
        #     screen.blit(bg, (0, 0))
        #     world.draw()
        #     screen.blit(theme.get('ground'), (0, HEIGHT))
        #     pygame.display.flip()
        #     clock.tick(60)

    # Decay epsilon and log
    agent.decay_epsilon()
    recent_rewards.append(episode_reward)
    avg_reward = np.mean(recent_rewards)

    if episode % LOG_INTERVAL == 0:
        print(f"Episode {episode:4d} | Reward: {episode_reward:6.2f} | "
              f"Avg (100): {avg_reward:6.2f} | Epsilon: {agent.epsilon:.4f}")

    # Save models periodically and on improvement
    if episode % SAVE_MODEL_EVERY_N_EPISODES == 0:
        agent.save_models(episode)

    # Early stopping if target reached
    if len(recent_rewards) >= 50 and avg_reward >= TARGET_SCORE:
        print(f"Target score of {TARGET_SCORE} reached (avg over last 50 episodes). Saving best model.")
        agent.save_models("best")
        break

# Cleanup
soundStop("background")
soundStop("day")
soundStop("night")
soundStop("hit")
pygame.quit()
sys.exit()
