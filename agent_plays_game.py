import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import pygame
import sys
import torch
import numpy as np
import time

from settings import WIDTH, HEIGHT, GROUND_HEIGHT
from world import World
from theme import ThemeManager
from ppo_agent import PPOAgent
from Ddqn_agent import DoubleDQNAgent
from sound import stop as soundStop, play as soundPlay

# --- Configuration --- 
MODEL_TYPES = ["ppo", "ddqn"]
MODEL_TAGS = ["best_1", "best"]
MODEL_DIRS = [
    "models/ppo_1/best",
    "models/doubleDQN/best",
    # "models/ppo_experiment3/best",
]
AGENTS = [PPOAgent, DoubleDQNAgent]
FPS = 60
NUM_GAMES = 10

# --- Agent Setup ---
STATE_DIM   = 4
ACTION_DIM  = 2

def parse_args():
    p = argparse.ArgumentParser(description="Play Flappy Bird with different trained models")
    p.add_argument("--model", "-m",
                   type=int,
                   choices=range(1, len(MODEL_TYPES)+1),
                   required=True,
                   help=f"Which model to load (1–{len(MODEL_TYPES)})")
    return p.parse_args()

def init_agent(model_idx):
    type = MODEL_TYPES[model_idx]
    tag = MODEL_TAGS[model_idx]
    directory = MODEL_DIRS[model_idx]

    AgentType = AGENTS[model_idx]
    agent = AgentType(
        state_dim=4,
        action_dim=2,
        chkpt_dir=directory
    )

    try:
        agent.load_models(tag)
        print(model_idx, 'model index')
        if model_idx == 0:
            agent.set_eval_mode()
        else:
            agent.q_online.eval()
        print(f"[INFO] Loaded model '{type}' from '{directory}'")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}\nCheck that {directory}/{tag}.* files exist.")
        pygame.quit()
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] unexpected error loading {tag}: {e}")
        pygame.quit()
        sys.exit(1)

    return agent, tag, directory

def play_game(agent):
    screen = pygame.display.get_surface()
    world = World(screen, theme)
    for i in range(1, NUM_GAMES+1):
        print(f"\n=== Game {i}/{NUM_GAMES} ===")
        state = np.array(world.reset(), dtype=np.float32)
        total_reward, steps = 0.0, 0
        start = time.time()
        done = False

        while not done:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    print("Playback stopped by user.")
                    return

            #  DDQN
            action = agent.get_greedy_action(state)

            next_state, reward, done = world.step(action)
            state = np.array(next_state, dtype=np.float32)
            total_reward += reward
            steps += 1

            # Render
            screen.fill((0,0,0))
            screen.blit(
                pygame.transform.scale(theme.get("background"), (WIDTH, HEIGHT)),
                (0,0)
            )
            world.draw()
            screen.blit(theme.get("ground"), (0, HEIGHT))
            pygame.display.flip()
            clock.tick(FPS)

        duration = time.time() - start
        print(f"Score:       {world.last_score}")
        print(f"TotalReward: {total_reward:.2f}")
        print(f"Steps:       {steps}")
        print(f"Duration:    {duration:.2f}s")
        time.sleep(1)

if __name__ == "__main__":
    args = parse_args()
    model_idx = args.model - 1

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT+GROUND_HEIGHT))
    pygame.display.set_caption(f"Flappy Bird - Model {args.model}")
    theme = ThemeManager()
    clock = pygame.time.Clock()

    agent, tag, directory = init_agent(model_idx)

    try:
        play_game(agent)
    except Exception as e:
        print(f"[ERROR] during playback: {e}")
    finally:
        # Stop any looping sounds
        for s in ("background", "day", "night", "hit"):
            soundStop(s)
        pygame.quit()
        sys.exit(0)