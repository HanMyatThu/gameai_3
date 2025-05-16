import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import pygame
import sys
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
# tags used when calling agent.load_models(tag)
MODEL_TAGS  = ["best_2", "best"]
# base directories under which we expect subfolders: day, hell, space
MODEL_DIRS  = [
    "models/ppo_1",         # for PPO
    "models/doubleDQN",     # for DDQN
]
# agent classes
AGENT_CLASSES = [PPOAgent, DoubleDQNAgent]

# themes we support
THEMES = ["day", "hell", "space","night"]

FPS       = 90
NUM_GAMES = 10

def parse_args():
    p = argparse.ArgumentParser(description="Play Flappy Bird with multiple trained models")
    p.add_argument(
        "--model", "-m",
        type=int,
        choices=[1,2],
        required=True,
        help="1 = PPO, 2 = DDQN"
    )
    return p.parse_args()

def init_agents(model_idx):
    base_dir    = MODEL_DIRS[model_idx]
    tag         = MODEL_TAGS[model_idx]
    AgentClass  = AGENT_CLASSES[model_idx]

    agents = {}
    for theme_name in THEMES:
        chkpt_dir = os.path.join(base_dir, theme_name, "best")
        agent = AgentClass(
            state_dim=4,
            action_dim=2,
            chkpt_dir=chkpt_dir
        )
        try:
            agent.load_models(tag)
            # for PPO we might need eval mode; for DDQN we eval the online net
            if model_idx == 0:
                agent.set_eval_mode()
            else:
                agent.q_online.eval()
            print(f"[INFO] Loaded {MODEL_TYPES[model_idx].upper()} agent for theme '{theme_name}' from '{chkpt_dir}'")
        except Exception as e:
            print(f"[ERROR] loading {theme_name} agent from {chkpt_dir}: {e}")
            pygame.quit()
            sys.exit(1)

        agents[theme_name] = agent

    return agents

def play_game(agents):
    screen = pygame.display.get_surface()
    world  = World(screen, theme, isMulti=True)

    # scrolling offsets
    bg_scroll     = 0
    ground_scroll = 0
    BG_SPEED      = 1
    GRD_SPEED     = 6

    for i in range(1, NUM_GAMES+1):
        print(f"\n=== Game {i}/{NUM_GAMES} ===")
        state = np.array(world.reset(), dtype=np.float32)
        total_reward, steps = 0.0, 0
        done = False

        # pick initial agent based on starting theme
        current_mode  = world.game_mode
        current_agent = agents[current_mode]

        start = time.time()
        while not done:
            # handle quit events
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    print("Playback stopped by user.")
                    return

            # if theme changed in the world, swap agent
            if world.game_mode != current_mode:
                current_mode  = world.game_mode
                current_agent = agents.get(current_mode, current_agent)
                print(f"[SWITCH] Now using '{current_mode}' agent")

            # agent chooses action
            action = current_agent.get_greedy_action(state)

            # step environment
            next_state, reward, done = world.step(action)
            state = np.array(next_state, dtype=np.float32)
            total_reward += reward
            steps += 1

            # --- render scrolling background ---
            bg_img   = pygame.transform.scale(theme.get("background"), (WIDTH, HEIGHT))
            w        = bg_img.get_width()
            bg_scroll = (bg_scroll + BG_SPEED) % w
            screen.blit(bg_img, (-bg_scroll, 0))
            screen.blit(bg_img, (-bg_scroll + w, 0))

            # draw game elements
            world.draw()

            # --- render scrolling ground ---
            gr_img      = theme.get("ground")
            gw          = gr_img.get_width()
            ground_scroll = (ground_scroll + GRD_SPEED) % gw
            screen.blit(gr_img, (-ground_scroll, HEIGHT))
            screen.blit(gr_img, (-ground_scroll + gw, HEIGHT))

            pygame.display.flip()
            clock.tick(FPS)

        duration = time.time() - start
        print(f"Score:       {world.last_score}")
        print(f"TotalReward: {total_reward:.2f}")
        print(f"Steps:       {steps}")
        print(f"Duration:    {duration:.2f}s")
        time.sleep(1)

if __name__ == "__main__":
    args      = parse_args()
    model_idx = args.model - 1  # 0 for PPO, 1 for DDQN

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT + GROUND_HEIGHT))
    pygame.display.set_caption(f"Flappy Bird - {MODEL_TYPES[model_idx].upper()}")
    theme  = ThemeManager()
    clock  = pygame.time.Clock()

    # load all three agents for chosen type
    agents = init_agents(model_idx)

    try:
        play_game(agents)
    except Exception as e:
        print(f"[ERROR] during playback: {e}")
    finally:
        # Stop any looping sounds
        for s in ("background", "day", "night", "hit"):
            soundStop(s)
        pygame.quit()
        sys.exit(0)
