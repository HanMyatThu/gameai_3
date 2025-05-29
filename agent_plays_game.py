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
from dqn_agent import DQNAgent
from sac_agent import SACAgent
from sound import stop as soundStop, play as soundPlay

# --- Configuration --- 
MODEL_TYPES = ["ppo", "ddqn", "dqn", "sac"]
# tags used when calling agent.load_models(tag)
MODEL_TAGS  = ["best_2", "best", "best", "best"]
# base directories under which we expect subfolders: day, hell, space
MODEL_DIRS  = [
    "models/ppo_1",         # for PPO
    "models/doubleDQN",     # for DDQN
    "models",               # for new DQN (will load models/dqn_model_best.pth)
    "models"                # for new SAC (will load models/sac_model_best.pth)
]
# agent classes
AGENT_CLASSES = [PPOAgent, DoubleDQNAgent, DQNAgent, SACAgent]

# themes we support
THEMES = ["day", "hell", "space","night"]
DEFAULT = ["best"]

FPS       = 120
NUM_GAMES = 10

def parse_args():
    p = argparse.ArgumentParser(description="Play Flappy Bird with multiple trained models")
    p.add_argument(
        "--model", "-m",
        type=int,
        choices=[1,2,3,4],
        required=True,
        help="1 = PPO, 2 = DDQN, 3 = DQN (new), 4 = SAC (new)"
    )
    return p.parse_args()

def init_agents(model_idx, isMulti=True):
    base_dir   = MODEL_DIRS[model_idx]
    tag        = MODEL_TAGS[model_idx]
    AgentClass = AGENT_CLASSES[model_idx]
    model_type_name = MODEL_TYPES[model_idx].upper()
    agents     = {}

    if model_type_name == "DQN" or model_type_name == "SAC":
        agent_init_params = {
            'state_dim': 4,
            'action_dim': 2,
            'chkpt_dir': base_dir
        }
        if model_type_name == "SAC":
            agent_init_params['action_space_type'] = 'continuous'
            agent_init_params['action_dim'] = 1

        agent = AgentClass(**agent_init_params)
        try:
            model_load_path = os.path.join(base_dir, f"{MODEL_TYPES[model_idx]}_model_{tag}.pth")
            if model_type_name == "DQN":
                agent.load_models(tag=tag, model_path_override=model_load_path)
                agent.policy_net.eval()
            elif model_type_name == "SAC":
                agent.load_models(tag=tag, model_path_override=model_load_path)
                agent.set_eval_mode()
            
            print(f"[INFO] Loaded {model_type_name} agent from '{model_load_path}'")
        except Exception as e:
            print(f"[ERROR] loading {model_type_name} agent from {model_load_path}: {e}")
            pygame.quit()
            sys.exit(1)
        
        agents["day"] = agent
        isMulti = False

    elif isMulti:
        for theme_name in THEMES:
            theme_specific_chkpt_dir = os.path.join(base_dir, theme_name, "best")
            agent = AgentClass(
                state_dim=4,
                action_dim=2,
                chkpt_dir=theme_specific_chkpt_dir
            )
            try:
                agent.load_models(tag)
                if model_type_name == "PPO":
                    agent.set_eval_mode()
                elif model_type_name == "DDQN":
                    agent.q_online.eval()
                print(f"[INFO] Loaded {model_type_name} agent for theme '{theme_name}' from '{theme_specific_chkpt_dir}' with tag '{tag}'")
            except Exception as e:
                print(f"[ERROR] loading {theme_name} agent for {model_type_name} from {theme_specific_chkpt_dir}: {e}")
                pygame.quit()
                sys.exit(1)
            agents[theme_name] = agent
    else:
        theme_name = "day"
        single_theme_chkpt_dir = os.path.join(base_dir, theme_name, "best")
        agent = AgentClass(
            state_dim=4,
            action_dim=2,
            chkpt_dir=single_theme_chkpt_dir
        )
        try:
            agent.load_models(tag)
            if model_type_name == "PPO":
                agent.set_eval_mode()
            elif model_type_name == "DDQN":
                agent.q_online.eval()
            print(f"[INFO] Loaded {model_type_name} agent (single mode) for theme '{theme_name}' from '{single_theme_chkpt_dir}' with tag '{tag}'")
        except Exception as e:
            print(f"[ERROR] loading single agent for {model_type_name} from {single_theme_chkpt_dir}: {e}")
            pygame.quit()
            sys.exit(1)
        agents["day"] = agent
    
    return agents, isMulti


def play_game(agents, isMulti=True):
    screen = pygame.display.get_surface()
    world  = World(screen, theme, isMulti)

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
          if isMulti: 
            if world.game_mode != current_mode:
              current_mode  = world.game_mode
              current_agent = agents.get(current_mode, current_agent)
              print(f"[SWITCH] Now using '{current_mode}' agent")
          else: 
            pass

          # agent chooses action
          action = current_agent.get_greedy_action(state)

          final_action = action # Default action
          # Discretize action if the current agent is SAC and continuous
          if isinstance(current_agent, SACAgent) and current_agent.action_space_type == 'continuous':
              # current_agent.get_action for continuous SAC with action_dim=1 returns a scalar numpy float.
              # current_agent.get_greedy_action should also return a scalar for continuous SAC with action_dim=1.
              action_value = action 
              final_action = 1 if action_value > 0.0 else 0 # Example: flap if positive value, else do nothing
              # print(f"SAC raw: {action_value}, discrete: {final_action}") # Optional: for debugging

          # step environment
          next_state, reward, done = world.step(final_action) # Use final_action
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
    model_idx = args.model - 1  # 0 for PPO, 1 for DDQN, 2 for DQN, 3 for SAC

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT + GROUND_HEIGHT))
    pygame.display.set_caption(f"Flappy Bird - {MODEL_TYPES[model_idx].upper()}")
    theme  = ThemeManager()
    clock  = pygame.time.Clock()

    # Determine if multi-theme based on original logic for PPO/DDQN
    # For new DQN/SAC, init_agents will override isMulti to False
    original_isMulti = True if model_idx == 0 else False 

    # load agents
    agents, final_isMulti = init_agents(model_idx, original_isMulti)

    try:
      play_game(agents, final_isMulti)
    except Exception as e:
      print(f"[ERROR] during playback: {e}")
    finally:
      # Stop any looping sounds
      for s in ("background", "day", "night", "hit"):
          soundStop(s)
      pygame.quit()
      sys.exit(0)
