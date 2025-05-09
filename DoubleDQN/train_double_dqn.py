


# import sys
# import pygame
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import deque

# from settings import WIDTH, HEIGHT, GROUND_HEIGHT
# from world    import World
# from theme    import ThemeManager
# from sound    import stop as soundStop

# from Ddqn_agent import DoubleDQNAgent  # your agent implementation

# # --- Configuration ---
# STATE_DIM            = 4
# ACTION_DIM           = 2
# MEMORY_SIZE          = 50000 # best 50000
# BATCH_SIZE           = 64
# GAMMA                = 0.99
# LR                   =  0.0000501 # fix #0.000055 #0.00001 #0.00005    #0.005  #0.01 #0.001  #0.0005   #0.0001            #0.00005 and 0.96  the best by far      0.00005 and 0.95 perform way better than  #0.00005 and 0.98 perofrm way better
# EPSILON_START        = 1.0
# EPSILON_DECAY        =  0.96  # fix # best 0.96#0.95#0.98 # 0.97 #0.99 #0.995
# EPSILON_MIN          = 0.01
# UPDATE_TARGET_EVERY  = 300
# MAX_EPISODES         = 2000
# LOG_INTERVAL         = 10
# SAVE_MODEL_EVERY     = 200
# TARGET_AVG_REWARD    = 300
# FPS                  = 60

# # --- Agent & Environment Setup ---
# agent = DoubleDQNAgent(
#     state_dim=STATE_DIM,
#     action_dim=ACTION_DIM,
#     memory_size=MEMORY_SIZE,
#     batch_size=BATCH_SIZE,
#     gamma=GAMMA,
#     lr=LR,
#     epsilon=EPSILON_START,
#     epsilon_decay=EPSILON_DECAY,
#     epsilon_min=EPSILON_MIN,
#     update_target_every=UPDATE_TARGET_EVERY,
#     chkpt_dir='DDQN/models_ddqn'
# )

# pygame.init()
# screen = pygame.display.set_mode((WIDTH, HEIGHT + GROUND_HEIGHT))
# pygame.display.set_caption("Flappy Bird - DDQN Training")
# theme = ThemeManager()
# clock = pygame.time.Clock()

# world = World(screen, theme)

# # --- Logging Containers ---
# recent_rewards  = deque(maxlen=100)
# episode_rewards = []
# episode_lengths = []
# epsilons        = []
# best_avg_reward = -float('inf')

# # --- Main Training Loop ---
# for ep in range(1, MAX_EPISODES + 1):
#     state = world.reset()
#     state = np.array(state, dtype=np.float32)
#     total_reward = 0.0
#     length = 0
#     done = False

#     while not done:
#         # Handle quit event
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 agent.save_models("interrupt")
#                 pygame.quit()
#                 sys.exit()

#         # Agent takes action
#         action = agent.get_action(state)

#         # Environment steps
#         next_state, reward, done = world.step(action)
#         next_state = np.array(next_state, dtype=np.float32)

#         # Learn
#         agent.store_transition(state, action, reward, next_state, done)
#         agent.train()

#         # --- Always render every frame ---
#         screen.fill((0, 0, 0))
#         bg = pygame.transform.scale(theme.get('background'), (WIDTH, HEIGHT))
#         screen.blit(bg, (0, 0))
#         world.draw()
#         screen.blit(theme.get('ground'), (0, HEIGHT))
#         pygame.display.flip()
#         clock.tick(FPS)

#         # Move to next state
#         state = next_state
#         total_reward += reward
#         length += 1

#     # End of episode updates
#     agent.decay_epsilon()
#     recent_rewards.append(total_reward)
#     episode_rewards.append(total_reward)
#     episode_lengths.append(length)
#     epsilons.append(agent.epsilon)

#     avg_reward = np.mean(recent_rewards)

#     # Logging
#     if ep % LOG_INTERVAL == 0:
#         print(f"Ep {ep:4d} | Last Rwd: {total_reward:6.2f} | "
#               f"Avg(100): {avg_reward:6.2f} | Eps: {agent.epsilon:.3f}")

#     # Checkpointing
#     if ep % SAVE_MODEL_EVERY == 0:
#         agent.save_models(ep)

#     if len(recent_rewards) == 100 and avg_reward > best_avg_reward:
#         best_avg_reward = avg_reward
#         agent.save_models("best")
#         print(f"*** New Best Avg Reward: {best_avg_reward:.2f} — Model Saved ***")

#     # Early stopping
#     if len(recent_rewards) == 100 and avg_reward >= TARGET_AVG_REWARD:
#         print(f"Target average reward {TARGET_AVG_REWARD} reached — stopping training.")
#         agent.save_models("final_target")
#         break

# # --- Plotting Results ---
# plt.figure(figsize=(12, 8))

# plt.subplot(3, 1, 1)
# plt.plot(episode_rewards)
# plt.title("Episode Reward")
# plt.xlabel("Episode")
# plt.ylabel("Reward")

# plt.subplot(3, 1, 2)
# plt.plot(episode_lengths)
# plt.title("Episode Length")
# plt.xlabel("Episode")
# plt.ylabel("Steps")

# # plt.subplot(3, 1, 3)
# # plt.plot(epsilons)
# # plt.title("Epsilon over Episodes")
# # plt.xlabel("Episode")
# # plt.ylabel("Epsilon")

# plt.tight_layout()
# plt.savefig("DDQN/ddqn_training_plots.png")
# print("DDQN training plots saved to ddqn_training_plots.png")
# plt.show()

# # --- Cleanup ---
# soundStop("background")
# soundStop("day")
# soundStop("night")
# soundStop("hit")
# pygame.quit()
# sys.exit()

import sys
import pygame
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from settings import WIDTH, HEIGHT, GROUND_HEIGHT
from world    import World
from theme    import ThemeManager
from sound    import stop as soundStop

from Ddqn_agent import DoubleDQNAgent  # your agent implementation

# --- Configuration ---
STATE_DIM            = 4
ACTION_DIM           = 2
MEMORY_SIZE          = 50000
BATCH_SIZE           = 64
GAMMA                = 0.99
LR                   = 0.0000501
EPSILON_START        = 1.0
EPSILON_DECAY        = 0.96
EPSILON_MIN          = 0.01
UPDATE_TARGET_EVERY  = 300

MAX_STEPS            = 3_000_000     # stop after this many env steps
LOG_INTERVAL_STEPS   = 100_000       # print stats every 100k steps
SAVE_INTERVAL_STEPS  = 1_000_000     # checkpoint every 1M steps

TARGET_AVG_REWARD    = 300
FPS                  = 60

# --- Agent & Environment Setup ---
agent = DoubleDQNAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    memory_size=MEMORY_SIZE,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    lr=LR,
    epsilon=EPSILON_START,
    epsilon_decay=EPSILON_DECAY,
    epsilon_min=EPSILON_MIN,
    update_target_every=UPDATE_TARGET_EVERY,
    chkpt_dir='DDQN/models_ddqn'
)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT + GROUND_HEIGHT))
pygame.display.set_caption("Flappy Bird - DDQN Training")
theme = ThemeManager()
clock = pygame.time.Clock()

world = World(screen, theme)

# --- Logging Containers ---
recent_rewards  = deque(maxlen=100)
episode_rewards = []
episode_lengths = []
epsilons        = []
best_avg_reward = -float('inf')

# --- Step-budgeted Training Loop ---
total_steps = 0
episode     = 0

while total_steps < MAX_STEPS:
    episode += 1
    state = world.reset()
    state = np.array(state, dtype=np.float32)
    total_reward = 0.0
    length = 0
    done = False

    while not done and total_steps < MAX_STEPS:
        # handle quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                agent.save_models("interrupt")
                pygame.quit()
                sys.exit()

        # agent action
        action = agent.get_action(state)
        next_state, reward, done = world.step(action)
        next_state = np.array(next_state, dtype=np.float32)

        # store & train
        agent.store_transition(state, action, reward, next_state, done)
        agent.train()

        # count step
        total_steps += 1

        # render
        screen.fill((0, 0, 0))
        bg = pygame.transform.scale(theme.get('background'), (WIDTH, HEIGHT))
        screen.blit(bg, (0, 0))
        world.draw()
        screen.blit(theme.get('ground'), (0, HEIGHT))
        pygame.display.flip()
        clock.tick(FPS)

        # advance
        state = next_state
        total_reward += reward
        length += 1

    # post-episode logging
    agent.decay_epsilon()
    recent_rewards.append(total_reward)
    episode_rewards.append(total_reward)
    episode_lengths.append(length)
    epsilons.append(agent.epsilon)
    avg_reward = np.mean(recent_rewards)

    # logging by steps
    if total_steps // LOG_INTERVAL_STEPS != (total_steps - length) // LOG_INTERVAL_STEPS:
        print(f"[Steps {total_steps:,}] Ep {episode:4d} | Last Rwd: {total_reward:6.2f} | Avg(100): {avg_reward:6.2f} | Eps: {agent.epsilon:.3f}")

    # checkpoint by steps
    if total_steps // SAVE_INTERVAL_STEPS != (total_steps - length) // SAVE_INTERVAL_STEPS:
        agent.save_models(f"step_{total_steps}")
        print(f"... checkpoint at step {total_steps:,} ...")

    # best model
    if len(recent_rewards) == 100 and avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        agent.save_models("best")
        print(f"*** New Best Avg Reward: {best_avg_reward:.2f} at step {total_steps:,} ***")

    # early stop
    if len(recent_rewards) >= 50 and \
        np.mean(list(recent_rewards)[-50:]) >= TARGET_AVG_REWARD:
        print("Target average reward reached over last 50 episodes—stopping training.")
        agent.save_models("final_target")
        break

# plotting
plt.figure(figsize=(12, 8))
plt.subplot(3,1,1)
plt.plot(episode_rewards)
plt.title("Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.subplot(3,1,2)
plt.plot(episode_lengths)
plt.title("Episode Length")
plt.xlabel("Episode")
plt.ylabel("Steps")

plt.subplot(3,1,3)
plt.plot(epsilons)
plt.title("Epsilon over Episodes")
plt.xlabel("Episode")
plt.ylabel("Epsilon")

plt.tight_layout()
plt.savefig("DDQN/ddqn_training_plots.png")
print("DDQN training plots saved to ddqn_training_plots.png")
plt.show()

# cleanup
soundStop("background")
soundStop("day")
soundStop("night")
soundStop("hit")
pygame.quit()
sys.exit()
