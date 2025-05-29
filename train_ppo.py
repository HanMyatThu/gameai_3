import pygame
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from settings import WIDTH, HEIGHT, GROUND_HEIGHT
from world import World
from theme import ThemeManager
from ppo_agent import PPOAgent
from sound import stop as soundStop

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT + GROUND_HEIGHT))
pygame.display.set_caption("Super Flappy Bird - PPO Training")
theme = ThemeManager()
FPS_CLOCK = pygame.time.Clock()

# PPO Agent Setup
STATE_DIM = 4
ACTION_DIM = 2
LEARNING_RATE = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
POLICY_CLIP = 0.2
BATCH_SIZE = 128
N_EPOCHS = 10
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
UPDATE_TIMESTEP = 2048

agent = PPOAgent(STATE_DIM, ACTION_DIM, lr=LEARNING_RATE, gamma=GAMMA,
                 gae_lambda=GAE_LAMBDA, policy_clip=POLICY_CLIP,
                 batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
                 entropy_coef=ENTROPY_COEF, value_loss_coef=VALUE_LOSS_COEF)

# Game World Setup
world = World(screen, theme)

# HyperParameters
MAX_TOTAL_TIMESTEPS = 2000000
LOG_INTERVAL = 20
RENDER_EVERY_N_EPISODES = 100
SAVE_MODEL_EVERY_N_EPISODES = 200
TARGET_SCORE = 300


recent_rewards = deque(maxlen=100)
recent_lengths = deque(maxlen=100)
episode_rewards_history = []
episode_lengths_history = []
actor_losses = []
critic_losses = []
entropy_losses = []
total_losses = []


def train():
    print("Starting PPO Training...")
    current_total_steps = 0
    episode = 0
    best_avg_reward = -float('inf')

    while current_total_steps < MAX_TOTAL_TIMESTEPS:
        episode += 1
        state = world.reset()
        state = np.array(state, dtype=np.float32)
        current_episode_reward = 0
        current_episode_length = 0
        done = False
        agent.set_eval_mode()
        agent.set_train_mode()

        # Collects transitions until memory has enough data for one update cycle
        steps_collected_in_update = 0
        while steps_collected_in_update < UPDATE_TIMESTEP:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Training interrupted.")
                    agent.save_models("interrupt")
                    pygame.quit()
                    sys.exit()

            action, log_prob, value = agent.select_action(state)

            next_state, reward, done = world.step(action)
            next_state = np.array(next_state, dtype=np.float32) if not done else np.zeros_like(state)
            if done: next_state = np.array(state, dtype=np.float32)


            # Store experience
            agent.store_transition(state, action, log_prob, value, reward, done)

            state = next_state
            current_episode_reward += reward
            current_total_steps += 1
            current_episode_length += 1
            steps_collected_in_update += 1

            render_this_episode = (episode % RENDER_EVERY_N_EPISODES == 0)
            if render_this_episode:
                screen.fill((0, 0, 0))
                bg_img = pygame.transform.scale(theme.get('background'), (WIDTH, HEIGHT))
                screen.blit(bg_img, (0, 0))
                world.draw()
                ground_img = theme.get('ground')
                screen.blit(ground_img, (0, HEIGHT))
                pygame.display.flip()
                FPS_CLOCK.tick(60) # Control render speed

            if done:
                break

        if len(agent.memory) >= agent.batch_size:
            print(f"--- Performing Update (Episode {episode}, Total Steps: {current_total_steps}) ---")
            agent.set_train_mode()
            avg_losses, avg_total_loss = agent.learn()
            print(f"Update Losses: Actor={avg_losses.get('actor_loss', 0):.4f}, Critic={avg_losses.get('critic_loss', 0):.4f}, Entropy={avg_losses.get('entropy_loss', 0):.4f}")

            actor_losses.append(avg_losses.get('actor_loss', 0))
            critic_losses.append(avg_losses.get('critic_loss', 0))
            entropy_losses.append(avg_losses.get('entropy_loss', 0))
            total_losses.append(avg_total_loss)
        else:
             print(f"Warning: Skipped learning step. Memory size {len(agent.memory)} < {agent.batch_size}")


        recent_rewards.append(current_episode_reward)
        recent_lengths.append(current_episode_length)
        episode_rewards_history.append(current_episode_reward)
        episode_lengths_history.append(current_episode_length)

        avg_reward_100 = np.mean(recent_rewards) if recent_rewards else 0
        avg_length_100 = np.mean(recent_lengths) if recent_lengths else 0

        if episode % LOG_INTERVAL == 0:
            print(f"Ep: {episode} | Steps: {current_total_steps}/{MAX_TOTAL_TIMESTEPS} | Avg Rew (100): {avg_reward_100:.2f} | Last Rew: {current_episode_reward:.2f} | Avg Len (100): {avg_length_100:.1f}")

        # Save Model Periodically and if improved
        if episode % SAVE_MODEL_EVERY_N_EPISODES == 0:
            agent.save_models(episode)
        if avg_reward_100 > best_avg_reward and len(recent_rewards) == 100: # Ensure we have 100 samples
             best_avg_reward = avg_reward_100
             agent.save_models("best")
             print(f"*** New Best Average Reward: {best_avg_reward:.2f} - Model Saved ***")
        if len(recent_rewards) >= 50 and np.mean(list(recent_rewards)[-50:]) > TARGET_SCORE:
             print(f"Target average score ({TARGET_SCORE}) reached over last 50 episodes!")
             agent.save_models("final_target")
             break


    print("Training Finished.")
    agent.save_models("final")
    plot_results(episode_rewards_history, episode_lengths_history, actor_losses, critic_losses, entropy_losses, total_losses)


def plot_results(rewards, lengths, actor_l, critic_l, entropy_l, total_l):
    plt.figure(figsize=(18, 12))

    plt.subplot(3, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Reward over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    if len(rewards) >= 100:
        rolling_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(rewards)), rolling_avg, label='100-episode avg')
        plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(lengths)
    plt.title('Episode Length over Time')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    if len(lengths) >= 100:
         rolling_avg_len = np.convolve(lengths, np.ones(100)/100, mode='valid')
         plt.plot(np.arange(99, len(lengths)), rolling_avg_len, label='100-episode avg')
         plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(actor_l)
    plt.title('Actor Loss per Update')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')

    plt.subplot(3, 2, 4)
    plt.plot(critic_l)
    plt.title('Critic Loss per Update')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')

    plt.subplot(3, 2, 5)
    plt.plot(entropy_l)
    plt.title('Entropy per Update')
    plt.xlabel('Update Step')
    plt.ylabel('Entropy')

    plt.subplot(3, 2, 6)
    plt.plot(total_l)
    plt.title('Approx Total Loss per Update')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')


    plt.tight_layout()
    plt.savefig("ppo_training_plots.png")
    print("PPO Training plots saved to ppo_training_plots.png")
    plt.show()

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\nAn error occurred during PPO training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        soundStop("background")
        soundStop("day")
        soundStop("night")
        soundStop("hit")
        pygame.quit()
        sys.exit()