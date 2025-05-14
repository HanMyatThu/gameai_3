# train_rl.py
import pygame
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt # For plotting results

from settings import WIDTH, HEIGHT, GROUND_HEIGHT
from world import World
from theme import ThemeManager
from dqn_agent import DQNAgent
from sound import stop as soundStop # Use a different name to avoid conflict

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT + GROUND_HEIGHT))
pygame.display.set_caption("Super Flappy Bird - DQN Training")
theme = ThemeManager()
FPS_CLOCK = pygame.time.Clock() # Renamed clock

# --- RL Agent Setup ---
STATE_DIM = 4  # Dimension of our state space
ACTION_DIM = 2 # Number of actions (0: None, 1: Jump)
agent = DQNAgent(STATE_DIM, ACTION_DIM, replay_capacity=20000, batch_size=128, lr=5e-4, gamma=0.98, epsilon_decay=2000) # Adjusted params

# --- Game World Setup ---
world = World(screen, theme)

# --- Training Parameters ---
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 5000 # Prevent infinitely long episodes if agent is perfect
RENDER_EVERY_N_EPISODES = 50 # Render the game visually occasionally
SAVE_MODEL_EVERY_N_EPISODES = 100
TARGET_SCORE = 500 # A high score target to aim for

# --- Logging ---
episode_rewards = []
episode_lengths = []
losses = []
epsilons = []

# --- Load Pre-trained Model (Optional) ---
LOAD_MODEL = False # Set to True to load a saved model
MODEL_PATH = "flappy_bird_dqn.pth"
if LOAD_MODEL:
    agent.load_model(MODEL_PATH)
    # agent.epsilon = agent.epsilon_end # Start with low epsilon if loading trained model

# --- Main Training Loop ---
def train():
    print("Starting Training...")
    best_avg_reward = -float('inf')

    for i_episode in range(1, NUM_EPISODES + 1):
        state = world.reset() # Reset environment and get initial state
        state = np.array(state, dtype=np.float32) # Ensure numpy array

        total_reward = 0
        episode_loss = []
        render_this_episode = (i_episode % RENDER_EVERY_N_EPISODES == 0) or (i_episode == NUM_EPISODES)

        for t in range(MAX_STEPS_PER_EPISODE):
            # --- Pygame Event Handling (allow quitting) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Training interrupted.")
                    agent.save_model(f"flappy_bird_dqn_interrupt_{i_episode}.pth")
                    pygame.quit()
                    sys.exit()
                # Add manual restart? (Optional)
                # if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                #    state = world.reset()
                #    total_reward = 0
                #    break # Break inner loop, start new episode

            # --- Agent chooses action ---
            action_tensor = agent.select_action(state)
            action = action_tensor.item() # Get Python integer action

            # --- Environment steps ---
            next_state, reward, done = world.step(action)
            next_state = np.array(next_state, dtype=np.float32) if next_state is not None else None
            total_reward += reward

            # --- Store experience ---
            # If done, next_state is None for storage purposes in DQN
            agent.store_transition(state, action_tensor, next_state, reward, done)

            # --- Move to the next state ---
            state = next_state

            # --- Perform one step of the optimization (on the policy network) ---
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)

            # --- Render (optional) ---
            if render_this_episode:
                screen.fill((0, 0, 0)) # Clear screen
                # Draw background (needs theme access)
                bg_img = pygame.transform.scale(theme.get('background'), (WIDTH, HEIGHT))
                screen.blit(bg_img, (0, 0))
                # Draw world elements (pipes, bird, score)
                world.draw()
                # Draw ground
                ground_img = theme.get('ground')
                # Simple ground blit (no scrolling needed visually here if world handles it)
                screen.blit(ground_img, (0, HEIGHT))

                pygame.display.flip() # Update the display
                FPS_CLOCK.tick(60) # Control render speed

            if done:
                break # End episode if game over

        # --- End of Episode Logging ---
        episode_rewards.append(total_reward)
        episode_lengths.append(t + 1)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)
        epsilons.append(agent.epsilon) # Log epsilon value at end of episode

        # Calculate average reward over last 100 episodes
        avg_reward_100 = np.mean(episode_rewards[-100:])

        print(f"Episode {i_episode}/{NUM_EPISODES} | Length: {t+1} | Reward: {total_reward:.2f} | Avg Reward (100): {avg_reward_100:.2f} | Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.3f}")

        # --- Save Model Periodically and if improved ---
        if i_episode % SAVE_MODEL_EVERY_N_EPISODES == 0:
            agent.save_model(f"flappy_bird_dqn_ep{i_episode}.pth")
        if avg_reward_100 > best_avg_reward and len(episode_rewards) >= 100:
             best_avg_reward = avg_reward_100
             agent.save_model(f"flappy_bird_dqn_best.pth")
             print(f"*** New Best Average Reward: {best_avg_reward:.2f} - Model Saved ***")

        # --- Optional: Stop training if target score reached consistently ---
        if len(episode_rewards) > 50 and np.mean(episode_rewards[-50:]) > TARGET_SCORE:
            print(f"Target average score ({TARGET_SCORE}) reached over last 50 episodes!")
            agent.save_model("flappy_bird_dqn_final_target.pth")
            # break # Uncomment to stop training early

    print("Training Finished.")
    agent.save_model("flappy_bird_dqn_final.pth")

    # --- Plotting ---
    plot_results(episode_rewards, episode_lengths, losses, epsilons)


def plot_results(rewards, lengths, losses, epsilons):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Reward over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    # Add rolling average
    if len(rewards) >= 100:
        rolling_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(rewards)), rolling_avg, label='100-episode avg')
        plt.legend()


    plt.subplot(2, 2, 2)
    plt.plot(lengths)
    plt.title('Episode Length over Time')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    plt.subplot(2, 2, 3)
    plt.plot(losses)
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 4)
    plt.plot(epsilons)
    plt.title('Epsilon Decay over Time')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')

    plt.tight_layout()
    plt.savefig("training_plots.png") # Save the plot
    print("Training plots saved to training_plots.png")
    # plt.show() # Uncomment to display plot immediately


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
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