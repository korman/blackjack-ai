# train.py
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm  # For displaying progress bars
from blackjack_env import BlackjackEnv
from dqn_agent import DQNAgent

# Ensure models directory exists
os.makedirs('models', exist_ok=True)


def train_agent(num_episodes=50000, save_frequency=5000):
    """
    Train the agent to play Blackjack

    Parameters:
        num_episodes: Total training episodes
        save_frequency: Frequency for saving the model

    Returns:
        agent: Trained agent
        rewards: List of rewards for each episode
        epsilons: List of exploration rates for each episode
    """
    print("Starting DQN Agent training...")

    # Set up environment and agent
    env = BlackjackEnv()
    state_size = 15  # 2 (player state) + 13 (visible card count)
    action_size = 2  # 0: hit, 1: stand
    agent = DQNAgent(state_size, action_size)

    # Training statistics
    rewards = []
    epsilons = []

    # Use tqdm to display training progress
    progress_bar = tqdm(range(num_episodes), desc="Training Progress")

    # Start training
    for episode in progress_bar:
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Choose action
            action = agent.act(state)

            # Execute action
            next_state, reward, done, _ = env.step(action)

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Experience replay
            agent.replay()

            # Update state and reward
            state = next_state
            total_reward += reward

        # Record statistics
        rewards.append(total_reward)
        epsilons.append(agent.epsilon)

        # Update progress bar information
        if episode > 0 and episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            progress_bar.set_postfix({
                "Avg Reward (Last 100)": f"{avg_reward:.4f}",
                "Epsilon": f"{agent.epsilon:.4f}"
            })

        # Save model periodically
        if (episode + 1) % save_frequency == 0:
            save_path = f"models/blackjack_dqn_episode_{episode+1}.pth"
            agent.save(save_path)
            print(f"\nModel saved to {save_path}")

    # Save final model
    final_path = "models/blackjack_dqn_final.pth"
    agent.save(final_path)
    print(f"Final model saved to {final_path}")

    return agent, rewards, epsilons


def evaluate_agent(agent, num_episodes=1000):
    """
    Evaluate the performance of the trained agent

    Parameters:
        agent: Agent to evaluate
        num_episodes: Number of game episodes for evaluation

    Returns:
        win_rate: Win rate
        draw_rate: Draw rate
        loss_rate: Loss rate
    """
    print(f"\nEvaluating agent performance (playing {num_episodes} games)...")
    env = BlackjackEnv()
    wins = 0
    draws = 0
    losses = 0

    # Use tqdm to display evaluation progress
    for episode in tqdm(range(num_episodes), desc="Evaluation Progress"):
        state = env.reset()
        done = False

        while not done:
            action = agent.act(state, training=False)  # No exploration
            state, reward, done, _ = env.step(action)

        if reward == 1:
            wins += 1
        elif reward == 0:
            draws += 1
        else:
            losses += 1

    win_rate = wins / num_episodes
    draw_rate = draws / num_episodes
    loss_rate = losses / num_episodes

    print("\nEvaluation Results:")
    print(f"Win Rate: {win_rate:.4f} ({wins}/{num_episodes})")
    print(f"Draw Rate: {draw_rate:.4f} ({draws}/{num_episodes})")
    print(f"Loss Rate: {loss_rate:.4f} ({losses}/{num_episodes})")

    return win_rate, draw_rate, loss_rate


def plot_training_results(rewards, epsilons, window_size=1000):
    """
    Plot the training results

    Parameters:
        rewards: List of rewards for each episode
        epsilons: List of exploration rates for each episode
        window_size: Sliding window size for smoothing the reward curve
    """
    print("\nPlotting training results...")
    plt.figure(figsize=(12, 10))

    # Plot smoothed reward curve
    plt.subplot(3, 1, 1)

    # Ensure window size doesn't exceed the length of rewards
    window_size = min(window_size, len(rewards))

    # Calculate moving average
    smoothed_rewards = []
    for i in range(len(rewards) - window_size + 1):
        smoothed_rewards.append(np.mean(rewards[i:i+window_size]))

    plt.plot(range(window_size, len(rewards) + 1), smoothed_rewards)
    plt.title(f'Average Reward (Window Size = {window_size})')
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Reward')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot raw reward curve
    plt.subplot(3, 1, 2)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Training Episodes')
    plt.ylabel('Reward')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot epsilon decay curve
    plt.subplot(3, 1, 3)
    plt.plot(epsilons)
    plt.title('Exploration Rate (Epsilon) Decay')
    plt.xlabel('Training Episodes')
    plt.ylabel('Epsilon')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('blackjack_training_results.png', dpi=300)
    print("Chart saved to 'blackjack_training_results.png'")
    plt.show()


def train_and_evaluate(train_episodes=50000, eval_episodes=1000):
    """
    Complete workflow for training and evaluating the agent

    Parameters:
        train_episodes: Total number of training episodes
        eval_episodes: Number of evaluation episodes
    """
    # Train the agent
    trained_agent, rewards, epsilons = train_agent(num_episodes=train_episodes)

    # Evaluate the agent
    evaluate_agent(trained_agent, num_episodes=eval_episodes)

    # Plot training results
    plot_training_results(rewards, epsilons)

    print("\nTraining and evaluation complete! Model saved to models/ directory.")
    print("You can run play_with_ai.py to play against the trained AI.")


if __name__ == "__main__":
    import argparse

    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Train Blackjack AI Agent')
    parser.add_argument('--episodes', type=int, default=50000,
                        help='Number of training episodes')
    parser.add_argument('--eval', type=int, default=1000,
                        help='Number of evaluation episodes')

    args = parser.parse_args()

    # Train and evaluate
    train_and_evaluate(train_episodes=args.episodes, eval_episodes=args.eval)
