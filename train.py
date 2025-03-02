# train.py
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm  # 用于显示进度条
from blackjack_env import BlackjackEnv
from dqn_agent import DQNAgent

# 确保模型目录存在
os.makedirs('models', exist_ok=True)


def train_agent(num_episodes=50000, save_frequency=5000):
    """
    训练智能体玩21点

    参数:
        num_episodes: 训练总回合数
        save_frequency: 保存模型的频率

    返回:
        agent: 训练好的智能体
        rewards: 每回合的奖励列表
        epsilons: 每回合的探索率列表
    """
    print("开始训练DQN智能体...")

    # 设置环境和智能体
    env = BlackjackEnv()
    state_size = 15  # 2 (玩家状态) + 13 (可见牌的计数)
    action_size = 2  # 0: hit, 1: stand
    agent = DQNAgent(state_size, action_size)

    # 训练统计
    rewards = []
    epsilons = []

    # 使用tqdm显示训练进度
    progress_bar = tqdm(range(num_episodes), desc="训练进度")

    # 开始训练
    for episode in progress_bar:
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 选择动作
            action = agent.act(state)

            # 执行动作
            next_state, reward, done, _ = env.step(action)

            # 存储经验
            agent.remember(state, action, reward, next_state, done)

            # 经验回放
            agent.replay()

            # 更新状态和奖励
            state = next_state
            total_reward += reward

        # 记录统计数据
        rewards.append(total_reward)
        epsilons.append(agent.epsilon)

        # 更新进度条信息
        if episode > 0 and episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            progress_bar.set_postfix({
                "平均奖励(最近100局)": f"{avg_reward:.4f}",
                "探索率": f"{agent.epsilon:.4f}"
            })

        # 定期保存模型
        if (episode + 1) % save_frequency == 0:
            save_path = f"models/blackjack_dqn_episode_{episode+1}.pth"
            agent.save(save_path)
            print(f"\n模型已保存到 {save_path}")

    # 保存最终模型
    final_path = "models/blackjack_dqn_final.pth"
    agent.save(final_path)
    print(f"最终模型已保存到 {final_path}")

    return agent, rewards, epsilons


def evaluate_agent(agent, num_episodes=1000):
    """
    评估训练好的智能体的性能

    参数:
        agent: 要评估的智能体
        num_episodes: 评估的游戏回合数

    返回:
        win_rate: 胜率
        draw_rate: 平局率
        loss_rate: 负率
    """
    print(f"\n评估智能体性能 (进行 {num_episodes} 局游戏)...")
    env = BlackjackEnv()
    wins = 0
    draws = 0
    losses = 0

    # 使用tqdm显示评估进度
    for episode in tqdm(range(num_episodes), desc="评估进度"):
        state = env.reset()
        done = False

        while not done:
            action = agent.act(state, training=False)  # 不使用探索
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

    print("\n评估结果:")
    print(f"胜率: {win_rate:.4f} ({wins}/{num_episodes})")
    print(f"平局率: {draw_rate:.4f} ({draws}/{num_episodes})")
    print(f"负率: {loss_rate:.4f} ({losses}/{num_episodes})")

    return win_rate, draw_rate, loss_rate


def plot_training_results(rewards, epsilons, window_size=1000):
    """
    绘制训练过程的结果

    参数:
        rewards: 每回合的奖励列表
        epsilons: 每回合的探索率列表
        window_size: 滑动窗口大小，用于平滑奖励曲线
    """
    print("\n绘制训练结果...")
    plt.figure(figsize=(12, 10))

    # 绘制平滑后的奖励曲线
    plt.subplot(3, 1, 1)

    # 确保窗口大小不超过奖励列表长度
    window_size = min(window_size, len(rewards))

    # 计算滑动平均
    smoothed_rewards = []
    for i in range(len(rewards) - window_size + 1):
        smoothed_rewards.append(np.mean(rewards[i:i+window_size]))

    plt.plot(range(window_size, len(rewards) + 1), smoothed_rewards)
    plt.title(f'平均奖励 (滑动窗口大小 = {window_size})')
    plt.xlabel('训练回合')
    plt.ylabel('平均奖励')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 绘制原始奖励曲线
    plt.subplot(3, 1, 2)
    plt.plot(rewards)
    plt.title('每回合奖励')
    plt.xlabel('训练回合')
    plt.ylabel('奖励')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 绘制探索率衰减曲线
    plt.subplot(3, 1, 3)
    plt.plot(epsilons)
    plt.title('探索率(Epsilon)衰减')
    plt.xlabel('训练回合')
    plt.ylabel('Epsilon')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('blackjack_training_results.png', dpi=300)
    print("图表已保存到 'blackjack_training_results.png'")
    plt.show()


def train_and_evaluate(train_episodes=50000, eval_episodes=1000):
    """
    训练并评估智能体的完整流程

    参数:
        train_episodes: 训练的总回合数
        eval_episodes: 评估的回合数
    """
    # 训练智能体
    trained_agent, rewards, epsilons = train_agent(num_episodes=train_episodes)

    # 评估智能体
    evaluate_agent(trained_agent, num_episodes=eval_episodes)

    # 绘制训练结果
    plot_training_results(rewards, epsilons)

    print("\n训练和评估完成！模型已保存到 models/ 目录。")
    print("你可以运行 play_with_ai.py 与训练好的AI对战。")


if __name__ == "__main__":
    import argparse

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='训练21点AI智能体')
    parser.add_argument('--episodes', type=int, default=50000, help='训练的回合数')
    parser.add_argument('--eval', type=int, default=1000, help='评估的回合数')

    args = parser.parse_args()

    # 训练并评估
    train_and_evaluate(train_episodes=args.episodes, eval_episodes=args.eval)
