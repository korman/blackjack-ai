import os
import torch
import numpy as np
import random
from blackjack_env import BlackjackEnv
from dqn_agent import DQNAgent


def clear_screen():
    """清屏函数"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_cards(cards):
    """打印牌的函数"""
    card_str = ""
    for card in cards:
        # 显示卡牌的值
        card_str += f"{card.rank}{card.suit} "
    return card_str.strip()


def calculate_hand_value(cards):
    """计算手牌点数"""
    value = 0
    aces = 0

    for card in cards:
        if card.rank == 'A':  # Ace
            aces += 1
            value += 11
        elif card.rank in ['J', 'Q', 'K']:  # Face cards
            value += 10
        else:
            value += int(card.rank)

    # 如果有A且点数超过21，将A视为1点
    while value > 21 and aces:
        value -= 10
        aces -= 1

    return value


def load_agent(model_path, state_size=15, action_size=2):
    """加载训练好的智能体"""
    agent = DQNAgent(state_size, action_size)

    # 加载保存的模型
    checkpoint = torch.load(model_path)

    # 将保存的状态加载到agent中
    agent.q_network.load_state_dict(checkpoint['q_network'])
    agent.target_network.load_state_dict(checkpoint['target_network'])
    agent.optimizer.load_state_dict(checkpoint['optimizer'])
    agent.epsilon = checkpoint.get('epsilon', 0.0)
    agent.step = checkpoint.get('step', 0)

    # 设置为评估模式
    agent.q_network.eval()
    agent.epsilon = 0.0  # 设置为不探索

    return agent


def play_with_ai():
    """与AI对战的主函数"""
    # 获取模型文件列表
    if not os.path.exists("models"):
        os.makedirs("models")

    model_files = [f for f in os.listdir("models") if f.endswith(".pth")]

    if not model_files:
        print("没有找到训练好的模型！请先训练模型。")
        return

    # 按照文件名排序，选择最新的模型
    model_files.sort()
    latest_model = os.path.join("models", model_files[-1])

    print(f"正在加载最新模型: {latest_model}")
    agent = load_agent(latest_model)
    env = BlackjackEnv()

    playing = True
    wins = 0
    losses = 0
    draws = 0

    while playing:
        clear_screen()
        print("\n===== 21点游戏 人类 vs AI =====")
        print(f"战绩: {wins}胜 {losses}负 {draws}平\n")

        # 初始化游戏
        state = env.reset()
        player_cards = env.player.hand
        dealer_cards = env.opponents[0].hand  # 把第一个对手当作庄家
        game_over = False

        while not game_over:
            clear_screen()
            print("\n===== 21点游戏 人类 vs AI =====")
            print(f"战绩: {wins}胜 {losses}负 {draws}平\n")

            # 显示当前状态
            print(f"庄家的牌: {print_cards([dealer_cards[0]])} ?")
            print(
                f"您的牌: {print_cards(player_cards)} (点数: {calculate_hand_value(player_cards)})")

            # 玩家回合
            if calculate_hand_value(player_cards) == 21:
                print("21点！等待庄家...")
                action = 1  # 停牌
            else:
                while True:
                    choice = input("\n您要做什么? (h-要牌, s-停牌): ").lower()
                    if choice in ['h', 's']:
                        break
                    print("无效的选择，请重新输入")

                action = 0 if choice == 'h' else 1

            # 执行动作
            next_state, reward, done, _ = env.step(action)
            state = next_state
            player_cards = env.player.hand
            dealer_cards = env.opponents[0].hand

            # 检查是否游戏结束
            if done:
                game_over = True
                clear_screen()
                print("\n===== 游戏结束 =====")

                # 显示最终状态
                print(
                    f"庄家的牌: {print_cards(dealer_cards)} (点数: {calculate_hand_value(dealer_cards)})")
                print(
                    f"您的牌: {print_cards(player_cards)} (点数: {calculate_hand_value(player_cards)})")

                # 显示结果
                if reward == 1:
                    print("\n恭喜，您赢了！")
                    wins += 1
                elif reward == 0:
                    print("\n平局！")
                    draws += 1
                else:
                    print("\n很遗憾，您输了！")
                    losses += 1

                # 询问是否继续
                continue_playing = input("\n是否继续游戏? (y/n): ").lower()
                if continue_playing != 'y':
                    playing = False
                break

            # 显示更新后的玩家牌
            print(f"\n您抽了一张牌...")
            print(
                f"您的牌: {print_cards(player_cards)} (点数: {calculate_hand_value(player_cards)})")

            if calculate_hand_value(player_cards) > 21:
                print("爆牌了！")
                input("按回车键继续...")
                game_over = True

                clear_screen()
                print("\n===== 游戏结束 =====")
                print(
                    f"庄家的牌: {print_cards(dealer_cards)} (点数: {calculate_hand_value(dealer_cards)})")
                print(
                    f"您的牌: {print_cards(player_cards)} (点数: {calculate_hand_value(player_cards)})")
                print("\n很遗憾，您输了！")
                losses += 1

                # 询问是否继续
                continue_playing = input("\n是否继续游戏? (y/n): ").lower()
                if continue_playing != 'y':
                    playing = False

    print("\n感谢游玩！最终战绩:")
    print(f"{wins}胜 {losses}负 {draws}平")
    win_rate = wins / (wins + losses + draws) * \
        100 if (wins + losses + draws) > 0 else 0
    print(f"胜率: {win_rate:.1f}%")


if __name__ == "__main__":
    play_with_ai()
