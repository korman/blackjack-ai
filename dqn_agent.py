import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
from typing import List, Tuple, Dict

# 导入之前的21点游戏代码

from blackjack_game import Card, Deck, Player, BlackjackGame

# 定义状态编码函数


def encode_state(
    player_hand_value: int, player_has_usable_ace: bool, visible_cards: List[Card]
) -> np.ndarray:
    """
    将游戏状态编码为神经网络的输入。

    Args:
        player_hand_value: 玩家手牌当前点数
        player_has_usable_ace: 玩家是否有可作为11点使用的A
        visible_cards: 桌面上可见的牌

    Returns:
        编码后的状态向量
    """
    # 编码玩家自己的状态
    state = [
        player_hand_value / 21.0,  # 归一化手牌点数
        1.0 if player_has_usable_ace else 0.0,
    ]

    # 编码已知牌的分布（牌计数）
    card_counts = [0] * 13  # 2-10, J, Q, K, A
    for card in visible_cards:
        if card.rank == "A":
            card_counts[12] += 1
        elif card.rank in ["J", "Q", "K"]:
            card_counts[9] += 1  # 10点牌统一计数
        else:
            card_counts[int(card.rank) - 2] += 1

    # 归一化牌计数
    num_decks = 1  # 假设使用1副牌
    for i in range(len(card_counts)):
        if i == 9:  # 10点牌(10,J,Q,K)有16张
            card_counts[i] /= 16 * num_decks
        elif i == 12:  # A有4张
            card_counts[i] /= 4 * num_decks
        else:  # 其他每种牌有4张
            card_counts[i] /= 4 * num_decks

    state.extend(card_counts)
    return np.array(state, dtype=np.float32)


# 定义DQN模型


class DQN(nn.Module):
    def __init__(self, state_size, action_size, name="DQN智能体"):
        super(DQN, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.name = name

        # 探索参数
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.9995  # 设置更慢的衰减率

        # 或者使用线性衰减
        self.epsilon_decay_linear = (
            1.0 - self.epsilon_min
        ) / 50000  # 线性衰减，50000回合降到最小值

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 定义经验回放缓冲区


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)


# 定义AI智能体


class DQNAgent:
    def __init__(self, state_size, action_size, name="DQNAgent"):
        self.state_size = state_size
        self.action_size = action_size
        self.name = name
        self.memory = ReplayBuffer(100000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_frequency = 1000
        self.batch_size = 64
        self.train_start = 1000
        self.step = 0

        # 创建Q网络和目标网络
        self.q_network = DQN(state_size, action_size)
        self.target_network = copy.deepcopy(self.q_network)
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # 设置设备(GPU/CPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)

    def act(self, state, training=True):
        """选择动作"""
        if training and np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)  # 探索

        with torch.no_grad():
            state_tensor = torch.FloatTensor(
                state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()  # 利用

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.add(state, action, reward, next_state, done)

    def replay(self):
        """经验回放更新网络"""
        if len(self.memory) < self.train_start:
            return

        # 从经验缓冲区采样批次
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # 转换为PyTorch张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 获取当前Q值
        curr_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 计算下一状态的目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]

        # 计算目标Q值
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 计算损失并更新网络
        loss = self.loss_fn(curr_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 根据需要更新目标网络
        self.step += 1
        if self.step % self.update_target_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # 衰减探索率（使用max函数的更简洁写法）
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        """保存模型"""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "step": self.step,
            },
            filename,
        )

    def load(self, filename):
        """加载模型"""
        checkpoint = torch.load(filename)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.step = checkpoint["step"]


# 修改Player类以支持AI智能体


class AIPlayer(Player):
    def __init__(self, name, agent):
        super().__init__(name, is_ai=True)
        self.agent = agent
        self.visible_cards = []  # 记录所有可见的牌

    def decide_action(self) -> str:
        # 编码当前状态
        player_has_usable_ace = (
            any(card.rank == "A" for card in self.hand)
            and self.calculate_hand_value() <= 21
        )
        state = encode_state(
            self.calculate_hand_value(), player_has_usable_ace, self.visible_cards
        )

        # 通过智能体选择动作
        action_idx = self.agent.act(state)
        return "hit" if action_idx == 0 else "stand"

    def observe_card(self, card):
        """记录观察到的牌"""
        self.visible_cards.append(card)
