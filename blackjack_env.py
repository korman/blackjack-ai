# blackjack_env.py
import numpy as np
from blackjack_game import Deck, Player, Card


def encode_state(player_hand_value: int, player_has_usable_ace: bool,
                 visible_cards: list) -> np.ndarray:
    """
    将游戏状态编码为神经网络的输入。

    参数:
        player_hand_value: 玩家手牌点数
        player_has_usable_ace: 玩家是否有可用的A（计为11点而不爆牌）
        visible_cards: 可见的牌列表

    返回:
        numpy数组，表示状态向量
    """
    # 编码玩家自己的状态
    state = [
        player_hand_value / 21.0,  # 归一化手牌点数
        1.0 if player_has_usable_ace else 0.0
    ]

    # 编码已知牌的分布（牌计数）
    card_counts = [0] * 13  # 2-10, J, Q, K, A
    for card in visible_cards:
        if card.rank == 'A':
            card_counts[12] += 1
        elif card.rank in ['J', 'Q', 'K']:
            card_counts[9] += 1  # 10点牌统一计数
        elif card.rank == '10':
            card_counts[8] += 1
        else:
            card_counts[int(card.rank) - 2] += 1

    # 归一化牌计数
    num_decks = 1  # 假设使用1副牌
    for i in range(len(card_counts)):
        if i == 9:  # 10点牌(J,Q,K)有12张
            card_counts[i] /= (12 * num_decks)
        elif i == 8:  # 10有4张
            card_counts[i] /= (4 * num_decks)
        elif i == 12:  # A有4张
            card_counts[i] /= (4 * num_decks)
        else:  # 其他每种牌有4张
            card_counts[i] /= (4 * num_decks)

    state.extend(card_counts)
    return np.array(state, dtype=np.float32)


class BlackjackEnv:
    """21点游戏环境，适用于强化学习"""

    def __init__(self):
        """初始化环境"""
        self.deck = None
        self.player = None
        self.opponents = []
        self.visible_cards = []  # 记录所有可见的牌
        self.done = False
        self.reward = 0

    def reset(self):
        """
        重置环境，返回初始状态

        返回:
            初始状态向量
        """
        self.deck = Deck()
        self.deck.shuffle()
        self.player = Player("AI学习者")
        self.opponents = [Player("对手1", is_ai=True), Player("对手2", is_ai=True)]
        self.visible_cards = []
        self.done = False
        self.reward = 0

        # 初始发牌
        for _ in range(2):
            for p in [self.player] + self.opponents:
                card = self.deck.deal()
                p.add_card(card)
                if p != self.player:  # 只记录对手的牌作为可见牌
                    self.visible_cards.append(card)

        # 返回初始状态
        player_has_usable_ace = self._has_usable_ace(self.player)
        return encode_state(self.player.calculate_hand_value(), player_has_usable_ace, self.visible_cards)

    def _has_usable_ace(self, player):
        """
        检查玩家是否有可用的A（计为11点而不爆牌）

        参数:
            player: 要检查的玩家

        返回:
            布尔值，表示是否有可用的A
        """
        # 计算没有A时的点数
        non_ace_value = sum(10 if card.rank in ['J', 'Q', 'K'] else
                            (int(card.rank) if card.rank not in ['A'] else 0)
                            for card in player.hand)

        # 计算有多少张A
        num_aces = sum(1 for card in player.hand if card.rank == 'A')

        # 检查是否有A可以计为11点而不爆牌
        for i in range(num_aces):
            if non_ace_value + 11 + (num_aces - 1) <= 21:
                return True
            non_ace_value += 1

        return False

    def step(self, action):
        """
        执行一步动作，返回(next_state, reward, done, info)

        参数:
            action: 0表示要牌，1表示停牌

        返回:
            next_state: 下一个状态向量
            reward: 获得的奖励
            done: 是否游戏结束
            info: 附加信息（空字典）
        """
        if self.done:
            return encode_state(self.player.calculate_hand_value(),
                                self._has_usable_ace(self.player),
                                self.visible_cards), self.reward, self.done, {}

        action_str = 'hit' if action == 0 else 'stand'

        if action_str == 'hit':
            card = self.deck.deal()
            self.player.add_card(card)

            # 检查是否爆牌
            if self.player.calculate_hand_value() > 21:
                self.player.is_busted = True
                self.done = True
                self.reward = -1
        else:  # stand
            self.player.is_standing = True

            # AI对手行动
            for opponent in self.opponents:
                while not opponent.is_standing and not opponent.is_busted:
                    if opponent.calculate_hand_value() < 17:
                        card = self.deck.deal()
                        opponent.add_card(card)
                        self.visible_cards.append(card)  # 记录可见牌

                        if opponent.calculate_hand_value() > 21:
                            opponent.is_busted = True
                    else:
                        opponent.is_standing = True

            # 游戏结束，计算结果
            self.done = True

        if self.player.calculate_hand_value() <= 21:
            player_value = self.player.calculate_hand_value()

            # 检查所有对手是否都爆牌
            all_opponents_busted = all(
                op.calculate_hand_value() > 21 for op in self.opponents)

            if all_opponents_busted:  # 所有对手都爆牌
                self.reward = 1
            else:
                # 获取没爆牌的对手的最大点数
                valid_opponents = [
                    op for op in self.opponents if op.calculate_hand_value() <= 21]
                max_opponent_value = max(
                    [op.calculate_hand_value() for op in valid_opponents])

                if player_value > max_opponent_value:  # 玩家点数最高
                    self.reward = 1
                elif player_value == max_opponent_value:  # 平局
                    self.reward = 0
                else:  # 玩家输
                    self.reward = -1
        else:
            self.reward = -1

        # 构建下一个状态
        player_has_usable_ace = self._has_usable_ace(self.player)
        next_state = encode_state(self.player.calculate_hand_value(
        ), player_has_usable_ace, self.visible_cards)

        return next_state, self.reward, self.done, {}
