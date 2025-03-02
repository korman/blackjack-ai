import random
import time
from typing import List, Optional

# 定义牌的花色和点数
SUITS = ['♠', '♥', '♦', '♣']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
          '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11}


class Card:
    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank
        self.value = VALUES[rank]

    def __str__(self):
        return f"{self.suit}{self.rank}"


class Deck:
    def __init__(self):
        self.cards = [Card(suit, rank) for suit in SUITS for rank in RANKS]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self) -> Optional[Card]:
        if not self.cards:
            return None
        return self.cards.pop()


class Player:
    def __init__(self, name: str, is_ai: bool = False):
        self.name = name
        self.hand: List[Card] = []
        self.is_ai = is_ai
        self.is_standing = False
        self.is_busted = False

    def add_card(self, card: Card):
        self.hand.append(card)
        # 检查是否爆牌
        if self.calculate_hand_value() > 21:
            self.is_busted = True
            self.is_standing = True

    def calculate_hand_value(self) -> int:
        value = sum(card.value for card in self.hand)
        # 处理A的特殊情况
        num_aces = sum(1 for card in self.hand if card.rank == 'A')
        # 如果有A且总点数超过21，将A的值从11改为1
        while value > 21 and num_aces:
            value -= 10
            num_aces -= 1
        return value

    def decide_action(self) -> str:
        if self.is_ai:
            # AI策略：小于17要牌，否则停牌
            if self.calculate_hand_value() < 17:
                return 'hit'
            else:
                return 'stand'
        else:
            # 人类玩家通过输入决定
            while True:
                action = input(
                    f"{self.name}, 你的手牌是 {self.show_hand()}, 点数是 {self.calculate_hand_value()}. 要牌(h)还是停牌(s)? ").lower()
                if action in ['h', 'hit']:
                    return 'hit'
                elif action in ['s', 'stand']:
                    return 'stand'
                else:
                    print("无效的输入，请输入 'h' 要牌或 's' 停牌。")

    def show_hand(self) -> str:
        return " ".join(str(card) for card in self.hand)


class BlackjackGame:
    def __init__(self):
        self.deck = Deck()
        self.human_player = Player("玩家")
        self.ai_player1 = Player("AI玩家1", is_ai=True)
        self.ai_player2 = Player("AI玩家2", is_ai=True)
        self.players = [self.human_player, self.ai_player1, self.ai_player2]

    def start_game(self):
        # 初始发牌，每人两张
        for _ in range(2):
            for player in self.players:
                player.add_card(self.deck.deal())

        # 游戏主循环
        game_over = False
        while not game_over:
            # 所有玩家轮流行动
            for player in self.players:
                if player.is_standing:
                    continue

                print(f"\n{player.name}的回合:")
                if player.is_ai:
                    print(
                        f"{player.name}的手牌是 {player.show_hand()}, 点数是 {player.calculate_hand_value()}")
                    time.sleep(1)  # 添加一点延迟，使游戏流程更易于跟踪

                action = player.decide_action()

                if action == 'hit':
                    card = self.deck.deal()
                    player.add_card(card)
                    print(f"{player.name} 要了一张牌: {card}")
                    if player.is_busted:
                        print(
                            f"{player.name} 爆牌了! 点数: {player.calculate_hand_value()}")
                else:  # stand
                    player.is_standing = True
                    print(f"{player.name} 选择停牌。点数: {player.calculate_hand_value()}")

            # 检查游戏是否结束
            if all(player.is_standing for player in self.players):
                game_over = True

        self.determine_winner()

    def determine_winner(self):
        print("\n===== 游戏结果 =====")

        # 显示所有玩家的最终手牌
        for player in self.players:
            status = "爆牌" if player.is_busted else "有效"
            print(
                f"{player.name}: {player.show_hand()} - 点数: {player.calculate_hand_value()} ({status})")

        # 找出没有爆牌的玩家
        valid_players = [p for p in self.players if not p.is_busted]

        if not valid_players:
            print("所有玩家都爆牌了！没有赢家。")
            return

        # 找出点数最高的玩家
        max_value = max(p.calculate_hand_value() for p in valid_players)
        winners = [p for p in valid_players if p.calculate_hand_value()
                   == max_value]

        if len(winners) == 1:
            print(f"\n赢家是 {winners[0].name}，点数: {max_value}")
        else:
            winner_names = ", ".join(w.name for w in winners)
            print(f"\n平局! 赢家是: {winner_names}，点数: {max_value}")


def play_blackjack():
    print("欢迎来到21点游戏!")
    print("规则: 尽可能接近21点但不超过。A可以算1点或11点。J、Q、K都算10点。")

    while True:
        game = BlackjackGame()
        game.start_game()

        play_again = input("\n想再玩一局吗? (y/n): ").lower()
        if play_again != 'y':
            print("谢谢游玩，再见!")
            break


if __name__ == "__main__":
    play_blackjack()
