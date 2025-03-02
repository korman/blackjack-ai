import esper
import random
from dataclasses import dataclass;

# 系统实现
@dataclass
class PlayOrder:  # 存储玩家实体ID的出牌顺序
    order: list[int]

@dataclass
class CurrentPlayer:  # 标记当前回合玩家
    index: int = 0

class TurnSystem(esper.Processor):
    def process(self):
        # 获取游戏状态组件
        play_order = esper.get_component(PlayOrder)[0][1]
        current_ent, current = esper.get_component(CurrentPlayer)[0]

        # 获取当前玩家手牌
        player_ent = play_order.order[current.index]
        hand = esper.component_for_entity(player_ent, Hand)

        if not hand.cards:
            return  # 玩家已无手牌

        # 模拟出牌逻辑（此处可替换为AI决策）
        if hand.cards:
            # 随机选择一张牌打出
            selected = random.choice(hand.cards)
            hand.cards.remove(selected)

            # 更新已打出牌堆
            played_ent = self.world.get_component(PlayedCards)[0][0]
            played = self.world.component_for_entity(played_ent, PlayedCards)
            played.cards.append(selected)

            print(f"玩家{player_ent} 打出: {selected}")

        # 切换到下个玩家
        current.index = (current.index + 1) % len(play_order.order)

        # 检查是否全部出完
        all_empty = all(
            len(self.world.component_for_entity(e, Hand).cards) == 0
            for e in play_order.order
        )

        if all_empty:
            print("所有玩家出牌完毕！")
            self.world.delete_entity(current_ent)  # 结束游戏
