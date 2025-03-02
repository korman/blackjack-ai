# 21点游戏AI训练版

一个基于深度强化学习的21点(Blackjack)游戏AI项目，使用深度Q网络(DQN)训练智能体学习最优游戏策略。

## 功能特点

- 🧠 **AI智能体**：使用深度Q学习算法训练的智能体，能够学习复杂的21点策略
- 🎮 **人机对战**：与训练好的AI进行游戏对战
- 📊 **牌面分析**：AI能够分析已出现的牌，调整决策策略
- 🔄 **自适应学习**：随着训练的深入，AI能够逐步提升游戏水平

## 简化版说明

**注意**：当前实现为简化版21点游戏，专注于基本的要牌和停牌决策，不包含以下高级功能：

- 分牌(Split)
- 双倍下注(Double Down)
- 保险(Insurance)
- 投降(Surrender)

这种简化设计使AI能够聚焦于核心策略学习。未来版本计划逐步添加这些高级功能，扩展游戏复杂度和策略深度。

## 项目结构

```bash
blackjack-ai/
│
├── blackjack_env.py   # 21点游戏环境实现
├── dqn_agent.py       # 深度Q网络智能体实现
├── play_with_ai.py    # 人机对战界面
├── train.py           # AI训练脚本
├── models/            # 保存训练好的模型
└── requirements.txt   # 项目依赖
```

## 安装指南

1. 克隆仓库：

   ```bash
   git clone https://github.com/korman/blackjack-ai.git
   cd blackjack-ai
   ```

2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

## 使用说明

### 训练AI模型

```bash
python train.py
```

训练完成后，模型会保存在`models/`目录中。

### 与AI对战

```bash
python play_with_ai.py
```

跟随屏幕提示进行游戏操作：

- `h`：要牌
- `s`：停牌

## 技术细节

### 状态表示（15维向量）

AI通过15维向量理解游戏状态：

- 2个玩家状态特征：手牌点数和是否有可用的A
- 13个牌计数特征：记录已知牌的分布情况

### 智能体架构

- **网络结构**：多层感知机(MLP)，输入层15个神经元，隐藏层，输出层2个神经元（对应要牌和停牌动作）
- **学习机制**：经验回放(Experience Replay)和双网络架构
- **决策策略**：ε-贪婪策略，在训练时进行探索，游戏时选择最优动作

## 游戏规则

标准21点规则：

- 目标是使手牌点数尽可能接近21点但不超过
- 数字牌按面值计分，J/Q/K为10点，A可记为1点或11点
- 玩家可以选择要牌(hit)或停牌(stand)
- 爆牌(超过21点)自动失败
- 庄家爆牌时，未爆牌的玩家获胜

## 许可证

[MIT License](chrome-extension://dhoenijjpgpeimemopealfcbiecgceod/LICENSE)