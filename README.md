# Blackjack AI

[简体中文](https://github.com/korman/blackjack-ai/blob/master/README_zhCN.md) | English

An AI project for Blackjack based on deep reinforcement learning, using Deep Q-Networks (DQN) to train agents that learn optimal game strategies.

## Key Features

- 🧠 **AI Agent**: Intelligent agent trained using Deep Q-Learning algorithm to master complex Blackjack strategies
- 🎮 **Human vs. AI**: Play against the trained AI opponent
- 📊 **Card Analysis**: AI analyzes cards that have appeared to adjust decision strategies
- 🔄 **Adaptive Learning**: AI continuously improves its gameplay through extensive training

## Simplified Version Note

**Note**: The current implementation is a simplified version of Blackjack, focusing on basic hit and stand decisions, excluding the following advanced features:

- Splitting pairs
- Double Down
- Insurance
- Surrender

This simplified design allows the AI to focus on learning core strategies. Future versions plan to gradually add these advanced features to expand game complexity and strategic depth.

## Project Structure

```bash
blackjack-ai/
│
├── blackjack_env.py   # Blackjack game environment implementation
├── dqn_agent.py       # Deep Q-Network agent implementation
├── play_with_ai.py    # Human vs. AI interface
├── train.py           # AI training script
├── models/            # Saved trained models
└── requirements.txt   # Project dependencies
```

## Installation Guide

1. Clone the repository:

   ```bash
   git clone https://github.com/korman/blackjack-ai.git
   cd blackjack-ai
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions

### Training the AI Model

```bash
python train.py
```

After training, the model will be saved in the `models/` directory.

### Playing Against the AI

```bash
python play_with_ai.py
```

Follow the on-screen prompts to play:

- `h`: Hit (request another card)
- `s`: Stand (end your turn)

## Technical Details

### State Representation (15-dimensional vector)

The AI understands the game state through a 15-dimensional vector:

- 2 player state features: hand value and availability of usable Ace
- 13 card counting features: tracking the distribution of known cards

### Agent Architecture

- **Network Structure**: Multi-layer Perceptron (MLP) with 15 neurons in the input layer, hidden layers, and 2 neurons in the output layer (corresponding to hit and stand actions)
- **Learning Mechanism**: Experience Replay and Double Network Architecture
- **Decision Strategy**: ε-greedy policy, exploring during training and selecting optimal actions during gameplay

## Game Rules

Standard Blackjack rules:

- The goal is to get a hand value as close to 21 as possible without exceeding it
- Number cards are worth their face value, J/Q/K are worth 10 points, and A can be worth 1 or 11 points
- Players can choose to hit (draw a card) or stand (end their turn)
- Busting (exceeding 21 points) results in automatic loss
- When the dealer busts, all non-busted players win

## License

[MIT License](https://github.com/korman/blackjack-ai/blob/master/LICENSE)

------

*This project was created for educational purposes to learn about deep reinforcement learning and AI applications.*