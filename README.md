🚕 Taxi-v3 Q-Learning Agent

A reinforcement learning agent trained on the Taxi-v3 environment
using Q-Learning with ε-greedy exploration decay — built with Gymnasium and NumPy.

📌 What it does

The agent learns to pick up a passenger and drop them at the correct
destination in a 5×5 grid world, by exploring the environment and
updating a Q-table over 10,000 episodes.

⚙️ Requirements
- Python 3.11+
- gymnasium
- numpy

pip install gymnasium numpy

🚀 Run

python main.py

- Training phase : 10,000 episodes, no rendering
- Evaluation phase: 5 episodes with visual rendering
