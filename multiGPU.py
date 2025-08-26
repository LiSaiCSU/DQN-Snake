import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

# 从您的文件中导入所有必要的类
from game import SnakeGameGrid, ACTIONS
from agent import DQNAgent
from model import create_modified_resnet18
from memory import ReplayMemory # 确保 memory.py 中的类也被导入

# --- 1. 超参数与配置 ---
# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 训练参数
NUM_EPISODES = 10000

# ----------> 改动 1: 增大批量和学习率 <----------
# 利用多GPU优势，使用更大的批量和相应的学习率
BATCH_SIZE = 1024          # (原为256, 扩大4倍以匹配4张GPU)
LEARNING_RATE = 0.0005     # (原为0.00025, 适当提高)
# ---------------------------------------------

GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 20000
TARGET_UPDATE_FREQ = 1000
MEMORY_CAPACITY = 50000

# --- 2. 初始化 ---
env = SnakeGameGrid()
n_actions = len(ACTIONS)

# 实例化您的ResNet18 Q网络
policy_net = create_modified_resnet18(n_actions=n_actions) # 确保模型创建函数能接收动作数
target_net = create_modified_resnet18(n_actions=n_actions)

# ----------> 改动 2: 启用DataParallel <----------
# 检查是否有多个GPU可用，并使用nn.DataParallel包装模型
if torch.cuda.device_count() > 1:
  print(f"Let's use {torch.cuda.device_count()} GPUs!")
  policy_net = nn.DataParallel(policy_net)
  target_net = nn.DataParallel(target_net)
# ---------------------------------------------

# 将模型移动到设备
policy_net.to(DEVICE)
target_net.to(DEVICE)

# 实例化Agent
agent = DQNAgent(
    policy_network=policy_net, # <--- 改动 3: 传入包装后的模型
    target_network=target_net, # <--- 改动 3: 传入包装后的模型
    n_actions=n_actions,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    epsilon_start=EPSILON_START,
    epsilon_end=EPSILON_END,
    epsilon_decay=EPSILON_DECAY,
    memory_capacity=MEMORY_CAPACITY,
    batch_size=BATCH_SIZE
)

# 用于记录和可视化的变量
episode_scores = []
moving_avg_scores = []
scores_window = deque(maxlen=100)

# --- 3. 主训练循环 ---
print("Starting Training...")
for episode in range(NUM_EPISODES):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, score = env.step(action.item())
        
        reward_tensor = torch.tensor([reward], device=DEVICE)
        done_tensor = torch.tensor([done], device=DEVICE)
        
        agent.memory.push(state, action, reward_tensor, next_state, done_tensor)
        state = next_state
        agent.optimize_model()
        
        if agent.steps_done % TARGET_UPDATE_FREQ == 0:
            agent.update_target_net()

    scores_window.append(env.score)
    episode_scores.append(env.score)
    moving_avg = np.mean(scores_window)
    moving_avg_scores.append(moving_avg)

    if (episode + 1) % 10 == 0:
        # 在 agent.py 中添加一个方法来获取当前的epsilon值
        # def current_epsilon(self):
        #     return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
        #            math.exp(-1. * self.steps_done / self.epsilon_decay)
        # print(f'Episode {episode+1}/{NUM_EPISODES} | Avg Score (Last 100): {moving_avg:.2f} | Epsilon: {agent.current_epsilon():.4f}')
        print(f'Episode {episode+1}/{NUM_EPISODES} | Avg Score (Last 100): {moving_avg:.2f}')


# --- 4. 训练结束，可视化结果 ---
print('Training finished.')

# ----------> 改动 4: 更新模型保存逻辑 <----------
if isinstance(agent.policy_net, nn.DataParallel):
    torch.save(agent.policy_net.module.state_dict(), 'snake_dqn_resnet18.pth')
else:
    torch.save(agent.policy_net.state_dict(), 'snake_dqn_resnet18.pth')
# ---------------------------------------------

print("Model Saved to snake_dqn_resnet18.pth")

# 绘制得分曲线
plt.figure(figsize=(12, 6))
plt.plot(episode_scores, label='Score per Episode')
plt.plot(moving_avg_scores, label='Moving Average (100 episodes)', linewidth=2)
plt.title('DQN Training Performance on Snake')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()