import torch
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

# 从您的文件中导入所有必要的类
from game import SnakeGameGrid, ACTIONS
from agent import DQNAgent
from model import create_modified_resnet18 # 假设您将修改后的ResNet18保存在这里

# --- 1. 超参数与配置 ---
# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 训练参数
NUM_EPISODES = 10000       # 总训练回合数
BATCH_SIZE = 256           # 每次从记忆中采样的数量
GAMMA = 0.99               # 折扣因子
EPSILON_START = 1.0        # 初始探索率
EPSILON_END = 0.01         # 最终探索率
EPSILON_DECAY = 20000      # 探索率衰减步数
TARGET_UPDATE_FREQ = 1000  # 目标网络更新频率 (按步数)
LEARNING_RATE = 0.00025    # 学习率
MEMORY_CAPACITY = 50000    # 经验回放池容量

# --- 2. 初始化 ---
env = SnakeGameGrid()
n_actions = len(ACTIONS) # 动作空间大小为3

# 实例化您的ResNet18 Q网络
# 注意：确保您的ResNet18_QNet类能正确接收n_actions
policy_net = create_modified_resnet18().to(DEVICE)
target_net = create_modified_resnet18().to(DEVICE)

# 实例化Agent
agent = DQNAgent(
    n_actions=n_actions,
    input_shape=(200, 200, 4),  # 你可以根据需要传递实际的输入shape
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
scores_window = deque(maxlen=100) # 存储最近100局的分数

# --- 3. 主训练循环 ---
print("Starting Training...")
for episode in range(NUM_EPISODES):
    state = env.reset() # 重置环境，获取初始状态
    done = False
    
    while not done:
        # 1. 智能体选择动作
        action = agent.select_action(state)
        
        # 2. 环境执行动作并返回结果
        # action是tensor，需要用.item()转为整数
        next_state, reward, done, score = env.step(action.item())
        
        # 将reward和done也转换为tensor
        reward_tensor = torch.tensor([reward], device=DEVICE)
        done_tensor = torch.tensor([done], device=DEVICE)
        
        # 3. 将经验存入回放池
        agent.memory.push(state, action, reward_tensor, next_state, done_tensor)
        
        # 4. 状态更新
        state = next_state
        
        # 5. 执行学习步骤 (Agent内部会判断是否达到batch_size)
        agent.optimize_model()
        
        # 6. 定期更新目标网络
        if agent.steps_done % TARGET_UPDATE_FREQ == 0:
            agent.update_target_net()

    # 一局游戏结束
    scores_window.append(env.score)
    episode_scores.append(env.score)
    moving_avg = np.mean(scores_window)
    moving_avg_scores.append(moving_avg)

    # 打印训练进度
    if (episode + 1) % 10 == 0:
        print(f'Episode {episode+1}/{NUM_EPISODES} | Avg Score (Last 100): {moving_avg:.2f} | Epsilon: {agent.current_epsilon():.4f}')

# --- 4. 训练结束，可视化结果 ---
print('Training finished.')

# 保存模型
torch.save(agent.policy_net.state_dict(), 'snake_dqn_resnet18.pth')
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