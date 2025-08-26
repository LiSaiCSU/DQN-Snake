import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import numpy as np
from model import create_modified_resnet18
from memory import ReplayMemory, Transition


class DQNAgent:
    
    def __init__(self,
                 policy_network, # <--- 新增
                 target_network, # <--- 新增
                 n_actions,
                 learning_rate=0.0001,
                 gamma=0.99,
                 epsilon_start=0.9,
                 epsilon_end=0.05,
                 epsilon_decay=10000,
                 memory_capacity=30000,
                 batch_size=32,
                 target_update_freq=1000):  # <--- 新增参数，设置默认值

        # (移除了 input_shape 参数，因为模型是外部传入的)

        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # ε-greedy 策略的相关参数
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 选择设备 (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建策略网络和目标网络
        # 注意：您需要先完成 CNN_QNet 类的具体实现
        self.policy_net = create_modified_resnet18().to(self.device)
        self.target_net = create_modified_resnet18().to(self.device)

        # 将策略网络的权重复制到目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # 目标网络只用于评估，不进行训练
        self.target_net.eval()

        # 定义优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # 实例化经验回放池
        self.memory = ReplayMemory(memory_capacity)
        
        # 用于epsilon衰减的步数计数器
        self.steps_done = 0

    def current_epsilon(self):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                    math.exp(-1. * self.steps_done / self.epsilon_decay)
        return epsilon

    def select_action(self, state):
        """根据当前状态和ε-greedy策略选择一个动作"""
        # 计算当前的epsilon值，它会随着训练步数增加而衰减
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        
        # 以 epsilon 的概率进行探索（随机选择动作）
        if random.random() < epsilon:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        else:
            # 以 1-epsilon 的概率进行利用（选择Q值最高的动作）
            with torch.no_grad():
                # state 是 (H, W, C) 的 numpy 数组，需要转换为 (B, C, H, W) 的 tensor
                # B=1, C=4, H=200, W=200
                state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
                state_tensor = state_tensor.permute(0, 3, 1, 2) # 转换为 (B, C, H, W)
                
                # 使用策略网络预测Q值
                q_values = self.policy_net(state_tensor)
                # 选择Q值最高的动作
                action = q_values.max(1)[1].view(1, 1)
                return action

    def optimize_model(self):
        """从经验回放池中采样并进行一次模型优化"""
        # 如果内存中的经验数量不足以采样一个batch，则不进行学习
        if len(self.memory) < self.batch_size:
            return

        # 1. 从内存中采样一批转换
        transitions = self.memory.sample(self.batch_size)
        # 将一批转换转换为一个转换对象，其中每个字段都是一个包含所有样本的批次
        batch = Transition(*zip(*transitions))

        # 2. 准备训练数据
        # 将numpy数组转换为tensor，并调整维度
        state_batch = np.array([s for s in batch.state])
        state_batch = torch.from_numpy(state_batch).float().to(self.device).permute(0, 3, 1, 2)
        
        next_state_batch = np.array([s for s in batch.next_state])
        next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device).permute(0, 3, 1, 2)

        action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor(batch.reward, device=self.device).float()
        done_batch = torch.tensor(batch.done, device=self.device)

        # 3. 计算预测Q值 (Q(s, a))
        # 使用策略网络计算当前状态下，所有动作的Q值
        all_q_values = self.policy_net(state_batch)
        # 从中选出实际执行过的动作的Q值
        predicted_q_values = all_q_values.gather(1, action_batch)

        # 4. 计算目标Q值 (r + γ * max Q(s', a'))
        # 使用目标网络计算下一个状态的最大Q值
        with torch.no_grad():
            next_state_q_values = self.target_net(next_state_batch).max(1)[0]
        
        # 如果下一个状态是终止状态，则其未来价值为0
        next_state_q_values[done_batch] = 0.0
        
        # 计算最终的目标Q值
        target_q_values = reward_batch + (self.gamma * next_state_q_values)

        # 5. 计算损失
        # 使用 Huber loss，它比 MSELoss 更稳健
        loss = F.smooth_l1_loss(predicted_q_values, target_q_values.unsqueeze(1))
        
        # 6. 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        # (可选) 梯度裁剪，防止梯度爆炸
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_net(self):
        """定期将策略网络的权重复制到目标网络"""
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())