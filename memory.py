import random
from collections import deque, namedtuple

# 使用 namedtuple 来定义一次转换（transition）的数据结构
# 这样可以使代码更具可读性，例如用 experience.state 代替 experience[0]
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayMemory:
    """
    经验回放池，用于存储和随机采样智能体的经历。
    这是DQN成功的关键组件之一，用于打破数据相关性。 [cite: 27]
    """

    def __init__(self, capacity):
        """
        初始化 ReplayMemory。
        :param capacity: 整数，回放池的最大容量。
        """
        # 使用deque作为内部存储，它是一个双端队列
        # 当队列满时，再添加新元素会自动从另一端移除旧元素
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """
        将一次完整的“经历”存入回放池。
        参数 *args 会被打包成一个 Transition 对象。
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        从回放池中随机抽取一批经历用于训练。
        :param batch_size: 整数，需要抽取的批量大小。
        :return: 一个包含 Transition 对象的列表，长度为 batch_size。
        """
        # random.sample 用于从序列中进行不重复的随机抽样
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        返回当前存储在回放池中的经历数量。
        """
        return len(self.memory)


# --- 使用示例 ---
if __name__ == "__main__":
    # 假设容量为100
    memory = ReplayMemory(100)

    # 1. 添加经验 (push)
    # 假设 state 是一个 (200, 200, 4) 的数组
    # action, reward, done 是数字
    print("Pushing 10 dummy experiences into memory...")
    for i in range(10):
        # 创建一些假的 state 数据
        dummy_state = f"state_{i}"
        dummy_next_state = f"state_{i+1}"

        # 将经验存入
        memory.push(dummy_state, i, i * 0.1, dummy_next_state, False)

    # 2. 检查长度 (__len__)
    print(f"Current memory size: {len(memory)}")  # 应输出 10

    # 3. 随机采样 (sample)
    if len(memory) > 3:
        print("\nSampling a batch of 3 experiences:")
        batch = memory.sample(3)

        # 打印采样结果
        for i, experience in enumerate(batch):
            print(f"  Sample {i+1}:")
            print(f"    State: {experience.state}")
            print(f"    Action: {experience.action}")
            print(f"    Reward: {experience.reward}")
            print(f"    Next State: {experience.next_state}")
            print(f"    Done: {experience.done}")

    # 模拟内存满的情况
    print("\nPushing 100 more experiences to fill the memory...")
    for i in range(10, 110):
        memory.push(f"state_{i}", i, i * 0.1, f"state_{i+1}", False)

    print(f"Memory size after filling: {len(memory)}")  # 应输出 100 (最大容量)
