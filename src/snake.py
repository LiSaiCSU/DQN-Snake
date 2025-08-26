import pygame
import random
import numpy as np
import cv2
from collections import deque

# 游戏参数
GRID_SIZE = 10
BLOCK_SIZE = 20
WIDTH = GRID_SIZE * BLOCK_SIZE
HEIGHT = GRID_SIZE * BLOCK_SIZE
SPEED = 40

# 方向
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]

# 动作空间: 0=前, 1=左, 2=右
ACTIONS = [0, 1, 2]


class SnakeGameGrid:
    def __init__(self):
        pygame.init()
        self.width = WIDTH
        self.height = HEIGHT
        self.display = pygame.Surface((self.width, self.height))
        pygame.display.set_caption("Snake 10x10 Final")
        self.clock = pygame.time.Clock()
        
        self.stacked_frames = None
        self.reset()

    def reset(self):
        """重置游戏并返回初始状态 (4帧叠加的图像)"""
        self.direction = RIGHT
        mid = GRID_SIZE // 2
        self.snake = [(mid, mid)]
        self.score = 0
        self.food = None
        self._place_food()
        self.done = False
        self.frame_iteration = 0

        self._update_ui()
        initial_frame = self._get_preprocessed_frame()
        self.stacked_frames = deque([initial_frame] * 4, maxlen=4)
        
        return np.stack(self.stacked_frames, axis=2)

    def _place_food(self):
        """随机生成食物"""
        positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE) if (x, y) not in self.snake]
        if not positions:
            self.done = True
        else:
            self.food = random.choice(positions)

    def step(self, action):
        """执行一步操作并返回 (next_state, reward, done, score)"""
        self.frame_iteration += 1
        
        if self.done:
            state = np.stack(self.stacked_frames, axis=2)
            return state, 0, True, self.score

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0

        if self._is_collision():
            self.done = True
            reward = -10
        elif self.frame_iteration > 100 * (len(self.snake) + 1):
             self.done = True
             reward = -10
        else:
            if self.head == self.food:
                self.score += 1
                reward = 10
                self._place_food()
            else:
                self.snake.pop()
        
        self._update_ui()
        new_frame = self._get_preprocessed_frame()
        self.stacked_frames.append(new_frame)
        next_state = np.stack(self.stacked_frames, axis=2)

        self.clock.tick(SPEED)
        return next_state, reward, self.done, self.score

    def _get_preprocessed_frame(self):
        """获取当前屏幕画面并进行预处理（灰度化）"""
        img_rgb = pygame.surfarray.array3d(self.display)
        img_transposed = np.transpose(img_rgb, (1, 0, 2))
        img_gray = cv2.cvtColor(img_transposed, cv2.COLOR_RGB2GRAY)
        img_normalized = img_gray / 255.0
        return img_normalized

    def _move(self, action):
        """根据action更新蛇的移动方向和头部位置"""
        idx = DIRECTIONS.index(self.direction)
        if action == 1:
            idx = (idx - 1) % 4
        elif action == 2:
            idx = (idx + 1) % 4
        
        self.direction = DIRECTIONS[idx]
        x, y = self.snake[0]
        dx, dy = self.direction
        self.head = (x + dx, y + dy)

    def _is_collision(self):
        """检查是否发生碰撞"""
        x, y = self.head
        if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
            return True
        if self.head in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        """更新游戏画面，使用新的灰度值方案"""
        self.display.fill((0, 0, 0)) # 背景 (灰度值 0)
        
        # 食物 (最亮)
        if self.food:
            fx, fy = self.food
            food_rect = pygame.Rect(fx * BLOCK_SIZE, fy * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, (255, 255, 255), food_rect) # 灰度值 255

        # 绘制蛇，蛇头为210，身体逐次递减
        for i, (x, y) in enumerate(self.snake):
            # 计算当前节的灰度值
            value = max(0, 210 - i * 10) # 乘以一个更大的系数让渐变更明显
            color = (value, value, value)
            
            body_rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, color, body_rect)

# 主程序入口，用于测试环境是否正常工作
if __name__ == "__main__":
    game = SnakeGameGrid()
    display = pygame.display.set_mode((WIDTH, HEIGHT))
    
    while True:
        action = random.choice(ACTIONS)
        state, reward, done, score = game.step(action)
        
        vis_frame = state[:, :, -1] * 255
        vis_frame_rgb = cv2.cvtColor(vis_frame.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        surf = pygame.surfarray.make_surface(np.transpose(vis_frame_rgb, (1, 0, 2)))
        display.blit(surf, (0, 0))
        
        pygame.display.set_caption(f"Score: {score}")
        pygame.display.flip()

        if done:
            print(f"Game over! Score: {score}")
            game.reset()
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()