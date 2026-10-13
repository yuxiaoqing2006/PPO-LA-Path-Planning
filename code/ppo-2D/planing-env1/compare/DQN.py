import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import torch
import torch.nn as nn
import torch.optim as optim
import os
from scipy.ndimage import gaussian_filter1d
import random
from collections import deque

# ===================== 全局参数设置 =====================
# 环境参数（和你的地图完全匹配）
MAP_SIZE = 20
START = (1, 1)
GOAL = (18, 18)
UNKNOWN_RADIUS = 3
# DQN算法参数
EPISODES = 2500
STEPS_PER_EPISODE = 200
HIDDEN_SIZE = 64
LR = 1e-4
GAMMA = 0.99
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64
TARGET_UPDATE_INTERVAL = 100
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.999
MAX_GRAD_NORM = 0.3
# 绘图与保存参数
PLOT_INTERVAL = 100
SAVE_DIR = "./dqn_fixed_map_planning"
os.makedirs(SAVE_DIR, exist_ok=True)
# 设备自动适配
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ===================== 固定地图环境类 =====================
class PathPlanningEnv:
    def __init__(self):
        self.map_size = MAP_SIZE
        self.start = np.array(START, dtype=np.int32)
        self.goal = np.array(GOAL, dtype=np.int32)
        self.unknown_radius = UNKNOWN_RADIUS
        self.max_dist = np.linalg.norm(self.goal - self.start)
        self.reset()

    def generate_map(self):
        """和你提供的地图100%匹配的固定障碍物"""
        grid_map = np.zeros((self.map_size, self.map_size), dtype=np.int32)
        # 边界障碍物
        grid_map[0, :] = 1
        grid_map[-1, :] = 1
        grid_map[:, 0] = 1
        grid_map[:, -1] = 1
        # 图中所有内部障碍物坐标 (行i, 列j)
        fixed_obstacles = [
            (17,10),
            (16,13), (16,14),
            (15,6), (15,18),
            (14,2),
            (13,1), (13,8), (13,11), (13,15), (13,17),
            (11,5),
            (10,1), (10,5), (10,11), (10,14), (10,16), (10,17), (10,18),
            (9,1),
            (8,1), (8,4), (8,11), (8,14),
            (7,1),
            (6,4),
            (5,5), (5,6), (5,7), (5,10), (5,12), (5,13), (5,15),
            (4,5), (4,6),
            (3,3), (3,13), (3,14),
            (2,5), (2,15), (2,18),
            (1,15)
        ]
        for (i, j) in fixed_obstacles:
            grid_map[i, j] = 1
        return grid_map

    def get_local_observation(self):
        x, y = self.pos
        local_obs = np.ones((2*self.unknown_radius+1, 2*self.unknown_radius+1), dtype=np.float32) * 0.5
        for dx in range(-self.unknown_radius, self.unknown_radius+1):
            for dy in range(-self.unknown_radius, self.unknown_radius+1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    local_obs[dx+self.unknown_radius, dy+self.unknown_radius] = self.grid_map[nx, ny]
        local_flat = local_obs.flatten()
        norm_pos = self.pos / self.map_size
        relative_goal = (self.goal - self.pos) / self.map_size
        state = np.concatenate([local_flat, norm_pos, relative_goal], dtype=np.float32)
        return state

    def reset(self):
        self.grid_map = self.generate_map()
        self.pos = self.start.copy()
        self.path = [self.pos.copy()]
        self.collision = False
        self.success = False
        return self.get_local_observation()

    def step(self, action):
        dx, dy = [(-1,0), (0,1), (1,0), (0,-1)][action]
        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy
        if 1 <= new_x < self.map_size-1 and 1 <= new_y < self.map_size-1:
            self.pos[0] = new_x
            self.pos[1] = new_y
        self.path.append(self.pos.copy())
        reward = 0.0
        done = False
        if self.grid_map[self.pos[0], self.pos[1]] == 1:
            self.collision = True
            reward = -1.0
            done = True
        elif np.linalg.norm(self.pos - self.goal) <= 0.0:
            self.success = True
            reward = 1.0
            done = True
        else:
            prev_dist = np.linalg.norm(self.path[-2] - self.goal)
            curr_dist = np.linalg.norm(self.pos - self.goal)
            reward = (prev_dist - curr_dist) * 0.1 - 0.01
        return self.get_local_observation(), reward, done

# ===================== DQN网络 =====================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, output_dim)
        # 权重初始化
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

# ===================== DQN智能体 =====================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eps = EPS_START
        # 网络
        self.policy_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # 优化器与经验回放
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR, eps=1e-5)
        self.memory = deque(maxlen=MEMORY_CAPACITY)

    def choose_action(self, state):
        # epsilon-greedy策略
        if random.random() < self.eps:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return
        # 采样批次
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)

        # 当前Q值
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        # 目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * GAMMA * next_q_values

        # 损失与更新
        loss = nn.MSELoss()(q_values.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()

        # 衰减探索率
        if self.eps > EPS_END:
            self.eps *= EPS_DECAY

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ===================== 评价指标计算器 =====================
class MetricsCalculator:
    def __init__(self):
        self.metrics = {
            "path_length": [], "reward": [], "collision_rate": [],
            "success_rate": [], "smoothness": []
        }
        self.episode_buffer = []

    def calculate_path_length(self, path):
        length = 0.0
        for i in range(1, len(path)):
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
        return length

    def calculate_smoothness(self, path):
        if len(path) < 3:
            return 0.0
        smoothness = 0.0
        for i in range(1, len(path)-1):
            vec1 = np.array(path[i]) - np.array(path[i-1])
            vec2 = np.array(path[i+1]) - np.array(path[i])
            norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            if norm > 0:
                cos_theta = np.clip(np.dot(vec1, vec2) / norm, -1.0, 1.0)
                angle = np.arccos(cos_theta)
                smoothness += np.abs(angle)
        return smoothness

    def update(self, path, reward, collision, success):
        length = self.calculate_path_length(path)
        smoothness = self.calculate_smoothness(path)
        self.episode_buffer.append([length, reward, float(collision), float(success), smoothness])

    def get_average(self):
        if not self.episode_buffer:
            return [0.0]*5
        arr = np.array(self.episode_buffer)
        avg_len = np.mean(arr[:,0])
        avg_rew = np.mean(arr[:,1])
        collision_rate = np.mean(arr[:,2])
        success_rate = np.mean(arr[:,3])
        avg_smooth = np.mean(arr[:,4])
        self.metrics["path_length"].append(avg_len)
        self.metrics["reward"].append(avg_rew)
        self.metrics["collision_rate"].append(collision_rate)
        self.metrics["success_rate"].append(success_rate)
        self.metrics["smoothness"].append(avg_smooth)
        self.episode_buffer = []
        return avg_len, avg_rew, collision_rate, success_rate, avg_smooth

# ===================== 可视化工具（全英文，零乱码） =====================
def plot_path(env, episode, save_path):
    plt.figure(figsize=(8, 8), dpi=100)
    ax = plt.gca()
    # 绘制障碍物
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            if env.grid_map[i, j] == 1:
                ax.add_patch(Rectangle((j, i), 1, 1, color='#2c3e50', alpha=0.8))
    # 绘制起点、终点、路径
    ax.add_patch(Rectangle((START[1], START[0]), 1, 1, color='#27ae60', label='Start'))
    ax.add_patch(Circle((GOAL[1]+0.5, GOAL[0]+0.5), 0.4, color='#e74c3c', label='Goal'))
    path = np.array(env.path)
    plt.plot(path[:,1]+0.5, path[:,0]+0.5, color='#3498db', linewidth=2.5, label='Planned Path')
    # 图表设置（全英文）
    plt.xlim(0, MAP_SIZE)
    plt.ylim(0, MAP_SIZE)
    ax.set_aspect('equal')
    plt.title(f"Path Planning - Episode {episode}", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_training_curve(metrics):
    plt.figure(figsize=(15, 10), dpi=100)
    index = np.arange(1, len(metrics["path_length"])+1) * PLOT_INTERVAL
    plt.subplot(2, 3, 1)
    plt.plot(index, metrics["path_length"], color='#3498db', linewidth=2)
    plt.title("Average Path Length", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    plt.plot(index, metrics["reward"], color='#e74c3c', linewidth=2)
    plt.title("Average Reward", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    plt.plot(index, metrics["collision_rate"], color='#2c3e50', linewidth=2)
    plt.title("Collision Rate", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 4)
    plt.plot(index, metrics["success_rate"], color='#27ae60', linewidth=2)
    plt.title("Success Rate", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    smooth_data = gaussian_filter1d(metrics["smoothness"], sigma=2)
    plt.plot(index, smooth_data, color='#f39c12', linewidth=2)
    plt.title("Path Smoothness", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig(f"{SAVE_DIR}/training_metrics.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✅ Training metrics saved to {SAVE_DIR}/training_metrics.png")

# ===================== 主训练流程 =====================
def main():
    local_obs_dim = (2*UNKNOWN_RADIUS+1)**2
    state_dim = local_obs_dim + 4
    action_dim = 4
    env = PathPlanningEnv()
    agent = DQNAgent(state_dim, action_dim)
    metrics = MetricsCalculator()
    print("🚀 Start training DQN path planning...")
    print(f"📊 Save path plot every {PLOT_INTERVAL} episodes\n")

    for episode in range(1, EPISODES+1):
        state = env.reset()
        total_reward = 0.0
        done = False
        for _ in range(STEPS_PER_EPISODE):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            agent.update()
            if done:
                break
        # 更新目标网络
        if episode % TARGET_UPDATE_INTERVAL == 0:
            agent.update_target_net()

        metrics.update(env.path, total_reward, env.collision, env.success)

        if episode % PLOT_INTERVAL == 0:
            avg_len, avg_rew, col_rate, suc_rate, avg_smooth = metrics.get_average()
            print(f"📌 Episode: {episode:4d} | Avg Length: {avg_len:6.2f} | Avg Reward: {avg_rew:7.2f} | Collision Rate: {col_rate:6.2%} | Success Rate: {suc_rate:6.2%} | Smoothness: {avg_smooth:.2f}")
            plot_path(env, episode, f"{SAVE_DIR}/path_episode_{episode}.png")

    plot_training_curve(metrics.metrics)
    print(f"\n🎉 Training finished! All results saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()