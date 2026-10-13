import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os
from scipy.ndimage import gaussian_filter1d

# ===================== 全局参数设置 =====================
# 环境参数（和你的地图完全匹配）
MAP_SIZE = 20
START = (1, 1)
GOAL = (18, 18)
UNKNOWN_RADIUS = 3

# PPO算法稳定参数
EPISODES = 2500
STEPS_PER_EPISODE = 200
HIDDEN_SIZE = 64
LR = 1e-4
GAMMA = 0.99
EPS_CLIP = 0.2
UPDATE_EPOCHS = 3
BATCH_SIZE = 64
MAX_GRAD_NORM = 0.3

# 绘图与保存参数
PLOT_INTERVAL = 100
SAVE_DIR = "./ppo_fixed_map_planning"
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
         (17,10),(17,11),(17,12),
          (5,14),
            (15,6),
            (3,10),(6,9),
            (13,11), (14,11), 
            (16,16),
            (10,15), (11,15), (12,15),(13,15), 
            (9,1),(8,1),(7,1),
            (10,4),(10,5), (10,6), 
            (6,4),(5,4),
            (8,10), (8,11), (8,12), 
            #(10,10), (10,11), (10,12), (4,5), (4,6),(8,9), (7,9), (10,9),
            (15,3), (15,4), (15,5),
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

# ===================== PPO网络 =====================
class PPO(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPO, self).__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.policy_head = nn.Linear(HIDDEN_SIZE, output_dim)
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.policy_head.bias, 0.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.value_head(x)
        logits = torch.clamp(self.policy_head(x), min=-20.0, max=20.0)
        return logits, value

    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_log_probs, values, entropy

# ===================== PPO训练器 =====================
class PPOTrainer:
    def __init__(self, input_dim, action_dim):
        self.model = PPO(input_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR, eps=1e-5)
        self.memory = []

    def store_transition(self, transition):
        self.memory.append(transition)

    def compute_gae(self, rewards, dones, values, next_value):
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
        values = torch.tensor(values, dtype=torch.float32).to(DEVICE)
        next_value = torch.tensor(next_value, dtype=torch.float32).to(DEVICE)

        values = torch.cat([values, next_value.unsqueeze(0)])
        gae = 0.0
        advantages = torch.zeros_like(rewards).to(DEVICE)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * 0.95 * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages.cpu().numpy(), returns.cpu().numpy()

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return

        states = torch.FloatTensor([t[0] for t in self.memory]).to(DEVICE)
        actions = torch.LongTensor([t[1] for t in self.memory]).to(DEVICE)
        old_log_probs = torch.FloatTensor([t[2] for t in self.memory]).to(DEVICE)
        rewards = [t[3] for t in self.memory]
        dones = [t[4] for t in self.memory]
        values = [t[5] for t in self.memory]

        if torch.isnan(states).any() or torch.isnan(actions).any() or torch.isnan(old_log_probs).any():
            print("⚠️  NaN detected, skip this update")
            self.memory.clear()
            return

        with torch.no_grad():
            _, next_value = self.model(states[-1:])
        advantages, returns = self.compute_gae(rewards, dones, values, next_value.item())

        advantages = torch.FloatTensor(advantages).to(DEVICE)
        returns = torch.FloatTensor(returns).to(DEVICE).unsqueeze(1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, advantages, returns)
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        for _ in range(UPDATE_EPOCHS):
            for state, action, old_log_prob, adv, ret in loader:
                new_log_prob, value, entropy = self.model.evaluate(state, action)

                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(value, ret)
                entropy_loss = -0.01 * entropy.mean()
                loss = policy_loss + 0.5 * value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=MAX_GRAD_NORM)
                self.optimizer.step()

        self.memory.clear()

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
    agent = PPOTrainer(state_dim, action_dim)
    metrics = MetricsCalculator()

    print("🚀 Start training PPO path planning...")
    print(f"📊 Save path plot every {PLOT_INTERVAL} episodes\n")

    for episode in range(1, EPISODES+1):
        state = env.reset()
        total_reward = 0.0
        done = False

        for _ in range(STEPS_PER_EPISODE):
            action, log_prob, value = agent.model.get_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            agent.store_transition([state, action, log_prob, reward, done, value])
            state = next_state
            if done:
                break

        metrics.update(env.path, total_reward, env.collision, env.success)

        if episode % PLOT_INTERVAL == 0:
            avg_len, avg_rew, col_rate, suc_rate, avg_smooth = metrics.get_average()
            print(f"📌 Episode: {episode:4d} | Avg Length: {avg_len:6.2f} | Avg Reward: {avg_rew:7.2f} | Collision Rate: {col_rate:6.2%} | Success Rate: {suc_rate:6.2%} | Smoothness: {avg_smooth:.2f}")
            plot_path(env, episode, f"{SAVE_DIR}/path_episode_{episode}.png")

        agent.update()

    plot_training_curve(metrics.metrics)
    print(f"\n🎉 Training finished! All results saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()