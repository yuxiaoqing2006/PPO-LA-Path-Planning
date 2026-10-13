#PPO
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import csv  # 新增csv模块

# ===================== 全局参数设置 =====================
# 环境参数（三维地图）
MAP_SIZE = 20
START = (1, 1, 1)
GOAL = (18, 18, 18)
UNKNOWN_RADIUS = 3

# PPO算法稳定参数
EPISODES = 2500
STEPS_PER_EPISODE = 300
HIDDEN_SIZE = 64
LR = 0.7e-4
GAMMA = 0.99
EPS_CLIP = 0.2
UPDATE_EPOCHS = 3
BATCH_SIZE = 64
MAX_GRAD_NORM = 0.3

# 绘图与保存参数
PLOT_INTERVAL = 100
SAVE_DIR = "./ppo_3d_map_planning"
os.makedirs(SAVE_DIR, exist_ok=True)

# 设备自动适配
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 障碍物参数
CUBE_OBSTACLES = [
    (10, 2, 3, 3),(2, 10, 9, 3),(9, 10, 11, 3),(17, 10, 2, 3),(5, 8, 2, 3),(2, 13, 17, 3),
    (17, 2, 3, 2),
    (12, 7, 2, 2),
    (12, 10, 7, 2),(14, 14, 7, 2),
    (8, 10, 8, 2),
    (17, 13, 10, 2), (10, 17, 10, 2),
    (2, 2, 13, 2),
    (12, 12, 17, 2),
    (17, 13, 15, 2), (5, 10, 15, 2),

]

# ===================== 三维地图环境类 =====================
class PathPlanningEnv3D:
    def __init__(self):
        self.map_size = MAP_SIZE
        self.start = np.array(START, dtype=np.int32)
        self.goal = np.array(GOAL, dtype=np.int32)
        self.unknown_radius = UNKNOWN_RADIUS
        self.max_dist = np.linalg.norm(self.goal - self.start)
        self.reset()

    def generate_map(self):
        grid_map = np.zeros((self.map_size, self.map_size, self.map_size), dtype=np.int32)
        grid_map[0, :, :] = 1
        grid_map[-1, :, :] = 1
        grid_map[:, 0, :] = 1
        grid_map[:, -1, :] = 1
        grid_map[:, :, 0] = 1
        grid_map[:, :, -1] = 1

        fixed_obstacles_2d = [
            (17,10), (16,13), (16,14), (15,6), (15,18), (14,2),
            (13,1), (13,8), (13,11), (13,15), (13,17), (11,5),
            (10,1), (10,5), (10,11), (10,14), (10,16), (10,17), (10,18),
            (9,1), (8,1), (8,4), (8,11), (8,14), (7,1), (6,4),
            (5,5), (5,6), (5,7), (5,10), (5,12), (5,13), (5,15),
            (4,5), (4,6), (3,3), (3,13), (3,14), (2,5), (2,15), (2,18), (1,15)
        ]
        
        fixed_obstacles_3d = []
        for (i, j) in fixed_obstacles_2d:
            for z in [2, 4, 6, 8, 10, 12, 14, 16]:
                fixed_obstacles_3d.append((i, j, z))
        
        for x in range(5, 15):
            for y in range(5, 15):
                fixed_obstacles_3d.append((x, y, 9))
        
        for (start_x, start_y, start_z, size) in CUBE_OBSTACLES:
            for x in range(start_x, start_x + size):
                for y in range(start_y, start_y + size):
                    for z in range(start_z, start_z + size):
                        fixed_obstacles_3d.append((x, y, z))
        
        for (i, j, k) in fixed_obstacles_3d:
            if 0 <= i < self.map_size and 0 <= j < self.map_size and 0 <= k < self.map_size:
                grid_map[i, j, k] = 1

        return grid_map

    def get_local_observation(self):
        x, y, z = self.pos
        local_obs = np.ones((2*self.unknown_radius+1, 
                            2*self.unknown_radius+1, 
                            2*self.unknown_radius+1), dtype=np.float32) * 0.5
        
        for dx in range(-self.unknown_radius, self.unknown_radius+1):
            for dy in range(-self.unknown_radius, self.unknown_radius+1):
                for dz in range(-self.unknown_radius, self.unknown_radius+1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size and 0 <= nz < self.map_size:
                        local_obs[dx+self.unknown_radius, 
                                 dy+self.unknown_radius, 
                                 dz+self.unknown_radius] = self.grid_map[nx, ny, nz]
                    
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
        dx, dy, dz = [(0,0,1), (0,0,-1), (0,1,0), (0,-1,0), (1,0,0), (-1,0,0)][action]
        
        new_x = np.clip(self.pos[0] + dx, 1, self.map_size-2)
        new_y = np.clip(self.pos[1] + dy, 1, self.map_size-2)
        new_z = np.clip(self.pos[2] + dz, 1, self.map_size-2)

        self.pos = np.array([new_x, new_y, new_z], dtype=np.int32)
        self.path.append(self.pos.copy())
        
        reward = 0.0
        done = False

        if self.grid_map[self.pos[0], self.pos[1], self.pos[2]] == 1:
            self.collision = True
            reward = -1.0
            done = True
        elif np.linalg.norm(self.pos - self.goal) <= 1.0:
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
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.policy_head = nn.Linear(HIDDEN_SIZE, output_dim)
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)
        nn.init.constant_(self.policy_head.bias, 0.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        value = self.value_head(x)
        logits = torch.clamp(self.policy_head(x), min=-20.0, max=20.0)
        return logits, value

    def get_action(self, state):
        with torch.no_grad():
            start_t = time.perf_counter()
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            infer_cost = time.perf_counter() - start_t
            return action.item(), dist.log_prob(action).item(), value.item(), infer_cost

    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_log_probs, values, entropy

# ===================== PPO训练器 =====================
class PPOTrainer:
    def __init__(self, input_dim, action_dim, batch_size):
        self.model = PPO(input_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR, eps=1e-5)
        self.memory = []
        self.batch_size = batch_size

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
        if len(self.memory) < self.batch_size:
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
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

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

# ===================== 评价指标计算器【增加CSV导出函数】 =====================
class MetricsCalculator:
    def __init__(self):
        self.metrics = {
            "path_length": [], "reward": [], "collision_rate": [],
            "success_rate": [], "smoothness": [],
            "infer_time": [],        # 单步推理耗时 s
            "total_plan_time": [],   # 单条episode完整规划耗时 s
            "avg_ep_steps": []       # 每100轮平均episode交互步数
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

    def update(self, path, reward, collision, success, infer_times, plan_total_t, ep_step_count):
        length = self.calculate_path_length(path)
        smoothness = self.calculate_smoothness(path)
        avg_infer = np.mean(infer_times) if len(infer_times) > 0 else 0.0
        self.episode_buffer.append([length, reward, float(collision), float(success), smoothness, avg_infer, plan_total_t, ep_step_count])

    def get_average(self):
        if not self.episode_buffer:
            return [0.0]*8
        arr = np.array(self.episode_buffer)
        avg_len = np.mean(arr[:,0])
        avg_rew = np.mean(arr[:,1])
        collision_rate = np.mean(arr[:,2])
        success_rate = np.mean(arr[:,3])
        avg_smooth = np.mean(arr[:,4])
        avg_infer_t = np.mean(arr[:,5])
        avg_plan_t = np.mean(arr[:,6])
        avg_step_num = np.mean(arr[:,7])

        self.metrics["path_length"].append(avg_len)
        self.metrics["reward"].append(avg_rew)
        self.metrics["collision_rate"].append(collision_rate)
        self.metrics["success_rate"].append(success_rate)
        self.metrics["smoothness"].append(avg_smooth)
        self.metrics["infer_time"].append(avg_infer_t)
        self.metrics["total_plan_time"].append(avg_plan_t)
        self.metrics["avg_ep_steps"].append(avg_step_num)
        self.episode_buffer = []

        return avg_len, avg_rew, collision_rate, success_rate, avg_smooth, avg_infer_t, avg_plan_t, avg_step_num

    # CSV导出函数，格式与A2C、改进PPO、DQN完全统一
    def save_metrics_to_csv(self, save_filepath, plot_interval):
        try:
            metric_keys = list(self.metrics.keys())
            num_records = len(self.metrics[metric_keys[0]])
            x_episodes = [i * plot_interval for i in range(1, num_records + 1)]
            header = ["Episode"] + metric_keys
            with open(save_filepath, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for idx, epi in enumerate(x_episodes):
                    row = [epi]
                    for k in metric_keys:
                        row.append(self.metrics[k][idx])
                    writer.writerow(row)
            print(f"✅ Metrics CSV saved to: {save_filepath}")
        except Exception as e:
            print(f"⚠️ CSV save error: {e}")

# ===================== 3D障碍物绘制工具 =====================
def cuboid_data(center, size=(1,1,1)):
    x, y, z = center
    dx, dy, dz = size
    vertices = [
        [x-dx/2, y-dy/2, z-dz/2],
        [x+dx/2, y-dy/2, z-dz/2],
        [x+dx/2, y+dy/2, z-dz/2],
        [x-dx/2, y+dy/2, z-dz/2],
        [x-dx/2, y-dy/2, z+dz/2],
        [x+dx/2, y-dy/2, z+dz/2],
        [x+dx/2, y+dy/2, z+dz/2],
        [x-dx/2, y+dy/2, z+dz/2]
    ]
    return np.array(vertices)

def plot_cube(ax, center, size, color='gray', alpha=0.5):
    vertices = cuboid_data(center, size)
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]]
    ]
    poly3d = [[face[0], face[1], face[2], face[3]] for face in faces]
    collection = Poly3DCollection(poly3d, facecolors=color, linewidths=1, edgecolors='black', alpha=alpha)
    ax.add_collection3d(collection)

def plot_path_3d(env, episode, save_path):
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    path = np.array(env.path)
    ax.plot(path[:,0], path[:,1], path[:,2], color='#3498db', linewidth=2.5, label='Planned Path')
    
    ax.scatter(START[0], START[1], START[2], color='#27ae60', s=100, label='Start', marker='o')
    ax.scatter(GOAL[0], GOAL[1], GOAL[2], color='#e74c3c', s=100, label='Goal', marker='*')

    cube_colors = ['#95a5a6', '#7f8c8d', '#bdc3c7', '#95a5a6', '#7f8c8d', '#bdc3c7', '#95a5a6', '#95a5a6', '#7f8c8d', '#95a5a6', '#7f8c8d', 
    '#7f8c8d','#7f8c8d','#95a5a6','#7f8c8d','#7f8c8d','#7f8c8d','#7f8c8d','#7f8c8d','#7f8c8d','#7f8c8d']
    for idx, (start_x, start_y, start_z, size) in enumerate(CUBE_OBSTACLES):
        center_x = start_x + size/2 - 0.5
        center_y = start_y + size/2 - 0.5
        center_z = start_z + size/2 - 0.5
        plot_cube(ax, (center_x, center_y, center_z), (size, size, size), color=cube_colors[idx], alpha=0.6)

    ax.set_xlim(0, MAP_SIZE)
    ax.set_ylim(0, MAP_SIZE)
    ax.set_zlim(0, MAP_SIZE)
    ax.set_xlabel('X Axis', fontsize=10)
    ax.set_ylabel('Y Axis', fontsize=10)
    ax.set_zlabel('Z Axis', fontsize=10)
    ax.set_title(f"3D Path Planning - Episode {episode}", fontsize=16, fontweight='bold')
    ax.legend(fontsize=10)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_training_curve(metrics):
    plt.figure(figsize=(18, 12), dpi=100)
    index = np.arange(1, len(metrics["path_length"])+1) * PLOT_INTERVAL

    plt.subplot(3, 2, 1)
    plt.plot(index, metrics["path_length"], color='#3498db', linewidth=2)
    plt.title("Average Path Length", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 2)
    plt.plot(index, metrics["reward"], color='#e74c3c', linewidth=2)
    plt.title("Average Reward", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 3)
    plt.plot(index, metrics["collision_rate"], color='#2c3e50', linewidth=2)
    plt.title("Collision Rate", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 4)
    plt.plot(index, metrics["success_rate"], color='#27ae60', linewidth=2)
    plt.title("Success Rate", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 5)
    smooth_data = gaussian_filter1d(metrics["smoothness"], sigma=2)
    plt.plot(index, smooth_data, color='#f39c12', linewidth=2)
    plt.title("Path Smoothness", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 6)
    infer_ms = np.array(metrics["infer_time"]) * 1000
    infer_smooth = gaussian_filter1d(infer_ms, sigma=2)
    plt.plot(index, infer_smooth, color='#9b59b6', linewidth=2)
    plt.title("Avg Single Step Inference Time (ms)", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig(f"{SAVE_DIR}/training_metrics.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 完整路径规划耗时曲线
    plt.figure(figsize=(10, 5), dpi=100)
    plan_ms = np.array(metrics["total_plan_time"]) * 1000
    plan_smooth = gaussian_filter1d(plan_ms, sigma=2)
    plt.plot(index, plan_smooth, color='#d35400', linewidth=2)
    plt.title("Avg Total Path Planning Time (ms)", fontsize=12, fontweight='bold')
    plt.xlabel("Episode")
    plt.ylabel("Time (ms)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/total_planning_time_curve.png", bbox_inches='tight')
    plt.close()

    # ========== Avg Steps/Episode 独立迭代曲线图 ==========
    plt.figure(figsize=(10, 5), dpi=100)
    step_data = np.array(metrics["avg_ep_steps"])
    step_smooth = gaussian_filter1d(step_data, sigma=2)
    plt.plot(index, step_smooth, color='#2980b9', linewidth=2)
    plt.title("Avg Steps/Episode", fontsize=12, fontweight='bold')
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/avg_steps_per_episode_curve.png", bbox_inches='tight')
    plt.close()

    print(f"✅ Training metrics saved to {SAVE_DIR}")
    print(f"✅ Total path planning time curve saved")
    print(f"✅ Avg Steps/Episode curve saved")

# ===================== 主训练流程 =====================
def main():
    local_obs_dim = (2*UNKNOWN_RADIUS+1)**3
    state_dim = local_obs_dim + 3 + 3
    action_dim = 6

    env = PathPlanningEnv3D()
    agent = PPOTrainer(state_dim, action_dim, BATCH_SIZE)
    metrics = MetricsCalculator()

    print("🚀 Start training 3D PPO path planning...")
    print(f"📊 Save path plot every {PLOT_INTERVAL} episodes\n")

    for episode in range(1, EPISODES+1):
        state = env.reset()
        total_reward = 0.0
        done = False
        episode_infer_times = []
        step_counter = 0

        plan_start = time.perf_counter()
        for _ in range(STEPS_PER_EPISODE):
            action, log_prob, value, infer_t = agent.model.get_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            episode_infer_times.append(infer_t)
            agent.store_transition([state, action, log_prob, reward, done, value])
            state = next_state
            step_counter += 1
            if done:
                break
        plan_total_time = time.perf_counter() - plan_start

        metrics.update(env.path, total_reward, env.collision, env.success, episode_infer_times, plan_total_time, step_counter)

        if episode % PLOT_INTERVAL == 0:
            avg_len, avg_rew, col_rate, suc_rate, avg_smooth, avg_infer, avg_plan, avg_steps = metrics.get_average()
            print(f"📌 Episode: {episode:4d} | Avg Length: {avg_len:6.2f} | Avg Reward: {avg_rew:7.2f} | Collision Rate: {col_rate:6.2%} | Success Rate: {suc_rate:6.2%}")
            print(f"    Smoothness: {avg_smooth:.2f} | Single Step Infer: {avg_infer*1000:.3f} ms | Total Plan Time: {avg_plan*1000:.3f} ms | Avg Steps/Episode: {avg_steps:.1f}")
            plot_path_3d(env, episode, f"{SAVE_DIR}/path_episode_{episode}.png")

        agent.update()

    plot_training_curve(metrics.metrics)
    # 训练结束导出CSV
    metrics.save_metrics_to_csv(os.path.join(SAVE_DIR, "metrics_log.csv"), PLOT_INTERVAL)
    print(f"\n🎉 Training finished! All results saved to: {SAVE_DIR}")

if __name__ == "__main__":
    # 设置随机种子确保可复现
   
    
    main()