# Strictly Aligned OpenAI Baselines Standard A2C
# Vectorized synchronous multi-environment rollout
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 适配服务器无GUI
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
import csv

# ===================== 全局参数设置 =====================
# 环境参数（三维地图）
MAP_SIZE = 20
START = (1, 1, 1)
GOAL = (18, 18, 18)
UNKNOWN_RADIUS = 3
# A2C算法参数【严格对齐 OpenAI Baselines A2C】
EPISODES = 2500
STEPS_PER_EPISODE = 300
HIDDEN_SIZE = 64
LR = 1.4e-4
GAMMA = 0.99
MAX_GRAD_NORM = 0.3
ENTROPY_COEFF = 0.01
NUM_ENVS = 4        # 并行环境数量
ROLLOUT_STEPS = 128 # n-step rollout

# 绘图与保存参数
PLOT_INTERVAL = 100
SAVE_DIR = "./a2c_3d_map_planning_obstacle_baselines"
os.makedirs(SAVE_DIR, exist_ok=True)
# 设备自动适配
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# 障碍物参数
CUBE_OBSTACLES = [
    (10, 2, 3, 2),(2, 10, 10, 2),(15, 4, 17, 2),(9, 10, 9, 2),(17, 2, 3, 2),(2, 2, 13, 2),(12, 10, 7, 2),
    (17, 13, 10, 2), (10, 17, 10, 2), (5, 10, 15, 2),(12, 12, 17, 2),(2, 13, 17, 2),(8, 10, 8, 2),
    (17, 12, 3, 2), (5, 10, 1, 2),(13, 5, 9, 2), (5, 5, 9, 2),
    (3, 5, 3, 2),(5, 15, 9, 2),(17, 12, 17, 2)
]

# ===================== 3D立方体绘制工具 =====================
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

# ===================== 三维带障碍物地图环境类 =====================
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
        dx, dy, dz = [(-1,0,0), (1,0,0),
                      (0,-1,0), (0,1,0),
                      (0,0,-1), (0,0,1)][action]
        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy
        new_z = self.pos[2] + dz

        reward = 0.0
        done = False
        if 1 <= new_x < self.map_size-1 and 1 <= new_y < self.map_size-1 and 1 <= new_z < self.map_size-1:
            if self.grid_map[new_x, new_y, new_z] == 1:
                self.collision = True
                reward = -0.8
                done = True
            else:
                self.pos[0] = new_x
                self.pos[1] = new_y
                self.pos[2] = new_z
        else:
            self.collision = True
            reward = -0.8
            done = True

        self.path.append(self.pos.copy())

        if not self.collision:
            if np.linalg.norm(self.pos - self.goal) <= 1.0:
                self.success = True
                reward = 1.0
                done = True
            else:
                prev_dist = np.linalg.norm(self.path[-2] - self.goal)
                curr_dist = np.linalg.norm(self.pos - self.goal)
                reward = (prev_dist - curr_dist) * 0.1 - 0.01

        next_obs = self.get_local_observation()
        return next_obs, reward, done

# ===================== A2C网络 =====================
class A2C3D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(A2C3D, self).__init__()
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
        infer_start = time.perf_counter()
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).item()
            act_item = action.item()
            val_item = value.item()
        infer_cost = time.perf_counter() - infer_start
        return act_item, log_prob, val_item, infer_cost

    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_log_probs, values, entropy

# ===================== 评价指标计算器 =====================
class MetricsCalculator3D:
    def __init__(self):
        self.metrics = {
            "path_length": [], "reward": [], "collision_rate": [],
            "success_rate": [], "smoothness": [],
            "infer_time": [],        # Single Step Infer (s)
            "total_plan_time": [],    # Total Plan Time (s)
            "avg_ep_steps": []        # Avg Steps/Episode
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

    def update(self, path, reward, collision, success, infer_list, plan_total_t, ep_step_count):
        length = self.calculate_path_length(path)
        smoothness = self.calculate_smoothness(path)
        avg_infer = np.mean(infer_list) if len(infer_list) > 0 else 0.0
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

# ===================== 三维可视化工具 =====================
def plot_3d_path(env, episode, save_path):
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(START[0], START[1], START[2], color='#27ae60', s=100, label='Start', marker='o')
    ax.scatter(GOAL[0], GOAL[1], GOAL[2], color='#e74c3c', s=100, label='Goal', marker='*')
    path = np.array(env.path)
    ax.plot(path[:,0], path[:,1], path[:,2], color='#3498db', linewidth=2.5, label='Planned Path')
    cube_colors = ['#95a5a6', '#7f8c8d', '#bdc3c7', '#95a5a6', '#7f8c8d', '#bdc3c7', '#95a5a6', '#95a5a6',
                   '#7f8c8d', '#95a5a6', '#7f8c8d', '#7f8c8d','#7f8c8d','#95a5a6','#7f8c8d','#7f8c8d',
                   '#7f8c8d','#7f8c8d','#7f8c8d','#7f8c8d','#7f8c8d']
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
    ax.legend()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_training_curve_3d(metrics):
    plt.figure(figsize=(15, 10), dpi=100)
    index = np.arange(1, len(metrics["path_length"])+1) * PLOT_INTERVAL

    plt.subplot(2, 3, 1)
    plt.plot(index, metrics["path_length"], color='#3498db', linewidth=2)
    plt.title("Average Path Length (3D)", fontsize=12, fontweight='bold')
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
    plt.title("Path Smoothness (Sum of Turning Angles)", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    infer_ms = np.array(metrics["infer_time"]) * 1000
    infer_smooth = gaussian_filter1d(infer_ms, sigma=2)
    plt.plot(index, infer_smooth, color='#9b59b6', linewidth=2)
    plt.title("Single Step Infer (ms)", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig(f"{SAVE_DIR}/training_metrics_3d.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Total Plan Time
    plt.figure(figsize=(10, 5), dpi=100)
    plan_ms = np.array(metrics["total_plan_time"]) * 1000
    plan_smooth = gaussian_filter1d(plan_ms, sigma=2)
    plt.plot(index, plan_smooth, color='#d35400', linewidth=2)
    plt.title("Total Plan Time (ms)", fontsize=12, fontweight='bold')
    plt.xlabel("Episode")
    plt.ylabel("Time (ms)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/total_planning_time_curve.png", bbox_inches='tight')
    plt.close()

    # Avg Steps per Episode
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

    print(f"✅ All training curves saved to {SAVE_DIR}")

# ===================== 【严格复刻 OpenAI Baselines A2C 向量化并行主循环】 =====================
def main():
    local_obs_dim = (2*UNKNOWN_RADIUS+1)**3
    state_dim = local_obs_dim + 3 + 3
    action_dim = 6
    
    model = A2C3D(state_dim, action_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    metrics = MetricsCalculator3D()

    # 创建NUM_ENVS独立环境
    envs = [PathPlanningEnv3D() for _ in range(NUM_ENVS)]
    obs_list = [env.reset() for env in envs]
    obs_list = np.array(obs_list, dtype=np.float32)

    eval_env = PathPlanningEnv3D()
    global_episode = 0

    print(f"🚀 Strict OpenAI Baselines A2C (Vectorized Sync Env) 3D path planning, Num Envs={NUM_ENVS}")
    print(f"📊 Save 3D path plot every {PLOT_INTERVAL} episodes\n")

    for update_iter in range(EPISODES):
        # ========== Baselines A2C Rollout Buffer 结构 ==========
        rollout_obs = np.zeros((ROLLOUT_STEPS, NUM_ENVS, state_dim), dtype=np.float32)
        rollout_acts = np.zeros((ROLLOUT_STEPS, NUM_ENVS), dtype=np.int64)
        rollout_rews = np.zeros((ROLLOUT_STEPS, NUM_ENVS), dtype=np.float32)
        rollout_dones = np.zeros((ROLLOUT_STEPS, NUM_ENVS), dtype=np.float32)
        rollout_vals = np.zeros((ROLLOUT_STEPS, NUM_ENVS), dtype=np.float32)

        # 【新增：用于记录每个环境episode计时、指标缓存，对齐PPO采集逻辑】
        env_running = [True] * NUM_ENVS
        env_start_time = [0.0] * NUM_ENVS
        env_infer_buffer = [[] for _ in range(NUM_ENVS)]
        env_total_reward = [0.0] * NUM_ENVS
        env_step_count = [0] * NUM_ENVS

        # 同步收集完整 ROLLOUT_STEPS 步（原版不提前截断）
        for step in range(ROLLOUT_STEPS):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs_list).to(DEVICE)
                logits, vals = model(obs_tensor)
                dist = Categorical(logits=logits)
                acts = dist.sample()
                acts_np = acts.cpu().numpy()
                vals_np = vals.squeeze(-1).cpu().numpy()

            rollout_obs[step] = obs_list.copy()
            rollout_acts[step] = acts_np
            rollout_vals[step] = vals_np

            # 同步所有环境步进
            next_obs_batch = []
            rew_batch = []
            done_batch = []
            for e_idx, env in enumerate(envs):
                # 环境刚reset，开启计时
                if env_running[e_idx]:
                    if env_step_count[e_idx] == 0:
                        env_start_time[e_idx] = time.perf_counter()
                    # 单步推理计时
                    _, _, _, infer_t = model.get_action(obs_list[e_idx])
                    env_infer_buffer[e_idx].append(infer_t)

                n_obs, r, d = env.step(acts_np[e_idx])
                next_obs_batch.append(n_obs)
                rew_batch.append(r)
                done_batch.append(float(d))
                env_total_reward[e_idx] += r
                env_step_count[e_idx] += 1

                # episode终止，采集指标存入metrics buffer（与PPO逻辑对齐）
                if env_running[e_idx] and d:
                    plan_total = time.perf_counter() - env_start_time[e_idx]
                    metrics.update(env.path, env_total_reward[e_idx], env.collision, env.success,
                                   env_infer_buffer[e_idx], plan_total, env_step_count[e_idx])
                    # 重置该环境episode状态
                    env_running[e_idx] = False
                    env_infer_buffer[e_idx] = []
                    env_total_reward[e_idx] = 0.0
                    env_step_count[e_idx] = 0

                if d:
                    next_obs_batch[e_idx] = env.reset()
                    env_running[e_idx] = True

            obs_list = np.array(next_obs_batch, dtype=np.float32)
            rollout_rews[step] = np.array(rew_batch, dtype=np.float32)
            rollout_dones[step] = np.array(done_batch, dtype=np.float32)

        # ========== 复刻Baselines n-step return计算 ==========
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_list).to(DEVICE)
            _, last_v = model(obs_tensor)
        last_v_np = last_v.squeeze(-1).cpu().numpy()

        returns = np.zeros_like(rollout_rews)
        current_R = last_v_np
        for t in reversed(range(ROLLOUT_STEPS)):
            current_R = rollout_rews[t] + GAMMA * current_R * (1.0 - rollout_dones[t])
            returns[t] = current_R

        # 展平数据用于训练
        obs_flat = rollout_obs.reshape(-1, state_dim)
        acts_flat = rollout_acts.reshape(-1)
        returns_flat = returns.reshape(-1)
        vals_flat = rollout_vals.reshape(-1)

        states_tensor = torch.FloatTensor(obs_flat).to(DEVICE)
        actions_tensor = torch.LongTensor(acts_flat).to(DEVICE)
        returns_tensor = torch.FloatTensor(returns_flat).unsqueeze(1).to(DEVICE)
        vals_tensor = torch.FloatTensor(vals_flat).unsqueeze(1).to(DEVICE)

        log_probs, values_pred, entropy = model.evaluate(states_tensor, actions_tensor)
        log_probs = log_probs.unsqueeze(1)

        # Advantage = Return - V(s) 【原版Baselines不做advantage标准化！删掉归一化】
        advantages = returns_tensor - vals_tensor.detach()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.MSELoss()(values_pred, returns_tensor)
        entropy_loss = -ENTROPY_COEFF * entropy.mean()
        loss = policy_loss + 0.5 * value_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()

        global_episode += 1

        # 周期性评估绘图：仅用来输出可视化图片，不再参与指标计算
        if global_episode % PLOT_INTERVAL == 0:
            avg_len, avg_rew, col_rate, suc_rate, avg_smooth, avg_infer, avg_plan, avg_steps = metrics.get_average()
            print(f"📌 Update Iter: {global_episode:4d} | Avg Length: {avg_len:6.2f} | Avg Reward: {avg_rew:7.2f} | Collision Rate: {col_rate:6.2%} | Success Rate: {suc_rate:6.2%}")
            print(f"    Smoothness: {avg_smooth:.3f} | Single Step Infer: {avg_infer*1000:.3f} ms | Total Plan Time: {avg_plan*1000:.3f} ms | Avg Steps/Episode: {avg_steps:.1f}")
            # 仅绘图使用
            state = eval_env.reset()
            done_flag = False
            for _ in range(STEPS_PER_EPISODE):
                act, _, _, _ = model.get_action(state)
                ns, rew, done_flag = eval_env.step(act)
                state = ns
                if done_flag:
                    break
            plot_3d_path(eval_env, global_episode, f"{SAVE_DIR}/3d_path_episode_{global_episode}.png")

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "a2c_3d_model_baselines.pth"))
    print("✅ Model weights saved.")
    plot_training_curve_3d(metrics.metrics)
    metrics.save_metrics_to_csv(os.path.join(SAVE_DIR, "metrics_log.csv"), PLOT_INTERVAL)
    print(f"\n🎉 Strict OpenAI Baselines A2C Training finished! All results saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()