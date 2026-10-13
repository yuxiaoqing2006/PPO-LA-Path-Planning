import numpy as np
import matplotlib
matplotlib.use('Agg')  # FIX: Linux无GUI支持，放plt导入前
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
import os
from scipy.ndimage import gaussian_filter1d
import random
from collections import deque
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time  # 导入计时模块
import csv  # 新增csv模块

# ===================== 全局参数设置 =====================
# 环境参数（三维地图）
MAP_SIZE = 20
START = (1, 1, 1)  # 三维起点
GOAL = (18, 18, 18) # 三维终点
UNKNOWN_RADIUS = 3  # 三维局部观测半径
# DQN算法参数
EPISODES = 2500
STEPS_PER_EPISODE = 300  # 三维路径需要更多步数
HIDDEN_SIZE = 64       # 增加隐藏层大小适配更高维度输入
LR = 2e-4
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
SAVE_DIR = "./dqn_3d_map_planning_obstacle"
os.makedirs(SAVE_DIR, exist_ok=True)
# 设备自动适配
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 障碍物参数（来自代码2）
CUBE_OBSTACLES = [
    (10, 2, 3, 2),(2, 10, 10, 2),(15, 4, 17, 2),(9, 10, 9, 2),(17, 2, 3, 2),(2, 2, 13, 2),(12, 10, 7, 2),
    (17, 13, 10, 2), (10, 17, 10, 2), (5, 10, 15, 2),(12, 12, 17, 2),(2, 13, 17, 2),(8, 10, 8, 2),
    (17, 12, 3, 2), (5, 10, 1, 2),(13, 5, 9, 2), (5, 5, 9, 2),
    (3, 5, 3, 2), (3, 5, 3, 2),(5, 15, 9, 2),(17, 12, 17, 2)
]

# ===================== 3D立方体绘制函数 =====================
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
        """生成带障碍物的三维地图（完全来自代码2）"""
        grid_map = np.zeros((self.map_size, self.map_size, self.map_size), dtype=np.int32)
        # 边界
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
        
        # 中间平面墙
        for x in range(5, 15):
            for y in range(5, 15):
                fixed_obstacles_3d.append((x, y, 9))
        
        # 立方体障碍物
        for (start_x, start_y, start_z, size) in CUBE_OBSTACLES:
            for x in range(start_x, start_x + size):
                for y in range(start_y, start_y + size):
                    for z in range(start_z, start_z + size):
                        fixed_obstacles_3d.append((x, y, z))
        
        # 写入地图
        for (i, j, k) in fixed_obstacles_3d:
            if 0 <= i < self.map_size and 0 <= j < self.map_size and 0 <= k < self.map_size:
                grid_map[i, j, k] = 1

        return grid_map

    def get_local_observation(self):
        """获取三维局部观测"""
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
        """FIX:重构碰撞检测逻辑：先预判下一位置，禁止穿墙"""
        dx, dy, dz = [(0,0,1), (0,0,-1), (0,1,0), (0,-1,0), (1,0,0), (-1,0,0)][action]
        
        # 候选下一坐标
        cand_x = np.clip(self.pos[0] + dx, 1, self.map_size-2)
        cand_y = np.clip(self.pos[1] + dy, 1, self.map_size-2)
        cand_z = np.clip(self.pos[2] + dz, 1, self.map_size-2)
        candidate_pos = np.array([cand_x, cand_y, cand_z], dtype=np.int32)

        reward = 0.0
        done = False

        # FIX：预判障碍物，如果下一格是障碍，发生碰撞，位置不变
        if self.grid_map[candidate_pos[0], candidate_pos[1], candidate_pos[2]] == 1:
            self.collision = True
            reward = -1.0
            done = True
            # 原地不动，路径不再新增碰撞点
        else:
            # 合法移动，更新位置
            self.pos = candidate_pos
            self.path.append(self.pos.copy())
            
            # FIX: 修正终点判定阈值
            if np.linalg.norm(self.pos - self.goal) <= 1.0:
                self.success = True
                reward = 1.0
                done = True
            else:
                prev_dist = np.linalg.norm(self.path[-2] - self.goal)
                curr_dist = np.linalg.norm(self.pos - self.goal)
                reward = (prev_dist - curr_dist) * 0.1 - 0.01

        return self.get_local_observation(), reward, done

# ===================== DQN网络（适配三维输入） =====================
class DQN3D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN3D, self).__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc4 = nn.Linear(HIDDEN_SIZE, output_dim)
        
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc4.weight, gain=0.01)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)
        nn.init.constant_(self.fc4.bias, 0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.fc4(x)

# ===================== DQN智能体【FIX: Double DQN + 移除内部eps衰减】 =====================
class DQNAgent3D:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eps = EPS_START
        self.policy_net = DQN3D(state_dim, action_dim).to(DEVICE)
        self.target_net = DQN3D(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR, eps=1e-5)
        self.memory = deque(maxlen=MEMORY_CAPACITY)

    def choose_action(self, state):
        infer_start = time.perf_counter()
        infer_time = 0.0  # FIX:随机动作不计入推理耗时
        if random.random() < self.eps:
            act = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state_tensor)
                act = q_values.argmax().item()
            infer_time = time.perf_counter() - infer_start
        return act, infer_time

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # FIX：升级为 Double DQN 解决过估计
        with torch.no_grad():
            next_action = self.policy_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(1, next_action.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * GAMMA * next_q_values

        loss = nn.MSELoss()(q_values.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()
        # FIX：删除此处eps衰减，移到主循环

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ===================== 评价指标计算器【新增CSV导出函数，与PPO/A2C格式统一】 =====================
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
        valid_infer = [t for t in infer_list if t>0]
        avg_infer = np.mean(valid_infer) if len(valid_infer) > 0 else 0.0
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

# ===================== 三维可视化工具（带障碍物） =====================
def plot_path_3d(env, episode, save_path):
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(START[0], START[1], START[2], color='#27ae60', s=100, label='Start', marker='o')
    ax.scatter(GOAL[0], GOAL[1], GOAL[2], color='#e74c3c', s=100, label='Goal', marker='*')
    
    path = np.array(env.path)
    ax.plot(path[:,0], path[:,1], path[:,2], color='#3498db', linewidth=2.5, label='Planned Path')
    
    # 绘制障碍物
    cube_colors = ['#95a5a6', '#7f8c8d', '#bdc3c7', '#95a5a6', '#7f8c8d', '#bdc3c7', '#95a5a6', '#95a5a6', 
                   '#7f8c8d', '#95a5a6', '#7f8c8d', '#7f8c8d','#7f8c8d','#95a5a6','#7f8c8d','#7f8c8d',
                   '#7f8c8d','#7f8c8d','#7f8c8d','#7f8c8d','#7f8c8d','#7f8c8d']
    for idx, (start_x, start_y, start_z, size) in enumerate(CUBE_OBSTACLES):
        cx = start_x + size/2 - 0.5
        cy = start_y + size/2 - 0.5
        cz = start_z + size/2 - 0.5
        plot_cube(ax, (cx, cy, cz), (size, size, size), color=cube_colors[idx % len(cube_colors)], alpha=0.6)
    
    ax.set_xlim(0, MAP_SIZE)
    ax.set_ylim(0, MAP_SIZE)
    ax.set_zlim(0, MAP_SIZE)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(f"3D Path Planning - Episode {episode}", fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
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
    plt.title("Path Smoothness (3D)", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    # 单步推理耗时总览图子图
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

    # 独立图1：Total Plan Time (ms)
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

    # 独立图2：Avg Steps/Episode
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

    print(f"✅ 3D Training metrics saved to {SAVE_DIR}")
    print(f"✅ Single Step Infer / Total Plan Time / Avg Steps 曲线已保存")

# ===================== 主训练流程【FIX: eps衰减移至episode循环 + 随机种子】 =====================
def main():

    local_obs_dim = (2*UNKNOWN_RADIUS+1)**3
    state_dim = local_obs_dim + 3 + 3
    action_dim = 6
    
    env = PathPlanningEnv3D()
    agent = DQNAgent3D(state_dim, action_dim)
    metrics = MetricsCalculator3D()
    
    print("🚀 Start training 3D DQN path planning with obstacles...")
    print(f"📊 Save 3D path plot every {PLOT_INTERVAL} episodes\n")

    for episode in range(1, EPISODES+1):
        state = env.reset()
        total_reward = 0.0
        done = False
        episode_infer_times = []
        step_counter = 0

        plan_start = time.perf_counter()
        for _ in range(STEPS_PER_EPISODE):
            action, infer_t = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            episode_infer_times.append(infer_t)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            step_counter += 1
            agent.update()
            if done:
                break
        plan_total_time = time.perf_counter() - plan_start

        # FIX：每轮episode结束统一衰减epsilon
        if agent.eps > EPS_END:
            agent.eps *= EPS_DECAY

        if episode % TARGET_UPDATE_INTERVAL == 0:
            agent.update_target_net()

        metrics.update(env.path, total_reward, env.collision, env.success, episode_infer_times, plan_total_time, step_counter)

        if episode % PLOT_INTERVAL == 0:
            avg_len, avg_rew, col_rate, suc_rate, avg_smooth, avg_infer, avg_plan, avg_steps = metrics.get_average()
            print(f"📌 Episode: {episode:4d} | Avg Length: {avg_len:6.2f} | Avg Reward: {avg_rew:7.2f} | Collision Rate: {col_rate:6.2%} | Success Rate: {suc_rate:6.2%}")
            print(f"    Smoothness: {avg_smooth:.2f} | Single Step Infer: {avg_infer*1000:.3f} ms | Total Plan Time: {avg_plan*1000:.3f} ms | Avg Steps/Episode: {avg_steps:.1f}")
            plot_path_3d(env, episode, f"{SAVE_DIR}/3d_path_episode_{episode}.png")

    plot_training_curve_3d(metrics.metrics)
    # 训练结束导出CSV
    metrics.save_metrics_to_csv(os.path.join(SAVE_DIR, "metrics_log.csv"), PLOT_INTERVAL)
    print(f"\n🎉 3D Training finished! All results saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()