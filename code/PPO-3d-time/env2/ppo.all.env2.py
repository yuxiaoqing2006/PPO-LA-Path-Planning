#PPO-all
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os
from scipy.ndimage import gaussian_filter1d
import time
import csv  # 新增csv模块

# ===================== 全局参数设置 =====================
# 环境参数（三维地图）
MAP_SIZE = 20
START = (1, 1, 1)  # 三维起点
GOAL = (18, 18, 18)  # 三维终点
UNKNOWN_RADIUS = 3

# PPO算法稳定参数
EPISODES = 2500
STEPS_PER_EPISODE = 300  # 增加步数适应三维空间
HIDDEN_SIZE = 64
LSTM_HIDDEN = 128  # LSTM隐藏层维度
ATTN_HEADS = 4    # 自注意力头数
LR = 1e-4
GAMMA = 0.99
EPS_CLIP = 0.2
UPDATE_EPOCHS = 3
BATCH_SIZE = 64
MAX_GRAD_NORM = 0.3

# 绘图与保存参数
PLOT_INTERVAL = 100
SAVE_DIR = "./ppo_3d_map_planning_lstm_attn"
os.makedirs(SAVE_DIR, exist_ok=True)

# 新增立方体障碍物参数
CUBE_OBSTACLES = [
    # (起始x, 起始y, 起始z, 立方体边长)
    (10, 2, 3, 2),(2, 10, 10, 2),(15, 4, 17, 2),(9, 10, 9, 2),(17, 2, 3, 2),(2, 2, 13, 2),(12, 10, 7, 2),
    (17, 13, 10, 2), (10, 17, 10, 2), (5, 10, 15, 2),(12, 12, 17, 2),(2, 13, 17, 2),(8, 10, 8, 2),
    (17, 12, 3, 2), (5, 10, 1, 2),(13, 5, 9, 2), (5, 5, 9, 2),
    (3, 5, 3, 2), (3, 5, 3, 2),(5, 15, 9, 2),(17, 12, 17, 2)
]

# 设备自动适配
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

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
        """生成三维固定障碍物地图，新增立方体障碍物"""
        grid_map = np.zeros((self.map_size, self.map_size, self.map_size), dtype=np.int32)
        # 边界障碍物（三维边界）
        grid_map[0, :, :] = 1
        grid_map[-1, :, :] = 1
        grid_map[:, 0, :] = 1
        grid_map[:, -1, :] = 1
        grid_map[:, :, 0] = 1
        grid_map[:, :, -1] = 1

        # 三维内部障碍物（基于原二维障碍物扩展）
        fixed_obstacles_2d = [
            (17,10), (16,13), (16,14), (15,6), (15,18), (14,2),
            (13,1), (13,8), (13,11), (13,15), (13,17), (11,5),
            (10,1), (10,5), (10,11), (10,14), (10,16), (10,17), (10,18),
            (9,1), (8,1), (8,4), (8,11), (8,14), (7,1), (6,4),
            (5,5), (5,6), (5,7), (5,10), (5,12), (5,13), (5,15),
            (4,5), (4,6), (3,3), (3,13), (3,14), (2,5), (2,15), (2,18), (1,15)
        ]

        # 将二维障碍物扩展到三维（多个高度层）
        fixed_obstacles_3d = []
        for (i, j) in fixed_obstacles_2d:
            # 在多个z轴高度生成障碍物
            for z in [2, 4, 6, 8, 10, 12, 14, 16]:
                fixed_obstacles_3d.append((i, j, z))

        # 添加纯三维障碍物
        for x in range(5, 15):
            for y in range(5, 15):
                fixed_obstacles_3d.append((x, y, 9))  # 中间层障碍物平面

        # 添加立方体障碍物
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
        """获取三维局部观测，确保数值稳定性"""
        x, y, z = self.pos
        # 三维局部观测初始化
        local_obs = np.ones((2*self.unknown_radius+1,
                            2*self.unknown_radius+1,
                            2*self.unknown_radius+1), dtype=np.float32) * 0.5

        # 遍历三维局部区域
        for dx in range(-self.unknown_radius, self.unknown_radius+1):
            for dy in range(-self.unknown_radius, self.unknown_radius+1):
                for dz in range(-self.unknown_radius, self.unknown_radius+1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size and 0 <= nz < self.map_size:
                        local_obs[dx+self.unknown_radius,
                                 dy+self.unknown_radius,
                                 dz+self.unknown_radius] = self.grid_map[nx, ny, nz]

        # 展平局部观测并添加数值稳定性
        local_flat = local_obs.flatten()
        local_flat = np.clip(local_flat, 0.0, 1.0)  # 防止异常值

        # 归一化位置（添加小常数防止除零）
        norm_pos = self.pos / (self.map_size + 1e-8)
        # 相对目标位置（添加小常数防止除零）
        relative_goal = (self.goal - self.pos) / (self.map_size + 1e-8)

        state = np.concatenate([local_flat, norm_pos, relative_goal], dtype=np.float32)
        # 确保没有NaN/Inf
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        return state

    def reset(self):
        self.grid_map = self.generate_map()
        self.pos = self.start.copy()
        self.path = [self.pos.copy()]
        self.collision = False
        self.success = False
        return self.get_local_observation()

    def step(self, action):
        """执行三维动作，返回新状态、奖励、结束标志"""
        # 6个动作：x-,y+,x+,y-,z+,z-，无原地
        action = np.clip(action, 0, 5)
        dx, dy, dz = [(-1,0,0), (0,1,0), (1,0,0), (0,-1,0),
                      (0,0,1), (0,0,-1)][action]

        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy
        new_z = self.pos[2] + dz

        # 三维边界检查
        if 1 <= new_x < self.map_size-1 and 1 <= new_y < self.map_size-1 and 1 <= new_z < self.map_size-1:
            self.pos[0] = new_x
            self.pos[1] = new_y
            self.pos[2] = new_z

        self.path.append(self.pos.copy())
        reward = 0.0
        done = False

        # 碰撞检查
        if self.grid_map[self.pos[0], self.pos[1], self.pos[2]] == 1:
            self.collision = True
            reward = -1.0
            done = True
        # 到达目标检查
        elif np.linalg.norm(self.pos - self.goal) < 1.0:  # 三维判定阈值
            self.success = True
            reward = 1.0  # 增加奖励鼓励到达目标
            done = True
        # 普通步骤奖励
        else:
            prev_dist = np.linalg.norm(self.path[-2] - self.goal)
            curr_dist = np.linalg.norm(self.pos - self.goal)
            reward = (prev_dist - curr_dist) * 0.1 - 0.01  # 增加向目标移动奖励

        return self.get_local_observation(), reward, done

# ===================== 改进版PPO网络（LSTM+自注意力） =====================
class PPO3D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPO3D, self).__init__()
        # 1. 共享特征主干（增大网络容量）
        self.fc1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)  # 新增一层适应三维

        # 2. Actor分支（策略网络）- 加入LSTM
        self.lstm = nn.LSTM(
            input_size=HIDDEN_SIZE,
            hidden_size=LSTM_HIDDEN,
            num_layers=1,
            batch_first=True,
            dropout=0.1
        )
        self.policy_head = nn.Linear(LSTM_HIDDEN, output_dim)

        # 3. Critic分支（价值网络）- 加入自注意力
        self.attn_embed = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=HIDDEN_SIZE,
            num_heads=ATTN_HEADS,
            dropout=0.1,
            batch_first=True
        )
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)

        # 4. 权重初始化
        self._init_weights()

    def _init_weights(self):
        """统一初始化所有层的权重"""
        # 共享层
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)

        # LSTM层
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data, gain=np.sqrt(2))
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)

        # 策略头
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)

        # 注意力层
        nn.init.orthogonal_(self.attn_embed.weight, gain=np.sqrt(2))
        nn.init.constant_(self.attn_embed.bias, 0.0)
        for name, param in self.self_attn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)

        # 价值头
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, x):
        # x.shape: (batch_size, input_dim)
        batch_size = x.shape[0]

        # 共享特征提取
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # (batch_size, HIDDEN_SIZE)

        # === Actor分支（LSTM）===
        lstm_input = x.unsqueeze(1)  # (batch_size, 1, HIDDEN_SIZE)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = lstm_out.squeeze(1)  # (batch_size, LSTM_HIDDEN)
        logits = torch.clamp(self.policy_head(lstm_out), min=-20.0, max=20.0)

        # === Critic分支（自注意力）===
        attn_input = self.attn_embed(x).unsqueeze(1)
        attn_out, _ = self.self_attn(attn_input, attn_input, attn_input)
        attn_out = attn_out.squeeze(1)
        value = self.value_head(attn_out)

        return logits, value

    def get_action(self, state):
        """获取动作，增加数值稳定性检查，新增计时"""
        with torch.no_grad():
            start_t = time.perf_counter()  # 推理计时起点
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

            logits, value = self.forward(state)
            logits = torch.clamp(logits, min=-100.0, max=100.0)
            dist = Categorical(logits=logits)

            action = dist.sample()
            log_prob = dist.log_prob(action)

            action = torch.clamp(action, 0, 5)
            log_prob = torch.nan_to_num(log_prob, nan=0.0)
            value = torch.nan_to_num(value, nan=0.0)

            infer_cost = time.perf_counter() - start_t  # 单步推理耗时 s

            return action.item(), log_prob.item(), value.item(), infer_cost

    def evaluate(self, states, actions):
        """评估动作概率，增加数值稳定性"""
        states = torch.nan_to_num(states, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = torch.clamp(actions, 0, 5)

        logits, values = self.forward(states)
        logits = torch.clamp(logits, min=-100.0, max=100.0)

        dist = Categorical(logits=logits)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        action_log_probs = torch.nan_to_num(action_log_probs, nan=0.0)
        values = torch.nan_to_num(values, nan=0.0)
        entropy = torch.nan_to_num(entropy, nan=0.0)

        return action_log_probs, values, entropy

# ===================== PPO训练器 =====================
class PPOTrainer3D:
    def __init__(self, input_dim, action_dim):
        self.model = PPO3D(input_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR, eps=1e-5)
        self.memory = []

    def store_transition(self, transition):
        """存储经验，过滤异常值"""
        for elem in transition:
            if isinstance(elem, np.ndarray):
                if np.any(np.isnan(elem)) or np.any(np.isinf(elem)):
                    return
            elif isinstance(elem, float):
                if np.isnan(elem) or np.isinf(elem):
                    return
        self.memory.append(transition)

    def compute_gae(self, rewards, dones, values, next_value):
        """计算GAE优势函数，增强数值稳定性"""
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
        values = torch.tensor(values, dtype=torch.float32).to(DEVICE)
        next_value = torch.tensor(next_value, dtype=torch.float32).to(DEVICE)

        rewards = torch.nan_to_num(rewards, nan=0.0)
        dones = torch.nan_to_num(dones, nan=0.0)
        values = torch.nan_to_num(values, nan=0.0)
        next_value = torch.nan_to_num(next_value, nan=0.0)

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
        """更新策略网络，增强稳定性"""
        if len(self.memory) < BATCH_SIZE:
            return

        # 提取经验数据
        states = torch.FloatTensor([t[0] for t in self.memory]).to(DEVICE)
        actions = torch.LongTensor([t[1] for t in self.memory]).to(DEVICE)
        old_log_probs = torch.FloatTensor([t[2] for t in self.memory]).to(DEVICE)
        rewards = [t[3] for t in self.memory]
        dones = [t[4] for t in self.memory]
        values = [t[5] for t in self.memory]

        # 检查并过滤NaN
        if (torch.isnan(states).any() or torch.isnan(actions).any() or
            torch.isnan(old_log_probs).any() or np.any(np.isnan(rewards)) or
            np.any(np.isnan(dones)) or np.any(np.isnan(values))):
            print("⚠️  NaN detected, skip this update")
            self.memory.clear()
            return

        # 计算GAE
        with torch.no_grad():
            _, next_value = self.model(states[-1:])
        advantages, returns = self.compute_gae(rewards, dones, values, next_value.item())

        # 转换为tensor并标准化优势函数
        advantages = torch.FloatTensor(advantages).to(DEVICE)
        returns = torch.FloatTensor(returns).to(DEVICE).unsqueeze(1)

        # 标准化（添加小常数防止除零）
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, advantages, returns)
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        # 多轮更新
        for _ in range(UPDATE_EPOCHS):
            for state, action, old_log_prob, adv, ret in loader:
                new_log_prob, value, entropy = self.model.evaluate(state, action)

                # PPO损失计算
                ratio = torch.exp(new_log_prob - old_log_prob)
                ratio = torch.clamp(ratio, 1e-10, 1e10)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(value, ret)
                entropy_loss = -0.01 * entropy.mean()
                loss = policy_loss + 0.5 * value_loss + entropy_loss

                # 梯度更新
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=MAX_GRAD_NORM)
                self.optimizer.step()

        self.memory.clear()

# ===================== 评价指标计算器（【已修改】增加Avg Steps/Episode + CSV导出） =====================
class MetricsCalculator3D:
    def __init__(self):
        # 新增 avg_ep_steps：每回合平均交互步数
        self.metrics = {
            "path_length": [], "reward": [], "collision_rate": [],
            "success_rate": [], "smoothness": [],
            "infer_time": [],    # 单步网络推理时间
            "total_plan_time": [],# 整条episode路径规划总时间
            "avg_ep_steps": []    # Avg Steps/Episode
        }
        self.episode_buffer = []

    def calculate_path_length(self, path):
        """计算三维路径长度"""
        length = 0.0
        for i in range(1, len(path)):
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
        return length

    def calculate_smoothness(self, path):
        """计算三维路径平滑度"""
        if len(path) < 3:
            return 0.0
        smoothness = 0.0
        for i in range(1, len(path)-1):
            vec1 = np.array(path[i]) - np.array(path[i-1])
            vec2 = np.array(path[i+1]) - np.array(path[i])
            norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            if norm > 1e-8:
                cos_theta = np.clip(np.dot(vec1, vec2) / norm, -1.0, 1.0)
                angle = np.arccos(cos_theta)
                smoothness += np.abs(angle)
        return smoothness

    def update(self, path, reward, collision, success, infer_times, plan_total_t, ep_step_count):
        """更新指标缓冲区
        :param infer_times: 当前episode每一步网络推理耗时列表
        :param plan_total_t: 当前episode完整路径规划总耗时(s)
        :param ep_step_count: 当前episode实际执行交互步数
        """
        length = self.calculate_path_length(path)
        smoothness = self.calculate_smoothness(path)
        avg_infer = np.mean(infer_times) if len(infer_times) > 0 else 0.0
        self.episode_buffer.append([length, reward, float(collision), float(success),
                                    smoothness, avg_infer, plan_total_t, ep_step_count])

    def get_average(self):
        """计算平均指标"""
        if not self.episode_buffer:
            return [0.0]*8
        arr = np.array(self.episode_buffer)
        arr = np.nan_to_num(arr, nan=0.0)

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
        self.metrics["collision_rate"].append(avg_len)
        self.metrics["success_rate"].append(success_rate)
        self.metrics["smoothness"].append(avg_smooth)
        self.metrics["infer_time"].append(avg_infer_t)
        self.metrics["total_plan_time"].append(avg_plan_t)
        self.metrics["avg_ep_steps"].append(avg_step_num)
        self.episode_buffer = []

        return avg_len, avg_rew, collision_rate, success_rate, avg_smooth, avg_infer_t, avg_plan_t, avg_step_num

    # ============【新增CSV导出函数】============
    def save_metrics_to_csv(self, save_filepath, plot_interval):
        """导出指标到CSV，格式与DQN/A2C/原生PPO完全统一"""
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
                        val = self.metrics[k][idx]
                        row.append(val)
                    writer.writerow(row)
            print(f"✅ Metrics CSV saved to: {save_filepath}")
        except Exception as e:
            print(f"⚠️ Failed to save csv, error: {e}")

# ===================== 3D可视化工具 =====================
def cuboid_data(center, size=(1,1,1)):
    """生成立方体的顶点数据"""
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
    """在3D轴上绘制立方体"""
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

def plot_3d_path(env, episode, save_path):
    """绘制三维路径图，包含立方体障碍物可视化"""
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

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, MAP_SIZE)
    ax.set_ylim(0, MAP_SIZE)
    ax.set_zlim(0, MAP_SIZE)
    ax.set_title(f"3D Path Planning - Episode {episode}", fontsize=16, fontweight='bold')
    ax.legend()

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_training_curve(metrics):
    """训练曲线布局改为3行2列，容纳7项指标；新增Avg Steps/Episode独立图片"""
    plt.figure(figsize=(18, 12), dpi=100)
    index = np.arange(1, len(metrics["path_length"])+1) * PLOT_INTERVAL

    # 1. 路径长度
    plt.subplot(3, 2, 1)
    plt.plot(index, metrics["path_length"], color='#3498db', linewidth=2)
    plt.title("Average Path Length (3D)", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    # 2. 平均奖励
    plt.subplot(3, 2, 2)
    plt.plot(index, metrics["reward"], color='#e74c3c', linewidth=2)
    plt.title("Average Reward", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    # 3. 碰撞率
    plt.subplot(3, 2, 3)
    plt.plot(index, metrics["collision_rate"], color='#2c3e50', linewidth=2)
    plt.title("Collision Rate", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    # 4. 成功率
    plt.subplot(3, 2, 4)
    plt.plot(index, metrics["success_rate"], color='#27ae60', linewidth=2)
    plt.title("Success Rate", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    # 5. 路径平滑度
    plt.subplot(3, 2, 5)
    smooth_data = gaussian_filter1d(metrics["smoothness"], sigma=2)
    plt.plot(index, smooth_data, color='#f39c12', linewidth=2)
    plt.title("Path Smoothness (3D)", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    # 6. 单步网络推理耗时
    plt.subplot(3, 2, 6)
    infer_ms = np.array(metrics["infer_time"]) * 1000
    infer_smooth = gaussian_filter1d(infer_ms, sigma=2)
    plt.plot(index, infer_smooth, color='#9b59b6', linewidth=2)
    plt.title("Avg Single Step Inference Time (ms)", fontsize=12, fontweight='bold')
    plt.xlabel("Episode", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig(f"{SAVE_DIR}/training_metrics_3d.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 单独输出路径规划总耗时曲线
    plt.figure(figsize=(10, 5), dpi=100)
    plan_ms = np.array(metrics["total_plan_time"]) * 1000
    plan_smooth = gaussian_filter1d(plan_ms, sigma=2)
    plt.plot(index, plan_smooth, color='#d35400', linewidth=2)
    plt.title("Average Total Path Planning Time (ms)", fontsize=12, fontweight='bold')
    plt.xlabel("Episode")
    plt.ylabel("Time (ms)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/total_planning_time_curve.png", bbox_inches='tight')
    plt.close()

    # =========【新增】Avg Steps/Episode 独立迭代曲线图 =========
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
    print(f"✅ Total path planning time curve saved separately")
    print(f"✅ Avg Steps/Episode curve saved separately")

# ===================== 主训练流程 =====================
def main():
    """主训练函数"""
    # 计算三维状态维度
    local_obs_dim = (2*UNKNOWN_RADIUS+1)**3  # 三维局部观测维度
    state_dim = local_obs_dim + 6  # 6 = 3(归一化位置) + 3(相对目标位置)
    action_dim = 6  # 三维动作：x±, y±, z±

    # 初始化环境、智能体、指标计算器
    env = PathPlanningEnv3D()
    agent = PPOTrainer3D(state_dim, action_dim)
    metrics = MetricsCalculator3D()

    print("🚀 Start training PPO 3D path planning (LSTM+Self-Attention)...")
    print(f"📊 Save 3D path plot every {PLOT_INTERVAL} episodes\n")

    # 训练主循环
    for episode in range(1, EPISODES+1):
        state = env.reset()
        total_reward = 0.0
        done = False
        episode_infer_times = []
        step_counter = 0  # 【新增】统计当前episode交互步数

        plan_start = time.perf_counter() # ==========整条路径规划计时起点
        # 单轮episode的步骤循环
        for _ in range(STEPS_PER_EPISODE):
            action, log_prob, value, infer_t = agent.model.get_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            episode_infer_times.append(infer_t)

            agent.store_transition([state, action, log_prob, reward, done, value])

            state = next_state
            step_counter += 1  # 每执行一步计数+1
            if done:
                break
        plan_total_time = time.perf_counter() - plan_start # 整条路径规划总耗时(s)

        # 【修改】传入step_counter
        metrics.update(env.path, total_reward, env.collision, env.success, episode_infer_times, plan_total_time, step_counter)

        # 定期打印日志和绘图
        if episode % PLOT_INTERVAL == 0:
            avg_len, avg_rew, col_rate, suc_rate, avg_smooth, avg_infer, avg_plan, avg_steps = metrics.get_average()
            print(f"📌 Episode: {episode:4d} | Avg Length: {avg_len:6.2f} | Avg Reward: {avg_rew:7.2f} | Collision Rate: {col_rate:6.2%} | Success Rate: {suc_rate:6.2%}")
            print(f"    Smoothness: {avg_smooth:.2f} | Single Step Infer: {avg_infer*1000:.3f} ms | Total Plan Time: {avg_plan*1000:.3f} ms | Avg Steps/Episode: {avg_steps:.1f}")
            plot_3d_path(env, episode, f"{SAVE_DIR}/3d_path_episode_{episode}.png")

        # 更新策略网络
        agent.update()

    # 绘制最终训练曲线
    plot_training_curve(metrics.metrics)
    # ============【调用导出CSV】============
    metrics.save_metrics_to_csv(os.path.join(SAVE_DIR, "metrics_log.csv"), PLOT_INTERVAL)
    # 保存模型
    torch.save(agent.model.state_dict(), f"{SAVE_DIR}/ppo_3d_path_planning_model.pth")
    print(f"\n🎉 3D Training finished! All results saved to: {SAVE_DIR}")
    print(f"🤖 Model saved to: {SAVE_DIR}/ppo_3d_path_planning_model.pth")

if __name__ == "__main__":

    # 运行主训练函数
    main()