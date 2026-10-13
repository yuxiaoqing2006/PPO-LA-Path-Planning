import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import os
from scipy.ndimage import gaussian_filter1d
# 服务器无图形界面必备
plt.switch_backend('Agg')

# ===================== 全局参数设置 =====================
MAP_SIZE = 20
START = (1, 1)
GOAL = (18, 18)
UNKNOWN_RADIUS = 3
STEPS_PER_EPISODE = 200
TEST_EPISODES = 500
PLOT_INTERVAL = 50
SAVE_DIR = "./dwa_fixed_map_planning"
os.makedirs(SAVE_DIR, exist_ok=True)

# DWA运动学参数
MAX_V = 1.8
MAX_W = np.pi / 3
ACC_V = 0.6
ACC_W = np.pi / 6
DT = 0.1
PREDICT_TIME = 2.0
VELOCITY_RES = 0.02
OMEGA_RES = 0.008

# 提升避障、朝向权重，压低速度权重
WEIGHT_HEADING = 1.5
WEIGHT_OBST = 0.9
WEIGHT_SPEED = 0.6
SAFE_MARGIN = 0.4  # 机器人安全缓冲半径

# ===================== 环境类 =====================
class PathPlanningEnv:
    def __init__(self):
        self.map_size = MAP_SIZE
        self.start = np.array(START, dtype=np.float32)
        self.goal = np.array(GOAL, dtype=np.float32)
        self.unknown_radius = UNKNOWN_RADIUS
        self.reset()

    def generate_map(self):
        grid_map = np.zeros((self.map_size, self.map_size), dtype=np.int32)
        grid_map[0, :] = 1
        grid_map[-1, :] = 1
        grid_map[:, 0] = 1
        grid_map[:, -1] = 1

        # 删除起点附近x=1列障碍物，防止开局碰撞
        fixed_obstacles = [
            (17,10), (16,13), (16,14), (15,6), (15,18),
            (13,8), (13,11), (13,15), (13,17),
            (11,5), (10,5), (10,11), (10,14), (10,16), (10,17), (10,18),
            (8,4), (8,11), (8,14), (6,4),
            (5,5), (5,6), (5,7), (5,10), (5,12), (5,13), (5,15),
            (4,5), (4,6), (3,3), (3,13), (3,14), (2,5), (2,15), (2,18), (1,15)
        ]
        for (y, x) in fixed_obstacles:
            grid_map[y, x] = 1
        return grid_map

    def get_local_observation(self):
        x, y = self.pos
        local_obs = np.ones((2*self.unknown_radius+1, 2*self.unknown_radius+1), dtype=np.float32) * 0.5
        for dx in range(-self.unknown_radius, self.unknown_radius+1):
            for dy in range(-self.unknown_radius, self.unknown_radius+1):
                nx = int(round(x + dx))
                ny = int(round(y + dy))
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    local_obs[dy+self.unknown_radius, dx+self.unknown_radius] = self.grid_map[ny, nx]
        local_flat = local_obs.flatten()
        norm_pos = self.pos / self.map_size
        relative_goal = (self.goal - self.pos) / self.map_size
        state = np.concatenate([local_flat, norm_pos, relative_goal], dtype=np.float32)
        return state

    def reset(self):
        self.grid_map = self.generate_map()
        self.pos = self.start.copy()
        self.theta = np.random.uniform(0, np.pi/2)
        self.path = [self.pos.copy()]
        self.collision = False
        self.success = False
        self.v = 0.4  # 初始给基础速度，解决首帧无法转弯
        self.w = 0.0
        return self.get_local_observation()

    def is_pos_collision(self, pos):
        x, y = pos
        # 边界检测
        if x <= SAFE_MARGIN or x >= self.map_size - SAFE_MARGIN or y <= SAFE_MARGIN or y >= self.map_size - SAFE_MARGIN:
            return True
        # 遍历所有障碍，增加安全距离缓冲
        ys, xs = np.where(self.grid_map == 1)
        for ox, oy in zip(xs, ys):
            dist = np.hypot(x - ox - 0.5, y - oy - 0.5)
            if dist < SAFE_MARGIN:
                return True
        return False

    def step_dwa(self, v, w):
        self.theta += w * DT
        dx = v * np.cos(self.theta) * DT
        dy = v * np.sin(self.theta) * DT
        new_pos = self.pos + np.array([dx, dy], dtype=np.float32)
        done = False

        if self.is_pos_collision(new_pos):
            self.collision = True
            done = True
        else:
            self.pos = new_pos
            self.v = v
            self.w = w
            self.path.append(self.pos.copy())
            if np.linalg.norm(self.pos - self.goal) < 0.5:
                self.success = True
                done = True
        return self.get_local_observation(), done

# ===================== DWAPlanner =====================
class DWAPlanner:
    def __init__(self, env: PathPlanningEnv):
        self.env = env

    def calc_dynamic_window(self, v_now, w_now):
        v_min = max(0.0, v_now - ACC_V * DT)
        v_max = min(MAX_V, v_now + ACC_V * DT)
        w_min = max(-MAX_W, w_now - ACC_W * DT)
        w_max = min(MAX_W, w_now + ACC_W * DT)
        return v_min, v_max, w_min, w_max

    def predict_traj(self, v, w):
        traj = []
        x, y = self.env.pos
        theta = self.env.theta
        t = 0.0
        while t <= PREDICT_TIME:
            traj.append(np.array([x, y]))
            theta += w * DT
            x += v * np.cos(theta) * DT
            y += v * np.sin(theta) * DT
            t += DT
        return np.array(traj)

    def cost_heading(self, traj):
        end_p = traj[-1]
        target_ang = np.arctan2(self.env.goal[1] - end_p[1], self.env.goal[0] - end_p[0])
        traj_end_ang = np.arctan2(traj[-1,1] - traj[-2,1], traj[-1,0] - traj[-2,0])
        angle_diff = np.abs(np.arctan2(np.sin(target_ang - traj_end_ang), np.cos(target_ang - traj_end_ang)))
        return angle_diff

    def min_dist_obs(self, pos):
        x, y = pos
        min_d = float("inf")
        ys, xs = np.where(self.env.grid_map == 1)
        for ox, oy in zip(xs, ys):
            d = np.hypot(x - ox - 0.5, y - oy - 0.5)
            if d < min_d:
                min_d = d
        return min_d

    def cost_obstacle(self, traj):
        min_d = float("inf")
        for p in traj:
            if self.env.is_pos_collision(p):
                return float("inf")
            d = self.min_dist_obs(p)
            if d < min_d:
                min_d = d
        cost = np.exp(3.0 / (min_d + 1e-6))
        return cost

    def cost_speed(self, v):
        return MAX_V - v

    def calc_total_cost(self, v, w):
        traj = self.predict_traj(v, w)
        c_h = self.cost_heading(traj) * WEIGHT_HEADING
        c_o = self.cost_obstacle(traj) * WEIGHT_OBST
        c_v = self.cost_speed(v) * WEIGHT_SPEED
        total_cost = c_h + c_o + c_v
        return total_cost, traj

    def plan(self):
        v_now = self.env.v
        w_now = self.env.w
        v_min, v_max, w_min, w_max = self.calc_dynamic_window(v_now, w_now)
        min_cost = float("inf")
        best_v, best_w = 0.2, 0.0

        v_samples = np.arange(v_min, v_max, VELOCITY_RES)
        w_samples = np.arange(w_min, w_max, OMEGA_RES)
        for v in v_samples:
            for w in w_samples:
                cost, _ = self.calc_total_cost(v, w)
                if cost < min_cost:
                    min_cost = cost
                    best_v = v
                    best_w = w
        return best_v, best_w

# ===================== 指标计算器 =====================
class MetricsCalculator:
    def __init__(self):
        self.metrics = {
            "path_length": [], "collision_rate": [],
            "success_rate": [], "smoothness": []
        }
        self.episode_buffer = []

    def calculate_path_length(self, path):
        if len(path) < 2:
            return 0.0
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
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            norm = norm1 * norm2
            if norm < 1e-6:
                continue
            cos_theta = np.clip(np.dot(vec1, vec2) / norm, -1.0, 1.0)
            angle = np.arccos(cos_theta)
            smoothness += np.abs(angle)
        return smoothness

    def update(self, path, collision, success):
        length = self.calculate_path_length(path)
        smoothness = self.calculate_smoothness(path)
        self.episode_buffer.append([length, float(collision), float(success), smoothness])

    def get_average(self):
        if not self.episode_buffer:
            return [0.0]*4
        arr = np.array(self.episode_buffer)
        avg_len = np.mean(arr[:, 0])
        collision_rate = np.mean(arr[:, 1])
        success_rate = np.mean(arr[:, 2])
        avg_smooth = np.mean(arr[:, 3])

        self.metrics["path_length"].append(avg_len)
        self.metrics["collision_rate"].append(collision_rate)
        self.metrics["success_rate"].append(success_rate)
        self.metrics["smoothness"].append(avg_smooth)
        self.episode_buffer = []
        return avg_len, collision_rate, success_rate, avg_smooth

# ===================== 绘图函数 =====================
def plot_path(env, episode, save_path):
    plt.figure(figsize=(8, 8), dpi=100)
    ax = plt.gca()
    ys, xs = np.where(env.grid_map == 1)
    for x, y in zip(xs, ys):
        ax.add_patch(Rectangle((x, y), 1, 1, color='#2c3e50', alpha=0.8))
    sx, sy = START
    gx, gy = GOAL
    ax.add_patch(Rectangle((sx, sy), 1, 1, color='#27ae60', label='Start'))
    ax.add_patch(Circle((gx+0.5, gy+0.5), 0.4, color='#e74c3c', label='Goal'))
    path = np.array(env.path)
    plt.plot(path[:, 0]+0.5, path[:, 1]+0.5, color='#3498db', linewidth=2.5, label='DWA Planned Path')
    plt.xlim(0, MAP_SIZE)
    plt.ylim(0, MAP_SIZE)
    ax.set_aspect('equal')
    plt.title(f"DWA Path Planning - Episode {episode}", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_training_curve(metrics):
    plt.figure(figsize=(12, 8), dpi=100)
    index = np.arange(1, len(metrics["path_length"])+1) * PLOT_INTERVAL

    plt.subplot(2, 2, 1)
    plt.plot(index, metrics["path_length"], color='#3498db', linewidth=2)
    plt.title("Average Path Length", fontsize=12, fontweight='bold')
    plt.xlabel("Episode")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(index, metrics["collision_rate"], color='#2c3e50', linewidth=2)
    plt.title("Collision Rate", fontsize=12, fontweight='bold')
    plt.xlabel("Episode")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(index, metrics["success_rate"], color='#27ae60', linewidth=2)
    plt.title("Success Rate", fontsize=12, fontweight='bold')
    plt.xlabel("Episode")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    smooth_data = gaussian_filter1d(metrics["smoothness"], sigma=2)
    plt.plot(index, smooth_data, color='#f39c12', linewidth=2)
    plt.title("Path Smoothness (Sum Turning Angle)", fontsize=12, fontweight='bold')
    plt.xlabel("Episode")
    plt.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig(f"{SAVE_DIR}/dwa_training_metrics.png", bbox_inches='tight')
    plt.close()
    print(f"✅ DWA metrics curve saved to {SAVE_DIR}/dwa_training_metrics.png")

# ===================== 主函数 =====================
def main():
    env = PathPlanningEnv()
    dwa_planner = DWAPlanner(env)
    metrics = MetricsCalculator()

    print("🚀 Start Fixed DWA Path Planning Simulation")
    print(f"📊 Save path figure every {PLOT_INTERVAL} episodes\n")

    for episode in range(1, TEST_EPISODES + 1):
        env.reset()
        done = False

        for step in range(STEPS_PER_EPISODE):
            v_opt, w_opt = dwa_planner.plan()
            _, done = env.step_dwa(v_opt, w_opt)
            if done:
                break

        metrics.update(env.path, env.collision, env.success)

        if episode % PLOT_INTERVAL == 0:
            avg_len, col_rate, suc_rate, avg_smooth = metrics.get_average()
            print(f"📌 Ep {episode:4d} | Avg Len: {avg_len:6.2f} | Collision: {col_rate:6.2%} | Success: {suc_rate:6.2%} | Smooth: {avg_smooth:.2f}")
            plot_path(env, episode, f"{SAVE_DIR}/dwa_path_ep_{episode}.png")

    plot_training_curve(metrics.metrics)
    print(f"\n🎉 DWA Simulation Finished! Output dir: {SAVE_DIR}")

if __name__ == "__main__":
    main()