import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ===================== 1. 数据准备 (请填入你的实验数据) =====================
# 假设你已经运行了三种算法，获得了以下 metrics 字典结构
# metrics = {"path_length": [], "reward": [], "collision_rate": [], "success_rate": [], "smoothness": []}

# -------- 请在这里替换为你的实际运行结果 --------
# 示例数据（请替换为你实际跑出来的数组）
plot_interval = 100  # 与实验保持一致，每多少episode取一次平均

# 1. PPO 数据 (替换为你PPO实验的metrics)
ppo_metrics = {
    "path_length": np.array([170, 160, 150, 120, 100, 95, 90, 88, 85, 85, 84]), # 示例：2500/100 = 11个点
    "reward": np.array([-0.5, 0.5, 1.0, 1.5, 1.8, 2.0, 1.9, 2.1, 2.0, 1.8, 1.9]),
    "collision_rate": np.array([0.6, 0.5, 0.4, 0.2, 0.1, 0.05, 0.05, 0.03, 0.02, 0.02, 0.01]),
    "success_rate": np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.9, 0.92, 0.93, 0.95, 0.96]),
    "smoothness": np.array([250, 230, 210, 190, 170, 160, 155, 150, 145, 145, 144])
}

# 2. DQN 数据 (替换为你DQN实验的metrics)
dqn_metrics = {
    "path_length": np.array([180, 170, 160, 130, 110, 100, 110, 120, 130, 140, 150]),
    "reward": np.array([-1.0, -0.8, -0.5, 0.0, 0.5, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2]),
    "collision_rate": np.array([0.7, 0.6, 0.5, 0.3, 0.1, 0.05, 0.08, 0.15, 0.2, 0.25, 0.3]),
    "success_rate": np.array([0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.65, 0.5, 0.4, 0.35, 0.3]),
    "smoothness": np.array([240, 220, 200, 180, 160, 150, 160, 170, 180, 190, 200])
}

# 3. A2C 数据 (替换为你当前这张图对应的A2C metrics)
a2c_metrics = {
    "path_length": np.array([175, 170, 165, 150, 130, 120, 140, 160, 150, 170, 160]),
    "reward": np.array([-0.9, -0.8, -0.7, -0.6, -0.5, -0.3, -0.5, -0.7, -0.6, -0.8, -0.9]),
    "collision_rate": np.array([1.0, 1.0, 1.0, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 1.0]),
    "success_rate": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01]),
    "smoothness": np.array([230, 220, 210, 200, 190, 180, 170, 165, 170, 180, 190])
}

# 横坐标刻度 (Episode)
episodes = np.arange(plot_interval, len(ppo_metrics["path_length"])*plot_interval + 1, plot_interval)

# ===================== 2. 绘图设置 (专业学术风格) =====================
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(18, 10)) # 2行3列，留一个空位保持5图布局
fig.delaxes(axes[1,2]) # 删除多余的第6个格子

# 颜色与标签配置 (专业配色)
colors = {
    'PPO': '#27ae60',    # 绿色
    'DQN': '#e74c3c',    # 红色
    'A2C': '#f39c12'     # 橙色
}

# ===================== 3. 绘制对比曲线 =====================

# 1. Average Path Length (左上)
ax1 = axes[0, 0]
ax1.plot(episodes, ppo_metrics["path_length"], label='PPO', color=colors['PPO'], linewidth=2.5, marker='o', markersize=4)
ax1.plot(episodes, dqn_metrics["path_length"], label='DQN', color=colors['DQN'], linewidth=2.5, marker='s', markersize=4)
ax1.plot(episodes, a2c_metrics["path_length"], label='A2C', color=colors['A2C'], linewidth=2.5, marker='^', markersize=4)
ax1.set_title('Average Path Length', fontsize=14, fontweight='bold')
ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Length', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 2. Average Reward (中上)
ax2 = axes[0, 1]
ax2.plot(episodes, ppo_metrics["reward"], label='PPO', color=colors['PPO'], linewidth=2.5, marker='o', markersize=4)
ax2.plot(episodes, dqn_metrics["reward"], label='DQN', color=colors['DQN'], linewidth=2.5, marker='s', markersize=4)
ax2.plot(episodes, a2c_metrics["reward"], label='A2C', color=colors['A2C'], linewidth=2.5, marker='^', markersize=4)
ax2.set_title('Average Reward', fontsize=14, fontweight='bold')
ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Reward', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# 3. Collision Rate (右上)
ax3 = axes[0, 2]
ax3.plot(episodes, ppo_metrics["collision_rate"], label='PPO', color=colors['PPO'], linewidth=2.5, marker='o', markersize=4)
ax3.plot(episodes, dqn_metrics["collision_rate"], label='DQN', color=colors['DQN'], linewidth=2.5, marker='s', markersize=4)
ax3.plot(episodes, a2c_metrics["collision_rate"], label='A2C', color=colors['A2C'], linewidth=2.5, marker='^', markersize=4)
ax3.set_title('Collision Rate', fontsize=14, fontweight='bold')
ax3.set_xlabel('Episode', fontsize=12)
ax3.set_ylabel('Collision Probability', fontsize=12)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 1.05) # 限制范围0-1

# 4. Success Rate (左下)
ax4 = axes[1, 0]
ax4.plot(episodes, ppo_metrics["success_rate"], label='PPO', color=colors['PPO'], linewidth=2.5, marker='o', markersize=4)
ax4.plot(episodes, dqn_metrics["success_rate"], label='DQN', color=colors['DQN'], linewidth=2.5, marker='s', markersize=4)
ax4.plot(episodes, a2c_metrics["success_rate"], label='A2C', color=colors['A2C'], linewidth=2.5, marker='^', markersize=4)
ax4.set_title('Success Rate', fontsize=14, fontweight='bold')
ax4.set_xlabel('Episode', fontsize=12)
ax4.set_ylabel('Success Rate', fontsize=12)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 1.05) # 限制范围0-1

# 5. Path Smoothness (右下)
ax5 = axes[1, 1]
# 平滑处理 (可选，看数据波动程度)
ppo_smooth = gaussian_filter1d(ppo_metrics["smoothness"], sigma=0.8)
dqn_smooth = gaussian_filter1d(dqn_metrics["smoothness"], sigma=0.8)
a2c_smooth = gaussian_filter1d(a2c_metrics["smoothness"], sigma=0.8)

ax5.plot(episodes, ppo_smooth, label='PPO', color=colors['PPO'], linewidth=2.5, marker='o', markersize=4)
ax5.plot(episodes, dqn_smooth, label='DQN', color=colors['DQN'], linewidth=2.5, marker='s', markersize=4)
ax5.plot(episodes, a2c_smooth, label='A2C', color=colors['A2C'], linewidth=2.5, marker='^', markersize=4)
ax5.set_title('Path Smoothness', fontsize=14, fontweight='bold')
ax5.set_xlabel('Episode', fontsize=12)
ax5.set_ylabel('Smoothness Score', fontsize=12)
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3)

# ===================== 4. 保存图片 =====================
plt.tight_layout(pad=2.0)
plt.savefig('./comparison_ppo_dqn_a2c.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ 对比图已生成并保存至 ./comparison_ppo_dqn_a2cc.png")