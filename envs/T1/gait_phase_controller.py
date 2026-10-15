"""
步态相位控制器
实时维护机器人步态相位状态,引导在正确相位执行踢球动作
"""

import torch
import numpy as np


class GaitPhaseController:
    """
    步态相位控制器
    维护周期性相位 φ ∈ [0, 1]:
    - [0.0-0.5]: 支撑相 (Stance Phase) - 保持重心稳定
    - [0.5-1.0]: 摆动相 (Swing Phase) - 执行踢球动作
    """

    def __init__(self, device, num_envs, frequency=1.0):
        """
        Args:
            device: torch设备
            num_envs: 环境数量
            frequency: 步态周期频率 (Hz)
        """
        self.device = device
        self.num_envs = num_envs
        self.frequency = frequency  # Hz
        self.period = 1.0 / frequency  # 秒

        # 当前相位 φ ∈ [0, 1]
        self.phase = torch.zeros(num_envs, device=device)

        # 相位速度 (每秒增加的相位)
        self.phase_velocity = torch.ones(num_envs, device=device) * frequency

        # 上一次脚触地状态
        self.prev_foot_contact = torch.zeros(num_envs, 2, device=device, dtype=torch.bool)

        print(f"⏱️ 步态相位控制器初始化完成")
        print(f"   步态频率: {frequency} Hz")
        print(f"   步态周期: {self.period:.2f} s")

    def update(self, dt):
        """
        更新相位

        Args:
            dt: 时间步长 (秒)
        """
        # 相位增量
        delta_phase = self.phase_velocity * dt

        # 更新相位并保持在 [0, 1]
        self.phase = (self.phase + delta_phase) % 1.0

    def reset(self, env_ids):
        """
        重置指定环境的相位

        Args:
            env_ids: 需要重置的环境ID
        """
        self.phase[env_ids] = 0.0
        self.prev_foot_contact[env_ids] = False

    def get_phase(self):
        """
        获取当前相位

        Returns:
            phase: [num_envs] 当前相位 φ ∈ [0, 1]
        """
        return self.phase

    def is_stance_phase(self):
        """
        判断是否处于支撑相 [0.0, 0.5]

        Returns:
            [num_envs] bool tensor
        """
        return self.phase < 0.5

    def is_swing_phase(self):
        """
        判断是否处于摆动相 [0.5, 1.0]

        Returns:
            [num_envs] bool tensor
        """
        return self.phase >= 0.5

    def get_kick_timing_weight(self):
        """
        获取踢球时机权重
        最佳踢球时机在相位 [0.85, 1.0]

        Returns:
            [num_envs] 权重 ∈ [0, 1]
        """
        # 使用高斯核,中心在0.9
        optimal_phase = 0.9
        sigma = 0.1
        weight = torch.exp(-((self.phase - optimal_phase) ** 2) / (2 * sigma ** 2))
        return weight

    def get_stance_weight(self):
        """
        获取支撑相稳定性权重
        在支撑相 [0.0, 0.5] 时权重为1,否则为0

        Returns:
            [num_envs] 权重 ∈ [0, 1]
        """
        return (self.phase < 0.5).float()

    def get_swing_weight(self):
        """
        获取摆动相权重
        在摆动相 [0.5, 1.0] 时权重为1,否则为0

        Returns:
            [num_envs] 权重 ∈ [0, 1]
        """
        return (self.phase >= 0.5).float()

    def update_foot_contact(self, foot_contact):
        """
        更新脚触地状态

        Args:
            foot_contact: [num_envs, 2] 左右脚触地状态
        """
        self.prev_foot_contact = foot_contact.clone()

    def compute_phase_rewards(self, foot_contact, foot_vel, kicking_foot_vel):
        """
        计算步态相位相关的奖励

        Args:
            foot_contact: [num_envs, 2] 左右脚触地状态
            foot_vel: [num_envs, 2, 3] 左右脚速度
            kicking_foot_vel: [num_envs, 3] 踢球脚速度

        Returns:
            dict: 各项相位相关奖励
        """
        rewards = {}

        # 1. 步态相位对齐奖励
        # 在支撑相时,脚应该接触地面;在摆动相时,踢球脚应该抬起
        stance_mask = self.is_stance_phase()
        swing_mask = self.is_swing_phase()

        # 支撑相: 两脚都应该接触地面
        both_feet_contact = foot_contact.all(dim=-1).float()
        stance_alignment = stance_mask.float() * both_feet_contact

        # 摆动相: 踢球脚应该离地
        kicking_foot_contact = foot_contact[:, 0]  # 假设左脚是踢球脚
        swing_alignment = swing_mask.float() * (~kicking_foot_contact).float()

        rewards['gait_phase_alignment'] = stance_alignment + swing_alignment

        # 2. 支撑相稳定性奖励
        # 在支撑相时,脚速度应该接近0
        foot_speed = torch.norm(foot_vel, dim=-1)  # [num_envs, 2]
        avg_foot_speed = foot_speed.mean(dim=-1)
        stance_stability = torch.exp(-avg_foot_speed / 0.1)
        rewards['stance_stability'] = stance_mask.float() * stance_stability

        # 3. 平滑相位过渡奖励
        # 相位接近0.5时,鼓励平滑过渡
        transition_distance = torch.abs(self.phase - 0.5)
        smooth_transition = torch.exp(-transition_distance / 0.1)
        rewards['smooth_phase_transition'] = smooth_transition

        # 4. 摆动相抬腿间隙奖励
        # 在摆动相时,踢球脚应该抬高
        kicking_foot_height_vel = kicking_foot_vel[:, 2]  # z方向速度
        swing_leg_clearance = torch.clamp(kicking_foot_height_vel, min=0.0)
        rewards['swing_leg_clearance'] = swing_mask.float() * swing_leg_clearance

        # 5. 接触时机奖励
        # 检测相位从摆动相回到支撑相时,脚是否正确接触地面
        phase_just_transitioned = (self.phase < 0.1).float()
        contact_correct = both_feet_contact
        rewards['contact_timing'] = phase_just_transitioned * contact_correct

        # 6. 踢球冲击时机奖励
        # 在最佳踢球时机 [0.85, 1.0] 时,踢球脚速度应该最大
        kick_timing_weight = self.get_kick_timing_weight()
        kicking_foot_speed = torch.norm(kicking_foot_vel, dim=-1)
        kick_impact = kicking_foot_speed * kick_timing_weight
        rewards['kick_impact_timing'] = kick_impact

        return rewards


def test_gait_phase_controller():
    """测试步态相位控制器"""
    print("=" * 50)
    print("测试步态相位控制器")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建控制器
    controller = GaitPhaseController(device, num_envs=4, frequency=1.0)

    # 模拟10秒的运行
    dt = 0.002  # 2ms
    num_steps = int(10.0 / dt)

    print(f"\n模拟 {num_steps} 步 (10秒)")

    # 记录相位历史
    phase_history = []

    for step in range(num_steps):
        controller.update(dt)

        # 每0.1秒记录一次
        if step % 50 == 0:
            phase_history.append(controller.get_phase()[0].item())

    # 打印相位历史
    print(f"\n相位变化 (环境0):")
    for i, phase in enumerate(phase_history[:20]):  # 只打印前2秒
        time = i * 0.1
        phase_type = "支撑相" if phase < 0.5 else "摆动相"
        print(f"  t={time:.1f}s: φ={phase:.3f} ({phase_type})")

    # 测试奖励计算
    print(f"\n测试奖励计算...")
    fake_foot_contact = torch.tensor([[True, True], [True, False], [False, False], [False, True]],
                                     device=device)
    fake_foot_vel = torch.randn(4, 2, 3, device=device) * 0.1
    fake_kicking_foot_vel = torch.randn(4, 3, device=device) * 0.5

    rewards = controller.compute_phase_rewards(fake_foot_contact, fake_foot_vel, fake_kicking_foot_vel)

    print(f"\n奖励结果:")
    for key, value in rewards.items():
        print(f"  {key}: mean={value.mean().item():.4f}, std={value.std().item():.4f}")

    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)


if __name__ == "__main__":
    test_gait_phase_controller()
