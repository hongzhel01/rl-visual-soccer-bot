"""
课程学习管理器
实现四阶段递进式训练：视觉寻球 → 稳定接近 → 精准射门 → 快速恢复
"""

import torch
import numpy as np


class CurriculumManager:
    """
    课程学习管理器
    根据训练进度自动调整任务难度和奖励权重
    """

    def __init__(self, cfg, num_envs, device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # 当前训练阶段 (1-4)
        self.stage = 1

        # 阶段成功计数器
        self.stage_success_count = torch.zeros(num_envs, dtype=torch.long, device=device)

        # 阶段切换阈值（连续成功N次才升级）
        self.stage_threshold = cfg.get("curriculum_learning", {}).get("success_threshold", 100)

        # 是否启用课程学习
        self.enabled = cfg.get("curriculum_learning", {}).get("enabled", False)

        # 各阶段配置
        self.stage_configs = self._init_stage_configs()

        print(f"🎓 课程学习管理器初始化完成")
        print(f"   启用状态: {self.enabled}")
        print(f"   阶段切换阈值: {self.stage_threshold}")

    def _init_stage_configs(self):
        """初始化各阶段配置"""
        return {
            1: {  # 视觉寻球
                "name": "visual_tracking",
                "ball_distance_range": [0.3, 0.5],
                "ball_angle_range": [-30, 30],
                "reward_scale_overrides": {
                    "ball_visible_align": 5.0,
                    "survival": 1.0,
                    "tracking_lin_vel_x": 0.1,
                    "kicking_foot_approach_ball_stationary": 0.5
                }
            },
            2: {  # 稳定接近
                "name": "stable_approach",
                "ball_distance_range": [0.5, 1.5],
                "ball_angle_range": [-45, 45],
                "reward_scale_overrides": {
                    "kicking_foot_approach_ball_stationary": 10.0,
                    "body_alignment_for_kick": 2.0,
                    "orientation": -50.0,
                    "ball_visible_align": 1.0
                }
            },
            3: {  # 精准射门
                "name": "accurate_kick",
                "ball_distance_range": [0.3, 0.5],
                "ball_angle_range": [-30, 30],
                "reward_scale_overrides": {
                    "ball_velocity_target_direction": 20.0,
                    "ball_acceleration": 10.0,
                    "kicking_foot_approach_ball_stationary": 5.0
                }
            },
            4: {  # 快速恢复
                "name": "post_kick_recovery",
                "ball_distance_range": [0.3, 0.5],
                "ball_angle_range": [-30, 30],
                "reward_scale_overrides": {
                    "post_kick_balance_recovery": 15.0,
                    "orientation": -100.0,
                    "base_height": -500.0,
                    "survival": 2.0
                }
            }
        }

    def check_stage_success(self, env_id, ball_pos, base_pos, ball_visible,
                           ball_center_offset, projected_gravity, ball_lin_vel):
        """
        检查指定环境是否完成当前阶段目标

        Returns:
            bool: 是否成功
        """
        if not self.enabled:
            return False

        if self.stage == 1:  # 视觉寻球
            return ball_visible[env_id] > 0.5 and ball_center_offset[env_id] < 0.2

        elif self.stage == 2:  # 稳定接近
            ball_dist = torch.norm(ball_pos[env_id] - base_pos[env_id])
            orientation_error = projected_gravity[env_id, :2].square().sum()
            return ball_dist < 0.25 and orientation_error < 0.01

        elif self.stage == 3:  # 精准射门
            ball_speed = torch.norm(ball_lin_vel[env_id])
            return ball_speed > 1.0

        elif self.stage == 4:  # 快速恢复
            ball_is_moving = torch.norm(ball_lin_vel[env_id]) > 0.5
            orientation_error = projected_gravity[env_id, :2].square().sum()
            height_ok = base_pos[env_id, 2] > 0.6
            return ball_is_moving and orientation_error < 0.02 and height_ok

        return False

    def update_stage(self, env_ids):
        """
        更新阶段进度（在episode结束时调用）

        Args:
            env_ids: 需要更新的环境ID列表
        """
        if not self.enabled:
            return

        for env_id in env_ids:
            # 重置成功计数
            self.stage_success_count[env_id] = 0

    def increment_success(self, env_id):
        """增加成功计数"""
        if not self.enabled:
            return

        self.stage_success_count[env_id] += 1

        # 检查是否需要升级阶段
        if self.stage_success_count[env_id] >= self.stage_threshold:
            if self.stage < 4:
                self.stage += 1
                self.stage_success_count[:] = 0  # 所有环境重置
                print(f"🎓 课程升级到阶段 {self.stage}: {self.stage_configs[self.stage]['name']}")

    def get_ball_init_config(self):
        """获取当前阶段的球初始化配置"""
        if not self.enabled:
            return None

        config = self.stage_configs[self.stage]
        return {
            "distance_range": config["ball_distance_range"],
            "angle_range": config["ball_angle_range"]
        }

    def get_reward_scale_overrides(self):
        """获取当前阶段的奖励权重覆盖"""
        if not self.enabled:
            return {}

        return self.stage_configs[self.stage]["reward_scale_overrides"]

    def get_stage_info(self):
        """获取当前阶段信息（用于日志）"""
        return {
            "stage": self.stage,
            "stage_name": self.stage_configs[self.stage]["name"],
            "progress": self.stage_success_count.float().mean().item(),
            "threshold": self.stage_threshold
        }