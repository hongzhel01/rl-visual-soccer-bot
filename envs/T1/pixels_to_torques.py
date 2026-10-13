"""
端到端像素到力矩策略网络
直接从原始图像输出关节力矩命令
"""

import torch
import torch.nn as nn


class PixelsToTorquesPolicy(nn.Module):
    """
    端到端策略网络
    输入：原始灰度图像 + 本体感知观测
    输出：关节动作
    """

    def __init__(self, action_dim=12, hidden_dim=256):
        super().__init__()

        # 视觉编码器（轻量级CNN）
        self.visual_encoder = nn.Sequential(
            # Input: [B, 1, 120, 160]
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # -> [B, 32, 60, 80]
            nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> [B, 64, 30, 40]
            nn.ELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # -> [B, 128, 15, 20]
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> [B, 128, 4, 4]
            nn.Flatten(),  # -> [B, 2048]
            nn.Linear(2048, 256),
            nn.ELU()
        )

        # 本体感知编码器
        self.proprio_encoder = nn.Sequential(
            nn.Linear(44, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU()
        )

        # 融合层 + Actor
        self.actor = nn.Sequential(
            nn.Linear(256 + 128, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic（用于PPO训练）
        self.critic = nn.Sequential(
            nn.Linear(256 + 128, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, raw_image, proprio_obs):
        """
        Args:
            raw_image: [B, 1, 120, 160] 原始灰度图
            proprio_obs: [B, 44] 本体感知观测

        Returns:
            actions: [B, action_dim] 关节动作
            value: [B, 1] 状态价值
        """
        visual_features = self.visual_encoder(raw_image)
        proprio_features = self.proprio_encoder(proprio_obs)

        fused = torch.cat([visual_features, proprio_features], dim=-1)

        actions = self.actor(fused)
        value = self.critic(fused)

        return actions, value

    def act(self, raw_image, proprio_obs):
        """推理时使用"""
        actions, _ = self.forward(raw_image, proprio_obs)
        return actions

    def evaluate(self, raw_image, proprio_obs):
        """评估状态价值"""
        _, value = self.forward(raw_image, proprio_obs)
        return value


def test_pixels_to_torques():
    """测试端到端网络"""
    print("=" * 50)
    print("测试端到端像素到力矩网络")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建模型
    model = PixelsToTorquesPolicy(action_dim=12, hidden_dim=256).to(device)

    # 模拟输入
    fake_image = torch.randn(4, 1, 120, 160, device=device, requires_grad=True)
    fake_proprio = torch.randn(4, 44, device=device)

    # 前向传播
    actions, value = model(fake_image, fake_proprio)

    print(f"\n✅ 前向传播成功")
    print(f"   输入图像形状: {fake_image.shape}")
    print(f"   输入本体感知形状: {fake_proprio.shape}")
    print(f"   输出动作形状: {actions.shape}")
    print(f"   输出价值形状: {value.shape}")

    # 反向传播测试
    loss = actions.sum() + value.sum()
    loss.backward()

    if fake_image.grad is not None:
        print(f"\n✅ 梯度成功回传到图像")
        print(f"   图像梯度范围: [{fake_image.grad.min():.6f}, {fake_image.grad.max():.6f}]")
    else:
        print(f"\n❌ 梯度未回传到图像")

    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)


if __name__ == "__main__":
    test_pixels_to_torques()