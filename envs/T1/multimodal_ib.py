"""
多模态信息瓶颈网络
实现视觉特征和本体感知特征的融合，通过互信息约束过滤冗余信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InformationBottleneckLayer(nn.Module):
    """
    变分信息瓶颈层
    实现 min I(X; Z) - β·I(Z; Y)
    其中 X 是输入特征，Z 是瓶颈表示，Y 是任务目标
    """

    def __init__(self, input_dim, bottleneck_dim, beta=0.01):
        """
        Args:
            input_dim: 输入维度
            bottleneck_dim: 瓶颈维度
            beta: KL散度权重
        """
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.beta = beta

        # 编码器：输入 → 均值和方差
        self.encoder_mu = nn.Linear(input_dim, bottleneck_dim)
        self.encoder_logvar = nn.Linear(input_dim, bottleneck_dim)

    def forward(self, x, deterministic=False):
        """
        Args:
            x: 输入特征 [batch, input_dim]
            deterministic: 是否使用确定性编码（测试时）

        Returns:
            z: 瓶颈表示 [batch, bottleneck_dim]
            kl_loss: KL散度损失
        """
        # 计算均值和方差
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)

        if deterministic:
            # 测试时使用均值
            return mu, torch.tensor(0.0, device=x.device)

        # 重参数化技巧: z = μ + σ * ε
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # KL散度损失: KL(q(z|x) || p(z))，其中 p(z) = N(0, I)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_loss = kl_loss.mean() * self.beta

        return z, kl_loss


class MultimodalInformationBottleneck(nn.Module):
    """
    多模态信息瓶颈网络
    融合视觉特征和本体感知特征，过滤冗余信息
    """

    def __init__(self, visual_dim, proprioceptive_dim, bottleneck_dim, output_dim, beta=0.01):
        """
        Args:
            visual_dim: 视觉特征维度
            proprioceptive_dim: 本体感知特征维度
            bottleneck_dim: 瓶颈维度
            output_dim: 输出维度
            beta: KL散度权重
        """
        super().__init__()

        # 视觉分支
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU()
        )
        self.visual_ib = InformationBottleneckLayer(64, bottleneck_dim // 2, beta=beta)

        # 本体感知分支
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprioceptive_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU()
        )
        self.proprio_ib = InformationBottleneckLayer(32, bottleneck_dim // 2, beta=beta * 0.5)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(bottleneck_dim, output_dim),
            nn.ELU()
        )

    def forward(self, visual_features, proprio_features, deterministic=False):
        """
        Args:
            visual_features: [batch, visual_dim]
            proprio_features: [batch, proprioceptive_dim]
            deterministic: 是否使用确定性编码

        Returns:
            fused_features: [batch, output_dim]
            total_kl_loss: 总的KL散度损失
        """
        # 编码
        visual_encoded = self.visual_encoder(visual_features)
        proprio_encoded = self.proprio_encoder(proprio_features)

        # 信息瓶颈
        visual_compressed, visual_kl = self.visual_ib(visual_encoded, deterministic)
        proprio_compressed, proprio_kl = self.proprio_ib(proprio_encoded, deterministic)

        # 融合
        bottleneck_features = torch.cat([visual_compressed, proprio_compressed], dim=-1)
        fused_features = self.fusion(bottleneck_features)

        total_kl_loss = visual_kl + proprio_kl

        return fused_features, total_kl_loss


def verify_gradient_flow():
    """验证梯度从输出反向传播到输入"""
    print("=" * 50)
    print("验证多模态信息瓶颈网络的梯度回传")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建模型
    model = MultimodalInformationBottleneck(
        visual_dim=52,
        proprioceptive_dim=44,
        bottleneck_dim=64,
        output_dim=52
    ).to(device)

    # 模拟输入
    fake_visual = torch.randn(10, 52, device=device, requires_grad=True)
    fake_proprio = torch.randn(10, 44, device=device, requires_grad=True)

    # 前向传播
    fused, kl_loss = model(fake_visual, fake_proprio, deterministic=False)

    # 模拟损失
    task_loss = fused.sum()
    total_loss = task_loss + kl_loss

    # 反向传播
    total_loss.backward()

    # 检查梯度
    print(f"\n✅ 前向传播成功")
    print(f"   融合特征形状: {fused.shape}")
    print(f"   任务损失: {task_loss.item():.4f}")
    print(f"   KL损失: {kl_loss.item():.4f}")
    print(f"   总损失: {total_loss.item():.4f}")

    if fake_visual.grad is not None:
        print(f"\n✅ 梯度成功回传到视觉特征")
        print(f"   视觉梯度范围: [{fake_visual.grad.min():.6f}, {fake_visual.grad.max():.6f}]")
        print(f"   视觉梯度均值: {fake_visual.grad.mean():.6f}")
    else:
        print(f"\n❌ 梯度未回传到视觉特征")

    if fake_proprio.grad is not None:
        print(f"\n✅ 梯度成功回传到本体感知特征")
        print(f"   本体感知梯度范围: [{fake_proprio.grad.min():.6f}, {fake_proprio.grad.max():.6f}]")
        print(f"   本体感知梯度均值: {fake_proprio.grad.mean():.6f}")
    else:
        print(f"\n❌ 梯度未回传到本体感知特征")

    print("\n" + "=" * 50)
    print("验证完成！")
    print("=" * 50)


if __name__ == "__main__":
    verify_gradient_flow()