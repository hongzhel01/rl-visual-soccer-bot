import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, num_act, num_obs, num_privileged_obs):
        super().__init__()
        
        # ---------------------------------------------------------
        # 1. 维度定义 (请确保这与你的 kicking.py 拼接顺序一致)
        # ---------------------------------------------------------
        self.base_obs_dim = 44   # 本体感觉 (Proprioception)
        self.visual_obs_dim = 52 # 视觉特征 (Leaky Top-K features)
        self.num_act = num_act
        
        # 内部检查：总维度是否匹配
        if (self.base_obs_dim + self.visual_obs_dim) != num_obs:
            print(f"⚠️ 警告: model.py 定义的总维度 ({self.base_obs_dim + self.visual_obs_dim}) "
                  f"与 Runner 传入的 num_obs ({num_obs}) 不符！")

        # ---------------------------------------------------------
        # 2. 视觉特征编码器 (Visual Encoder)
        # ---------------------------------------------------------
        self.visual_encoder = nn.Sequential(
            nn.Linear(self.visual_obs_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU()
        )

        # ---------------------------------------------------------
        # 3. Actor 网络 (策略网络)
        # ---------------------------------------------------------
        # 输入 = 基础 44 + 编码后视觉 64 = 108
        self.actor = nn.Sequential(
            nn.Linear(self.base_obs_dim + 64, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, num_act),
        )

        # ---------------------------------------------------------
        # 4. Critic 网络 (价值网络)
        # ---------------------------------------------------------
        # Critic 输入 = 所有观测 (96) + 所有特权观测 (72) = 168
        self.critic = nn.Sequential(
            nn.Linear(num_obs + num_privileged_obs, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

        # ---------------------------------------------------------
        # 5. 探索参数 (LogStd)
        # ---------------------------------------------------------
        # 提高初始标准差到 -1.2 (约 0.3)，增加初期视觉不确定时的探索能力
        self.logstd = nn.Parameter(torch.full((1, num_act), fill_value=-1.2), requires_grad=True)

    def _split_obs(self, obs):
        """
        核心修复点：
        使用 [..., :dim] 确保无论输入是 (env_num, 96) 还是 (batch, 96)，
        都能在最后一个维度正确切分。
        """
        base = obs[..., :self.base_obs_dim]
        visual = obs[..., self.base_obs_dim:]
        return base, visual

    def act(self, obs):
        # 1. 拆分数据
        base, visual = self._split_obs(obs)
        
        # 维度防御检查 (运行时)
        if visual.shape[-1] != self.visual_obs_dim:
            raise RuntimeError(f"视觉切片维度错误！预期 {self.visual_obs_dim}, 实际得到 {visual.shape[-1]}. "
                               f"总输入维度为 {obs.shape[-1]}. 请检查切片逻辑。")

        # 2. 编码视觉
        visual_encoded = self.visual_encoder(visual)
        
        # 3. 融合并输出动作均值
        combined_obs = torch.cat([base, visual_encoded], dim=-1)
        action_mean = self.actor(combined_obs)
        
        # 4. 构建正态分布
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        return torch.distributions.Normal(action_mean, action_std)

    def get_action(self, obs, deterministic=False):
        """Play 模式使用的便捷函数"""
        dist = self.act(obs)
        if deterministic:
            return dist.mode
        return dist.sample()

    def est_value(self, obs, privileged_obs):
        """Critic 价值估算"""
        # 直接拼接所有可用信息，无需切片
        critic_input = torch.cat((obs, privileged_obs), dim=-1)
        return self.critic(critic_input).squeeze(-1)

    def evaluate(self, obs, privileged_obs, action):
        """用于 PPO 更新时的评估"""
        dist = self.act(obs)
        action_log_probs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        value = self.est_value(obs, privileged_obs)
        return action_log_probs, dist_entropy, value