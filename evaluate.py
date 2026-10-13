"""
评估脚本
用于评估训练好的模型在测试集上的性能
"""

import os
import argparse
import torch
import numpy as np
from collections import defaultdict


def evaluate(args):
    """评估模型性能"""

    print("=" * 70)
    print(f"开始评估: {args.checkpoint}")
    print("=" * 70)

    # 创建环境（需要根据实际项目调整）
    # env = make_env(args.task, num_envs=args.num_envs)

    # 加载模型（需要根据实际项目调整）
    # model = load_checkpoint(args.checkpoint)

    # 评估指标
    metrics = defaultdict(list)

    print(f"\n运行 {args.num_episodes} 个评估episodes...")

    for episode in range(args.num_episodes):
        # obs = env.reset()
        episode_reward = 0
        episode_length = 0
        ball_kicked = False
        kick_success = False
        standing_after_kick = False

        # 模拟episode（实际使用时取消注释并删除模拟数据部分）
        # while not done:
        #     action = model.act(obs)
        #     obs, reward, done, info = env.step(action)
        #
        #     episode_reward += reward
        #     episode_length += 1
        #
        #     # 检测踢球事件
        #     if info.get('ball_speed', 0) > 1.0:
        #         ball_kicked = True
        #         kick_success = True
        #
        #     # 检测站立稳定性
        #     if ball_kicked and info.get('base_height', 0) > 0.6:
        #         standing_after_kick = True

        # 模拟数据（实际使用时删除这部分）
        episode_reward = np.random.uniform(0, 100)
        episode_length = np.random.randint(50, 200)
        kick_success = np.random.random() > 0.3
        standing_after_kick = np.random.random() > 0.4

        # 记录指标
        metrics['episode_reward'].append(episode_reward)
        metrics['episode_length'].append(episode_length)
        metrics['kick_success'].append(1.0 if kick_success else 0.0)
        metrics['standing_success'].append(1.0 if standing_after_kick else 0.0)

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{args.num_episodes}")

    # 打印统计结果
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)

    print(f"\n平均回报: {np.mean(metrics['episode_reward']):.2f} ± {np.std(metrics['episode_reward']):.2f}")
    print(f"平均episode长度: {np.mean(metrics['episode_length']):.2f}")
    print(f"踢中率: {np.mean(metrics['kick_success']) * 100:.1f}%")
    print(f"站立保持成功率: {np.mean(metrics['standing_success']) * 100:.1f}%")

    # 保存结果
    if args.save_results:
        results_path = f"evaluation_results_{os.path.basename(args.checkpoint)}.txt"
        with open(results_path, 'w') as f:
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"平均回报: {np.mean(metrics['episode_reward']):.2f}\n")
            f.write(f"踢中率: {np.mean(metrics['kick_success']) * 100:.1f}%\n")
            f.write(f"站立保持成功率: {np.mean(metrics['standing_success']) * 100:.1f}%\n")
        print(f"\n✅ 结果已保存到: {results_path}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="T1/Kicking", help="任务名称")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--num_episodes", type=int, default=100, help="评估episode数量")
    parser.add_argument("--num_envs", type=int, default=1, help="并行环境数")
    parser.add_argument("--save_results", action="store_true", help="是否保存结果")

    args = parser.parse_args()
    evaluate(args)