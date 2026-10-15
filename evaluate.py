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

    # TODO: 创建环境（需要根据实际项目调整）
    # from envs.T1.kicking import Kicking
    # env = Kicking(load_config(args.task))
    print("\n⚠️  警告: 环境创建代码需要取消注释并根据实际项目调整")

    # TODO: 加载模型（需要根据实际项目调整）
    # model = torch.load(args.checkpoint)
    # model.eval()
    print("⚠️  警告: 模型加载代码需要取消注释并根据实际项目调整")

    # 评估指标
    metrics = defaultdict(list)

    print(f"\n运行 {args.num_episodes} 个评估episodes...")

    for episode in range(args.num_episodes):
        # TODO: 实际评估逻辑（取消注释下面的代码）
        # obs = env.reset()
        episode_reward = 0
        episode_length = 0
        ball_kicked = False
        kick_success = False
        standing_after_kick = False

        # 实际使用时取消注释这部分
        # with torch.no_grad():
        #     done = False
        #     while not done:
        #         # 获取动作
        #         action = model.act(obs)
        #
        #         # 执行动作
        #         obs, reward, done, info = env.step(action)
        #
        #         episode_reward += reward.item()
        #         episode_length += 1
        #
        #         # 检测踢球事件
        #         ball_speed = info.get('ball_speed', 0)
        #         if ball_speed > 1.0:
        #             ball_kicked = True
        #             kick_success = True
        #
        #         # 检测站立稳定性（踢球后机器人是否站立）
        #         if ball_kicked and info.get('base_height', 0) > 0.6:
        #             standing_after_kick = True

        # 模拟数据（实际使用时删除这部分）
        episode_reward = np.random.uniform(50, 150)
        episode_length = np.random.randint(50, 200)
        kick_success = np.random.random() > 0.3
        standing_after_kick = np.random.random() > 0.4 if kick_success else False

        # 记录指标
        metrics['episode_reward'].append(episode_reward)
        metrics['episode_length'].append(episode_length)
        metrics['kick_success'].append(1.0 if kick_success else 0.0)
        metrics['standing_success'].append(1.0 if standing_after_kick else 0.0)

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{args.num_episodes} 完成")

    # 打印统计结果
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)

    avg_reward = np.mean(metrics['episode_reward'])
    std_reward = np.std(metrics['episode_reward'])
    avg_length = np.mean(metrics['episode_length'])
    kick_rate = np.mean(metrics['kick_success']) * 100
    stand_rate = np.mean(metrics['standing_success']) * 100

    print(f"\n平均回报: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"平均episode长度: {avg_length:.2f}")
    print(f"踢中率: {kick_rate:.1f}%")
    print(f"站立保持成功率: {stand_rate:.1f}%")

    # 保存结果
    if args.save_results:
        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)

        checkpoint_name = os.path.basename(args.checkpoint).replace('.pt', '')
        results_path = os.path.join(results_dir, f"results_{checkpoint_name}.txt")

        with open(results_path, 'w', encoding='utf-8') as f:
            f.write(f"检查点: {args.checkpoint}\n")
            f.write(f"任务: {args.task}\n")
            f.write(f"评估episodes数: {args.num_episodes}\n\n")
            f.write("=" * 50 + "\n")
            f.write(f"平均回报: {avg_reward:.2f} ± {std_reward:.2f}\n")
            f.write(f"平均episode长度: {avg_length:.2f}\n")
            f.write(f"踢中率: {kick_rate:.1f}%\n")
            f.write(f"站立保持成功率: {stand_rate:.1f}%\n")

        print(f"\n✅ 结果已保存到: {results_path}")

    print("\n" + "=" * 70)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估踢球模型性能")
    parser.add_argument("--task", type=str, default="T1/Kicking", help="任务名称")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--num_episodes", type=int, default=100, help="评估episode数量")
    parser.add_argument("--num_envs", type=int, default=1, help="并行环境数")
    parser.add_argument("--save_results", action="store_true", help="是否保存结果到文件")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备 (cuda/cpu)")

    args = parser.parse_args()

    # 检查checkpoint文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"❌ 错误: 检查点文件不存在: {args.checkpoint}")
        exit(1)

    evaluate(args)
