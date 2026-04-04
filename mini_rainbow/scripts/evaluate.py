"""Standalone evaluation script. Run with: python -m mini_rainbow.scripts.evaluate"""

from __future__ import annotations

import argparse
import logging

import torch
from omegaconf import OmegaConf

from mini_rainbow.src.agents.dqn_agent import DQNAgent
from mini_rainbow.src.envs.atari_env import make_atari_env
from mini_rainbow.src.evaluation.evaluator import Evaluator
from mini_rainbow.src.replay.uniform import UniformReplayBuffer
from mini_rainbow.src.utils.checkpoint import load_checkpoint
from mini_rainbow.src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--video-dir", type=str, default="eval_videos", help="Video output dir")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|cuda")
    parser.add_argument("--env", type=str, default="ALE/Breakout-v5", help="Env name")
    parser.add_argument("--dueling", action="store_true", help="Use dueling network")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Build a minimal config for the evaluator
    cfg = OmegaConf.create(
        {
            "seed": args.seed,
            "device": str(device),
            "video_dir": args.video_dir,
            "env": {
                "name": args.env,
                "frame_skip": 4,
                "frame_stack": 4,
                "screen_size": 84,
                "grayscale": True,
                "clip_rewards": True,
                "episodic_life": True,
                "noop_max": 30,
            },
            "agent": {
                "double_dqn": False,
                "dueling": args.dueling,
                "per": False,
                "epsilon_start": 0.0,
                "epsilon_end": 0.0,
                "epsilon_decay_steps": 1,
                "learning_rate": 1e-4,
                "adam_eps": 1.5e-4,
                "max_grad_norm": 10.0,
                "gamma": 0.99,
                "target_update_freq": 10000,
            },
            "training": {
                "batch_size": 32,
                "learning_starts": 50000,
                "eval_freq": 100000,
                "eval_episodes": args.episodes,
                "record_video": True,
                "save_freq": 250000,
                "log_freq": 1000,
                "total_steps": 0,
            },
            "replay": {
                "buffer_size": 1000,
                "type": "uniform",
            },
        }
    )

    # Create a dummy env to get obs_shape and num_actions
    tmp_env = make_atari_env(cfg=cfg.env, seed=args.seed)
    obs_shape = tmp_env.observation_space.shape
    num_actions = tmp_env.action_space.n
    tmp_env.close()

    # Create dummy buffer (not used during eval)
    dummy_buffer = UniformReplayBuffer(buffer_size=1000, obs_shape=obs_shape)

    # Create agent
    agent = DQNAgent(
        obs_shape=obs_shape,
        num_actions=num_actions,
        replay_buffer=dummy_buffer,
        cfg=cfg,
        device=device,
    )

    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    agent.load_state(checkpoint)

    # Run evaluation
    evaluator = Evaluator(agent=agent, cfg=cfg)
    mean_reward = evaluator.evaluate(step=0)

    logger.info(f"Evaluation complete: mean_reward={mean_reward:.2f}")


if __name__ == "__main__":
    main()
