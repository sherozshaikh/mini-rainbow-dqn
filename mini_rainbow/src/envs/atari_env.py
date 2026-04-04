"""Atari environment factory with standard DQN preprocessing wrappers."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStack,
    RecordVideo,
    TransformReward,
)
from omegaconf import DictConfig


class FireResetWrapper(gym.Wrapper):
    """Press FIRE on reset for environments that require it (e.g., Breakout)."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)  # do nothing after fire
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class ChannelsFirstWrapper(gym.ObservationWrapper):
    """Convert observation from (H, W, C) to (C, H, W) for PyTorch conv layers.

    Also handles FrameStack's LazyFrames by converting to numpy first.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        old_shape = env.observation_space.shape
        # old_shape is (H, W, C) or from FrameStack (H, W, num_stack)
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, obs):
        # LazyFrames -> numpy, then transpose
        obs = np.array(obs)
        return np.transpose(obs, (2, 0, 1))


def make_atari_env(
    cfg: DictConfig,
    seed: int = 0,
    video_dir: str | None = None,
    eval_mode: bool = False,
) -> gym.Env:
    """Create a fully wrapped Atari environment.

    Args:
        cfg: Environment config (env section of Hydra config).
        seed: Random seed.
        video_dir: If provided, wrap with RecordVideo.
        eval_mode: If True, do not use episodic life wrapper.

    Returns:
        Wrapped Gymnasium environment with observation shape (C, H, W).
    """
    env = gym.make(
        cfg.name,
        render_mode="rgb_array" if video_dir else None,
    )

    # AtariPreprocessing handles:
    #   - NoopReset (noop_max)
    #   - Frame skip (frame_skip)
    #   - Grayscale (grayscale_obs)
    #   - Resize (screen_size)
    #   - Terminal on life loss (terminal_on_life_loss) -- only during training
    env = AtariPreprocessing(
        env,
        noop_max=cfg.noop_max,
        frame_skip=cfg.frame_skip,
        screen_size=cfg.screen_size,
        terminal_on_life_loss=cfg.episodic_life and not eval_mode,
        grayscale_obs=cfg.grayscale,
        scale_obs=False,  # keep uint8 [0,255], normalize in network
    )

    # Fire reset for Breakout-style games
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetWrapper(env)

    # Reward clipping: clip to [-1, 1]
    if cfg.clip_rewards and not eval_mode:
        env = TransformReward(env, lambda r: np.sign(r))

    # Frame stacking
    env = FrameStack(env, num_stack=cfg.frame_stack)

    # Convert to channels-first for PyTorch
    env = ChannelsFirstWrapper(env)

    # Video recording
    if video_dir:
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep: True,  # record every episode
            name_prefix="eval",
        )

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env
