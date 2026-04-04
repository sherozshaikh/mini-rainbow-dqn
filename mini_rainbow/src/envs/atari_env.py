"""Atari environment factory with production-grade DQN preprocessing."""

from __future__ import annotations

import collections
from typing import Deque

import ale_py  # noqa: F401 (ensures ALE envs are registered)
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, RecordVideo, TransformReward
from omegaconf import DictConfig


class FireResetWrapper(gym.Wrapper):
    """Press FIRE on reset for environments that require it (e.g., Breakout)."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)

        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)

        return obs, info


class FrameStackWrapper(gym.Wrapper):
    """Custom frame stacker that returns (C, H, W) uint8 arrays directly.

    Unlike gymnasium's FrameStack which returns LazyFrames in (H, W, C) order,
    this wrapper outputs (num_stack, H, W) — ready for PyTorch conv layers.
    """

    def __init__(self, env: gym.Env, num_stack: int):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames: Deque[np.ndarray] = collections.deque(maxlen=num_stack)

        obs_shape = env.observation_space.shape  # (H, W) or (H, W, 1)
        if len(obs_shape) == 3:
            h, w, _ = obs_shape
        else:
            h, w = obs_shape

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(num_stack, h, w),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        frame = self._process_obs(obs)
        for _ in range(self.num_stack):
            self.frames.append(frame)

        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        frame = self._process_obs(obs)
        self.frames.append(frame)

        return self._get_obs(), reward, terminated, truncated, info

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 3:
            obs = obs.squeeze(-1)
        return obs.astype(np.uint8)

    def _get_obs(self) -> np.ndarray:
        return np.stack(self.frames, axis=0)


def make_atari_env(
    cfg: DictConfig,
    seed: int = 0,
    video_dir: str | None = None,
    eval_mode: bool = False,
) -> gym.Env:
    """Create fully wrapped Atari environment.

    Args:
        cfg: Environment config (env section of Hydra config).
        seed: Random seed.
        video_dir: If provided, wrap with RecordVideo.
        eval_mode: If True, do not use episodic life wrapper or reward clipping.

    Returns:
        Wrapped Gymnasium environment with observation shape (C, H, W).
    """
    env_id = cfg.get("id", cfg.get("name"))

    # ALE/Breakout-v5 has built-in frameskip — disable it so AtariPreprocessing controls it
    env = gym.make(
        env_id,
        frameskip=1,
        render_mode="rgb_array" if video_dir else None,
    )

    env = AtariPreprocessing(
        env,
        noop_max=cfg.noop_max,
        frame_skip=cfg.frame_skip,
        screen_size=cfg.screen_size,
        terminal_on_life_loss=cfg.episodic_life and not eval_mode,
        grayscale_obs=cfg.grayscale,
        scale_obs=False,
    )

    # Fire reset for Breakout-style games
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetWrapper(env)

    # Reward clipping: clip to {-1, 0, 1}
    if cfg.clip_rewards and not eval_mode:
        env = TransformReward(env, lambda r: np.sign(r))

    # Frame stacking — outputs (C, H, W) directly
    env = FrameStackWrapper(env, num_stack=cfg.frame_stack)

    # Video recording
    if video_dir:
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep: True,
            name_prefix="eval",
        )

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.reset(seed=seed)

    return env
