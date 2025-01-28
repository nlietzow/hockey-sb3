from pathlib import Path
from typing import Type

import gymnasium as gym
import wandb
from hockey import OpponentType, REGISTERED_ENVS
from sb3_contrib import CrossQ
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from callbacks import get_eval_callback, get_wandb_callback
from utils import CROSS_Q_DIR

assert REGISTERED_ENVS, "Hockey environments are not registered."

TOTAL_TIME_STEPS = 1_000_000


def _make_env(
    opponent_type: OpponentType = OpponentType.rule_based,
    algorithm_cls: Type[BaseAlgorithm] = None,
    checkpoint_path: Path = None,
    checkpoint_dir: Path = None,
):
    def _init():
        env = gym.make(
            "Hockey-One-v0",
            opponent_type=opponent_type,
            algorithm_cls=algorithm_cls,
            checkpoint_path=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
        )
        env = Monitor(env)
        return env

    return _init


def make_env():
    envs = [
        _make_env(
            OpponentType.checkpoint,
            CrossQ,
            checkpoint_path=CROSS_Q_DIR / "baseline.zip",
        )
    ]
    return DummyVecEnv(envs)


def main():
    run = wandb.init(project="cross_q-self-play", sync_tensorboard=True)
    checkpoint_dir = CROSS_Q_DIR / run.id
    checkpoint_dir.mkdir(exist_ok=True)

    env, eval_env = make_env(), make_env()
    callback = CallbackList(
        [
            get_wandb_callback(run.id),
            get_eval_callback(run.id, env, eval_env, checkpoint_dir),
        ]
    )
    success = False
    try:
        model = CrossQ.load(
            CROSS_Q_DIR / "baseline.zip",
            env,
            verbose=1,
            tensorboard_log=f"logs/{run.id}",
        )
        model.learn(
            total_timesteps=TOTAL_TIME_STEPS * env.num_envs,
            reset_num_timesteps=False,
            callback=callback,
        )
        success = True
    finally:
        env.close()
        eval_env.close()
        run.finish(exit_code=0 if success else 1)


if __name__ == "__main__":
    main()
