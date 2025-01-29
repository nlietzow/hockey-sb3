from pathlib import Path
from typing import Type

import gymnasium as gym
import wandb
from hockey import OpponentType, REGISTERED_ENVS
from sbx import CrossQ
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from callbacks import get_eval_callback, get_update2_callback, get_wandb_callback
from utils import CHECKPOINTS_DIR

assert REGISTERED_ENVS, "Hockey environments are not registered."

TOTAL_TIME_STEPS = 1_000_000
BASELINE_PATH = CHECKPOINTS_DIR / "cross_q" / "model.zip"
assert BASELINE_PATH.exists(), "Baseline model not found."


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
        return env

    return _init


def make_train_env(checkpoint_dir: Path):
    envs = [
        _make_env(
            OpponentType(ot),
            CrossQ,
            checkpoint_path=BASELINE_PATH,
            checkpoint_dir=checkpoint_dir,
        )
        for ot in list(OpponentType) * 4
    ]

    vec_env = DummyVecEnv(envs)
    vec_env = VecMonitor(vec_env)

    return vec_env


def make_eval_env():
    envs = [
        _make_env(
            OpponentType.checkpoint,
            CrossQ,
            checkpoint_path=BASELINE_PATH,
        )
        for _ in range(4)
    ]

    vec_env = DummyVecEnv(envs)
    vec_env = VecMonitor(vec_env)

    return vec_env


def main():
    run = wandb.init(
        project="cross_q-self-play",
        sync_tensorboard=True,
        settings=wandb.Settings(silent=True),
    )
    checkpoint_dir = BASELINE_PATH.parent / run.name
    checkpoint_dir.mkdir(exist_ok=True)

    env = make_train_env(checkpoint_dir)
    eval_env = make_eval_env()
    callback = CallbackList(
        [
            get_wandb_callback(run.id),
            get_eval_callback(
                run.id,
                env,
                eval_env,
                checkpoint_dir=checkpoint_dir,
            ),
            get_update2_callback(env, checkpoint_dir),
        ]
    )
    success = False
    try:
        model = CrossQ.load(
            BASELINE_PATH,
            env,
            verbose=False,
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
