from pathlib import Path

import gymnasium as gym
import wandb
from hockey import OpponentType, REGISTERED_ENVS
from sbx import CrossQ
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from callbacks import get_eval_callback, get_wandb_callback
from utils import Algorithm, CHECKPOINTS_DIR

assert REGISTERED_ENVS, "Hockey environments are not registered."

TOTAL_TIME_STEPS = 4_000_000
BASELINE_PATH = CHECKPOINTS_DIR / "cross_q" / "model.zip"


def _make_env(
    opponent_type: OpponentType = OpponentType.rule_based,
    algorithm_cls: Algorithm = None,
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
        return Monitor(env)

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

    return DummyVecEnv(envs)


def make_eval_env():
    envs = [
        _make_env(
            OpponentType.checkpoint,
            CrossQ,
            checkpoint_path=BASELINE_PATH,
        )
        for _ in range(4)
    ]

    return DummyVecEnv(envs)


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
                eval_env,
                checkpoint_dir=checkpoint_dir,
                update_player2_after_eval=True,
            ),
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
            total_timesteps=TOTAL_TIME_STEPS,
            reset_num_timesteps=False,
            callback=callback,
        )
        success = True
    finally:
        env.close()
        eval_env.close()
        run.finish(exit_code=int(not success))


if __name__ == "__main__":
    main()
