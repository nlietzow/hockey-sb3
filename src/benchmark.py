from typing import Type, Union

import gymnasium as gym
import wandb
from hockey import OpponentType, REGISTERED_ENVS
from sb3_contrib import CrossQ, TQC, TRPO
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from utils import CHECKPOINTS_DIR, get_eval_callback, get_wandb_callback

assert REGISTERED_ENVS, "Hockey environments are not registered."


def make_env():
    def init():
        env = gym.make(
            "Hockey-One-v0",
            opponent_type=OpponentType.rule_based,
            checkpoint_dir=CHECKPOINTS_DIR,
        )
        env = Monitor(env)
        return env

    return DummyVecEnv([init])


def run_for_algo(
    algo: Union[
        Type[DDPG],
        Type[SAC],
        Type[TD3],
        Type[CrossQ],
        Type[TQC],
        Type[PPO],
        Type[A2C],
        Type[TRPO],
    ]
):
    config = {
        "algorithm": algo.__name__,
        "policy_type": "MlpPolicy",
        "total_timesteps": 1_000_000,
    }
    run = wandb.init(
        project="hockey-benchmark",
        config=config,
        sync_tensorboard=True,
    )

    env, eval_env = make_env(), make_env()
    callback = CallbackList(
        [
            get_wandb_callback(run.id),
            get_eval_callback(run.id, env, eval_env),
        ]
    )
    success = False
    try:
        model = algo(
            config["policy_type"], env, verbose=1, tensorboard_log=f"logs/{run.id}"
        )
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callback,
        )
        success = True
    finally:
        env.close()
        eval_env.close()
        run.finish(exit_code=0 if success else 1)


def run_off_policy():
    # run for off-policy algorithms
    for algo in (DDPG, SAC, TD3, CrossQ, TQC):
        try:
            run_for_algo(algo)
        except Exception as e:
            print(f"Error during {algo} training:", e)


def run_on_policy():
    # run for on-policy algorithms
    for algo in (PPO, A2C, TRPO):
        try:
            run_for_algo(algo)
        except Exception as e:
            print(f"Error during {algo} training:", e)


if __name__ == "__main__":
    run_on_policy()
