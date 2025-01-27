from typing import Type, Union

import gymnasium as gym
import wandb
from hockey import OpponentType, REGISTERED_ENVS
from sb3_contrib import CrossQ, TQC
from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from utils import CHECKPOINTS_DIR

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


def get_callbacks(run_id: str, eval_env: DummyVecEnv):
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/{run_id}/best_model",
        log_path=f"models/{run_id}/eval_logs",
        eval_freq=1_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    wandb_callback = WandbCallback(
        model_save_path=f"models/{run_id}",
        verbose=2,
    )
    return CallbackList([eval_callback, wandb_callback])


def run_for_algo(
    algo: Union[Type[DDPG], Type[SAC], Type[TD3], Type[CrossQ], Type[TQC]]
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
    callback = get_callbacks(run.id, eval_env)
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


def main():
    for algo in (DDPG, SAC, TD3, CrossQ, TQC):
        try:
            run_for_algo(algo)
        except Exception as e:
            print(f"Error during {algo} training:", e)


if __name__ == "__main__":
    main()
