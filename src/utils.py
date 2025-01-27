from pathlib import Path

import gymnasium as gym
from hockey import OpponentType
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    EveryNTimesteps,
)
from stable_baselines3.common.vec_env import VecEnv
from wandb.integration.sb3 import WandbCallback

_base_dir = Path(__file__).resolve().parent
CHECKPOINTS_DIR = _base_dir / "checkpoints" / "sac"


class UpdatePlayer2(BaseCallback):
    def _on_step(self) -> bool:
        self.model.save(f"{str(self.num_timesteps).zfill(9)}.zip")
        self.training_env.env_method("update_player2")
        return True


def make_env(opponent_type: OpponentType, checkpoint_dir: Path = CHECKPOINTS_DIR):
    def _init():
        env = gym.make(
            "Hockey-One-v0",
            opponent_type=opponent_type,
            checkpoint_dir=checkpoint_dir,
        )
        return env

    return _init


def get_wandb_callback(run_id: str, verbose: int = 2):
    return WandbCallback(
        model_save_path=f"models/{run_id}",
        verbose=verbose,
    )


def get_eval_callback(
    run_id: str,
    eval_env: VecEnv,
    eval_freq: int = 1_000,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
):
    return EvalCallback(
        eval_env,
        best_model_save_path=str(CHECKPOINTS_DIR / "best.zip"),
        log_path=f"models/{run_id}/eval_logs",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=render,
    )


def get_update2_callback(env: VecEnv, n_steps: int = 10_000):
    return EveryNTimesteps(n_steps=n_steps * env.num_envs, callback=UpdatePlayer2())
