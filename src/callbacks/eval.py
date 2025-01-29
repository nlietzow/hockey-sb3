import pickle
from pathlib import Path

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.vec_env import VecEnv


class SaveBestModelParameters(BaseCallback):
    def __init__(self, checkpoint_dir: Path):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir

    def _on_step(self) -> bool:
        params = self.model.get_parameters()
        with open(self.checkpoint_dir / "best_model.pkl", "wb") as f:
            pickle.dump(params, f)

        return True


def get_eval_callback(
    run_id: str,
    env: VecEnv,
    eval_env: VecEnv,
    *,
    checkpoint_dir: Path | None = None,
    eval_freq: int = 1_000,
    n_eval_episodes: int = 10,
    stop_training_on_reward: float | None = None,
    deterministic: bool = True,
    render: bool = False,
):
    if checkpoint_dir:
        checkpoint_dir = checkpoint_dir.resolve()

    callbacks = []
    if checkpoint_dir is not None:
        callbacks.append(SaveBestModelParameters(checkpoint_dir))

    if stop_training_on_reward is not None:
        callbacks.append(
            StopTrainingOnRewardThreshold(
                reward_threshold=stop_training_on_reward, verbose=1
            )
        )

    callback_on_new_best = CallbackList(callbacks) if callbacks else None

    return EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir) if checkpoint_dir else None,
        log_path=f"models/{run_id}/eval_logs",
        eval_freq=eval_freq * env.num_envs,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=render,
        callback_on_new_best=callback_on_new_best,
    )
