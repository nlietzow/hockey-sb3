from pathlib import Path

from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.vec_env import VecEnv

from callbacks.update2 import UpdatePlayer2


def get_eval_callback(
    run_id: str,
    eval_env: VecEnv,
    *,
    checkpoint_dir: Path | None = None,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 5,
    stop_training_on_reward: float | None = None,
    update_player2_after_eval: bool = False,
    verbose: int = 1,
):
    callback_on_new_best = None
    if stop_training_on_reward is not None:
        callback_on_new_best = StopTrainingOnRewardThreshold(
            reward_threshold=stop_training_on_reward,
            verbose=verbose,
        )

    callback_after_eval = None
    if update_player2_after_eval:
        callback_after_eval = UpdatePlayer2(
            checkpoint_dir=checkpoint_dir,
            verbose=verbose,
        )

    return EvalCallback(
        eval_env,
        log_path=f"models/{run_id}/eval_logs",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        callback_on_new_best=callback_on_new_best,
        callback_after_eval=callback_after_eval,
        verbose=verbose,
    )
