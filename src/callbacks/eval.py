import pickle
from pathlib import Path

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.vec_env import VecEnv

from callbacks.update2 import UpdatePlayer2


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
    update_player2_after_eval: bool = False,
    verbose: int = 1,
    deterministic: bool = True,
    render: bool = False,
):
    if checkpoint_dir:
        checkpoint_dir = checkpoint_dir.resolve()

    callback_on_new_best = []
    if checkpoint_dir is not None:
        callback_on_new_best.append(SaveBestModelParameters(checkpoint_dir))

    if stop_training_on_reward is not None:
        callback_on_new_best.append(
            StopTrainingOnRewardThreshold(
                reward_threshold=stop_training_on_reward, verbose=1
            )
        )

    if update_player2_after_eval:
        callback_after_eval = UpdatePlayer2(checkpoint_dir=checkpoint_dir, verbose=verbose)
    else:
        callback_after_eval = None

    return EvalCallback(
        eval_env,
        log_path=f"models/{run_id}/eval_logs",
        eval_freq=eval_freq * env.num_envs,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=render,
        callback_on_new_best=CallbackList(callback_on_new_best) if callback_on_new_best else None,
        verbose=verbose,
        callback_after_eval=callback_after_eval
    )
