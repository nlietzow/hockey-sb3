import pickle
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.vec_env import VecEnv


class UpdatePlayer2(BaseCallback):
    """
    Save the model as a checkpoint and call the update_player2 method of the environment
    """

    def __init__(self, checkpoint_dir: Path, min_steps: int = 1_000, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.checkpoint_dir = checkpoint_dir
        self._min_steps = min_steps

    def _on_step(self) -> bool:
        params = self.model.get_parameters()
        fp = self.checkpoint_dir / f"{str(self.num_timesteps).zfill(9)}.pkl"
        with open(fp, "wb") as f:
            pickle.dump(params, f)

        if self.n_calls > self._min_steps:
            self.training_env.env_method("update_player2", verbose=self.verbose)

        return True


def get_update2_callback(
    env: VecEnv,
    checkpoint_dir: Path,
    n_steps: int = 10_000,
    verbose: int = 1,
):
    return EveryNTimesteps(
        n_steps=n_steps * env.num_envs,
        callback=UpdatePlayer2(checkpoint_dir=checkpoint_dir, verbose=verbose),
    )
