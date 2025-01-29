import pickle
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.vec_env import VecEnv


class UpdatePlayer2(BaseCallback):
    """
    Save the model as a checkpoint and call the update_player2 method of the environment
    """

    def __init__(self, checkpoint_dir: Path, min_steps: int = 1_000):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self._min_steps = min_steps

    def _on_step(self) -> bool:
        params = self.model.get_parameters()
        fp = self.checkpoint_dir / f"{str(self.num_timesteps).zfill(9)}.pkl"
        with open(fp, "wb") as f:
            pickle.dump(params, f)

        if self.n_calls > self._min_steps:
            self.logger.info("Triggering update_player2 method.")
            self.training_env.env_method("update_player2", logger=self.logger)

        return True


def get_update2_callback(
    env: VecEnv,
    checkpoint_dir: Path,
    n_steps: int = 10_000,
):
    """
    Get the callback to update the opponent player after every n steps
    :param env:
    :type env:
    :param checkpoint_dir:
    :type checkpoint_dir:
    :param n_steps:
    :type n_steps:
    :return:
    :rtype:
    """
    return EveryNTimesteps(
        n_steps=n_steps * env.num_envs,
        callback=UpdatePlayer2(checkpoint_dir=checkpoint_dir),
    )
