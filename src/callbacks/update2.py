import pickle
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class UpdatePlayer2(BaseCallback):
    """
    Save the model as a checkpoint and call the update_player2 method of the environment
    """

    def __init__(self, checkpoint_dir: Path, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.checkpoint_dir = checkpoint_dir

    def _on_step(self) -> bool:
        params = self.model.get_parameters()
        fp = self.checkpoint_dir / f"{str(self.num_timesteps).zfill(9)}.pkl"
        with open(fp, "wb") as f:
            pickle.dump(params, f)

        if self.verbose > 0:
            print(f"Updating player2 at step {self.num_timesteps}")

        self.training_env.env_method("update_player2", verbose=self.verbose)

        return True
