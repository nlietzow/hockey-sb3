from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class StopTrainingOnSuccessRate(BaseCallback):
    """
    Callback to stop training early when the success rate surpasses a threshold,
    but only after a minimum number of steps has been completed.

    :param success_threshold: (float) The success rate threshold to stop training.
    :param min_steps: (int) Minimum number of training steps before stopping.
    :param verbose: (int) Verbosity level: 0 for no output, 1 for info messages.
    """

    def __init__(
        self,
        env: VecEnv,
        success_threshold: float = 0.95,
        min_steps: int = 10_000,
        consecutive_steps: int = 1_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.success_threshold = success_threshold
        self.min_steps = min_steps * env.num_envs
        self.consecutive_steps = consecutive_steps * env.num_envs
        self._step = 0

    def _on_step(self) -> bool:
        """
        Called at each step. Stops training if the success rate is above the threshold
        and the minimum number of steps has been completed.
        """
        # Ensure enough steps have been taken
        if self.n_calls < self.min_steps:
            return True

        # Ensure the `rollout/ep_success_rate` metric exists
        success_rate = self.logger.name_to_value["rollout/success_rate"]

        if success_rate >= self.success_threshold:
            self._step += 1
            continue_training = self._step < self.consecutive_steps
            if not continue_training:
                self.logger.info(
                    f"Stopping training early due to success rate of {success_rate:.2f} "
                    f"exceeding {self.success_threshold} for {self.consecutive_steps} consecutive steps."
                )

            return continue_training
        else:
            self._step = 0

        return True  # Continue training


def get_early_stopping_callback(
    env: VecEnv, success_threshold: float = 0.95, consecutive_steps: int = 1_000
):
    return StopTrainingOnSuccessRate(env, success_threshold, consecutive_steps)
