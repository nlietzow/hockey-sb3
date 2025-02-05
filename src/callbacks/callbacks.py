from typing import Optional

import numpy as np
from hockey.hockey_env import BasicOpponent, HockeyEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EveryNTimesteps,
)
from wandb.integration.sb3 import WandbCallback


class _UpdatePlayerTwoCallback(BaseCallback):
    def __init__(self, update_player_two: bool, verbose: int = 0):
        super().__init__(verbose)
        self.update_player_two = update_player_two

    def _on_step(self) -> bool:
        if self.update_player_two:
            self.training_env.env_method("update_player_two")
        return True


class _CustomEvalCallback(BaseCallback):
    def __init__(
        self,
        n_episodes: int = 100,
        stop_on_reward_threshold: Optional[float] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.n_episodes = n_episodes
        self.stop_on_reward_threshold = stop_on_reward_threshold

        self.env = HockeyEnv()
        self.opponent = BasicOpponent(weak=False)
        self.best_mean_reward = -np.inf

    def _run_eval(self):
        rewards = np.full((self.n_episodes, self.env.max_timesteps), np.nan)
        success = np.full(self.n_episodes, False)

        for episode in range(self.n_episodes):
            obs, info = self.env.reset()
            for step in range(self.env.max_timesteps):
                obs2 = self.env.obs_agent_two()
                a1, _ = self.model.predict(obs)
                a2 = self.opponent.act(obs2)
                obs, reward, done, _, info = self.env.step(np.hstack([a1, a2]))

                rewards[episode, step] = reward
                if done:
                    break

            success[episode] = info.get("winner", 0) == 1

        mean_reward = np.mean(np.nansum(rewards, axis=1))
        mean_episode_length = np.mean(np.sum(~np.isnan(rewards), axis=1))
        success_rate = np.mean(success)

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_episode_length", mean_episode_length)
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("time/total_timesteps", self.num_timesteps)
        self.logger.dump(step=self.num_timesteps)

        return mean_reward, mean_episode_length, success_rate

    def _on_step(self) -> bool:
        mean_reward, _, _ = self._run_eval()

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.verbose > 0:
                print("New best mean reward: {:.2f}".format(mean_reward))

        return self._continue_training

    @property
    def _continue_training(self) -> bool:
        stop_training = (
            self.stop_on_reward_threshold is not None
            and self.best_mean_reward > self.stop_on_reward_threshold
        )
        if self.verbose > 0 and stop_training:
            print("Stopping training because mean reward is greater than threshold.")

        return not stop_training


def init(
    n_steps: int = 10_000,
    n_episodes_eval: int = 100,
    update_player_two: bool = False,
    stop_on_reward_threshold: Optional[float] = None,
    verbose: int = 1,
) -> CallbackList:
    eval_callback = _CustomEvalCallback(
        n_episodes=n_episodes_eval,
        stop_on_reward_threshold=stop_on_reward_threshold,
        verbose=verbose,
    )
    every_n_steps = EveryNTimesteps(
        n_steps=n_steps,
        callback=CallbackList(
            [
                eval_callback,
                _UpdatePlayerTwoCallback(update_player_two),
            ]
        ),
    )
    return CallbackList(
        [
            every_n_steps,
            WandbCallback(),
        ]
    )
