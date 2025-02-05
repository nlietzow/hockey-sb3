from pathlib import Path

import wandb
from hockey import REGISTERED_ENVS
from sbx import CrossQ, DDPG, SAC, TD3, TQC
from stable_baselines3.common.env_util import make_vec_env

from callbacks import callbacks
from utils import Algorithm

assert REGISTERED_ENVS, "Hockey environments are not registered."

POLICY_TYPE = "MlpPolicy"
TOTAL_TIME_STEPS = 2_000_000
STOP_TRAINING_ON_REWARD = 9.25


def run_for_algo(algorithm: Algorithm, n_envs: int = 4, verbose: bool = True):
    run = wandb.init(
        project="hockey-benchmark",
        sync_tensorboard=True,
        settings=wandb.Settings(silent=True),
    )

    if verbose:
        print(f"Training {algorithm} with {n_envs} environments.")

    env = make_vec_env("Hockey-One-v0", n_envs=n_envs)
    callback = callbacks.init(
        run_id=run.id,
        stop_on_reward_threshold=STOP_TRAINING_ON_REWARD
    )
    success = False
    try:
        model = algorithm(
            algorithm.policy_aliases[POLICY_TYPE],
            env=env,
            verbose=False,
            tensorboard_log=f"logs/{run.id}",
        )
        model.learn(
            total_timesteps=TOTAL_TIME_STEPS,
            callback=callback,
        )
        success = True
    finally:
        env.close()
        run.finish(exit_code=int(not success))


def main(verbose: bool = True):
    for algo in (CrossQ, TQC, SAC, TD3, DDPG):
        try:
            run_for_algo(algo, verbose=verbose)
        except Exception as e:
            print(f"Error during {algo} training:", e)


if __name__ == "__main__":
    main()
