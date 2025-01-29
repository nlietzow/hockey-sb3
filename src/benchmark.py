from typing import Type, Union

import wandb
from hockey import REGISTERED_ENVS
from sbx import CrossQ, DDPG, PPO, SAC, TD3, TQC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env

from callbacks import get_eval_callback, get_wandb_callback

assert REGISTERED_ENVS, "Hockey environments are not registered."

POLICY_TYPE = "MlpPolicy"
TOTAL_TIME_STEPS = 1_000_000
NUM_ENVS = 4

Algorithm = Union[
    Type[DDPG],
    Type[SAC],
    Type[TD3],
    Type[CrossQ],
    Type[TQC],
    Type[PPO],
]


def run_for_algo(algorithm: Algorithm):
    run = wandb.init(
        project="hockey-benchmark",
        sync_tensorboard=True,
    )

    env = make_vec_env("Hockey-One-v0", n_envs=NUM_ENVS)
    eval_env = make_vec_env("Hockey-One-v0", n_envs=NUM_ENVS)

    callback = CallbackList(
        [
            get_wandb_callback(run.id),
            get_eval_callback(run.id, env, eval_env),
        ]
    )
    success = False
    try:
        model = algorithm(
            POLICY_TYPE, env, verbose=1, tensorboard_log=f"logs/{run.id}"
        )
        model.learn(
            total_timesteps=TOTAL_TIME_STEPS,
            callback=callback,
        )
        success = True
    finally:
        env.close()
        eval_env.close()
        run.finish(exit_code=0 if success else 1)


def run_off_policy():
    # run for off-policy algorithms
    for algo in (CrossQ, SAC, TQC, TD3, DDPG):
        try:
            run_for_algo(algo)
        except Exception as e:
            print(f"Error during {algo} training:", e)


def run_on_policy():
    algo = PPO
    try:
        run_for_algo(algo)
    except Exception as e:
        print(f"Error during {algo} training:", e)


if __name__ == "__main__":
    run_off_policy()
    run_on_policy()
