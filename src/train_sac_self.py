import random
from pathlib import Path

import gymnasium as gym
import wandb
from hockey import REGISTERED_ENVS
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints" / "sac"
BASE_CHECKPOINT = CHECKPOINTS_DIR / "hockey_sac_base.zip"

assert REGISTERED_ENVS, "Hockey environments are not registered."

N_ENVS = 4


def _make_env(checkpoint: Path | None, monitor: bool):
    def _init():
        env = gym.make("Hockey-One-v0", checkpoint=checkpoint)
        if monitor:
            env = Monitor(env)

        return env

    return _init


def make_eval_env():
    envs = [_make_env(None, monitor=False) for _ in range(N_ENVS - 1)]
    envs.append(_make_env(BASE_CHECKPOINT, monitor=False))

    vec_env = DummyVecEnv(envs)
    vec_env = VecMonitor(vec_env)

    return vec_env


def make_train_env(last_checkpoint: Path | None):
    envs = [_make_env(None, monitor=False)]

    if last_checkpoint is not None:
        envs.append(_make_env(last_checkpoint, monitor=False))

    all_checkpoints = set(CHECKPOINTS_DIR.glob("*.zip"))
    all_checkpoints.discard(BASE_CHECKPOINT)
    all_checkpoints.discard(last_checkpoint)
    all_checkpoints = list(all_checkpoints)
    random.shuffle(all_checkpoints)

    for cp in all_checkpoints[: N_ENVS - len(envs)]:
        envs.append(_make_env(cp, monitor=False))

    while len(envs) < N_ENVS:
        envs.append(_make_env(None, monitor=False))

    vec_env = DummyVecEnv(envs)
    vec_env = VecMonitor(vec_env)

    return vec_env


def main():
    # Init wandb run
    run = wandb.init(project="hockey-sb3", sync_tensorboard=True)

    # Set up evaluation
    eval_env = make_eval_env()
    eval_callback = EvalCallback(
        eval_env,
        log_path=f"models/{run.id}/eval_logs",
        eval_freq=1_000,
        n_eval_episodes=4,
        deterministic=True,
        render=False,
    )

    # Define the model
    vec_env = make_train_env(None)
    model = SAC("MlpPolicy", vec_env, verbose=1, tensorboard_log=f"logs/{run.id}")

    # Train the model
    last_checkpoint = None
    for iteration in range(50):
        vec_env = make_train_env(last_checkpoint)
        model.set_env(vec_env)
        model.learn(20_000, reset_num_timesteps=False, callback=[eval_callback])

        last_checkpoint = CHECKPOINTS_DIR / f"hockey_sac_{str(iteration).zfill(2)}.zip"
        model.save(last_checkpoint)

    if last_checkpoint is not None:
        run.save(last_checkpoint)

    run.finish()


if __name__ == "__main__":
    main()
