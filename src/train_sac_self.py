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
assert BASE_CHECKPOINT.exists(), "Base checkpoint not found."

N_ENVS = 4


def _make_env(checkpoint: Path | None, monitor: bool):
    def _init():
        env = gym.make("Hockey-One-v0", checkpoint=checkpoint)
        if monitor:
            env = Monitor(env)

        return env

    return _init


def make_eval_env():
    envs = [_make_env(BASE_CHECKPOINT, monitor=False) for _ in range(N_ENVS)]

    vec_env = DummyVecEnv(envs)
    vec_env = VecMonitor(vec_env)

    return vec_env


def make_train_env(last_checkpoint: Path):
    envs = [
        _make_env(None, monitor=False),
        _make_env(BASE_CHECKPOINT, monitor=False),
    ]
    if last_checkpoint != BASE_CHECKPOINT:
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

    # Train the model
    last_checkpoint = BASE_CHECKPOINT
    for iteration in range(50):
        vec_env = make_train_env(last_checkpoint)
        model = SAC.load(
            last_checkpoint, env=vec_env, tensorboard_log=f"logs/{run.id}", verbose=1
        )
        model.load_replay_buffer(last_checkpoint.with_suffix(".replay_buffer"))

        model.learn(20_000, callback=[eval_callback])

        last_checkpoint = CHECKPOINTS_DIR / f"hockey_sac_{str(iteration).zfill(2)}.zip"
        model.save(last_checkpoint)
        model.save_replay_buffer(last_checkpoint.with_suffix(".replay_buffer"))

    run.save(last_checkpoint)
    run.finish()


if __name__ == "__main__":
    main()
