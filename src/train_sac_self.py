import random
from pathlib import Path

import gymnasium as gym
import wandb
from hockey import REGISTERED_ENVS
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints" / "sac"
BASE_CHECKPOINT = CHECKPOINTS_DIR / "hockey_sac_base.zip"

assert REGISTERED_ENVS, "Hockey environments are not registered."
assert BASE_CHECKPOINT.exists(), "Base checkpoint not found."


def make_single_env(checkpoint: Path | None):
    def _init():
        env = gym.make("Hockey-One-v0", checkpoint=checkpoint)
        return Monitor(env)

    return _init


def make_parallel_envs(last_checkpoint: Path):
    envs = [
        make_single_env(None),
        make_single_env(BASE_CHECKPOINT),
    ]
    if last_checkpoint != BASE_CHECKPOINT:
        envs.append(make_single_env(last_checkpoint))

    all_checkpoints = set(CHECKPOINTS_DIR.glob("*.zip"))
    all_checkpoints.discard(BASE_CHECKPOINT)
    all_checkpoints.discard(last_checkpoint)
    all_checkpoints = list(all_checkpoints)
    random.shuffle(all_checkpoints)

    for cp in all_checkpoints[:5]:
        envs.append(make_single_env(cp))

    return envs


def main():
    # Init wandb run
    run = wandb.init(project="hockey-sb3", sync_tensorboard=True)

    # Set up evaluation
    eval_env = DummyVecEnv([make_single_env(BASE_CHECKPOINT)])
    eval_callback = EvalCallback(
        eval_env,
        log_path=f"models/{run.id}/eval_logs",
        eval_freq=1_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # Load the model
    model = SAC.load(
        BASE_CHECKPOINT, learning_rate=1e-4, tensorboard_log=f"logs/{run.id}", verbose=1
    )

    # Train the model
    last_checkpoint = BASE_CHECKPOINT
    for iteration in range(100):
        # Create parallel environments
        envs = make_parallel_envs(last_checkpoint)
        vec_env = SubprocVecEnv(envs)
        vec_env = VecMonitor(vec_env)

        # Train the model
        model.set_env(vec_env)
        model.learn(20_000, reset_num_timesteps=False, callback=[eval_callback])

        # Save the model
        last_checkpoint = CHECKPOINTS_DIR / f"hockey_sac_{str(iteration).zfill(2)}.zip"
        model.save(last_checkpoint)

    run.save(last_checkpoint)
    run.finish()


if __name__ == "__main__":
    main()
