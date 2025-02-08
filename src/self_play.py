from pathlib import Path

import wandb
from hockey import REGISTERED_ENVS
from sbx import CrossQ
from sbx.common.off_policy_algorithm import OffPolicyAlgorithmJax
from stable_baselines3.common.env_util import make_vec_env

from callbacks import callbacks
from utils import CHECKPOINTS_DIR

assert REGISTERED_ENVS, "Hockey environments are not registered."

POLICY_TYPE = "MlpPolicy"
TOTAL_TIME_STEPS = 4_000_000


def main(
    model_class: type[OffPolicyAlgorithmJax], checkpoint_path: Path, n_envs: int = 4
):
    run = wandb.init(
        project="cross_q-self-play",
        sync_tensorboard=True,
        settings=wandb.Settings(silent=True),
    )
    success = False
    try:
        env = make_vec_env(
            "Hockey-One-v0-AIOpponent",
            n_envs=n_envs,
            env_kwargs=dict(
                model_class=model_class,
                checkpoint=checkpoint_path,
            ),
        )
        model = CrossQ.load(
            checkpoint_path,
            env=env,
            verbose=False,
            tensorboard_log=f"logs/{run.id}",
        )
        callback = callbacks.init(
            run_id=run.id,
            n_steps=100_000,
            update_player_two=True,
        )
        model.learn(
            total_timesteps=TOTAL_TIME_STEPS,
            callback=callback,
        )
        success = True
    finally:
        run.finish(exit_code=int(not success))


if __name__ == "__main__":
    cp = CHECKPOINTS_DIR / "cross_q" / "model.zip"
    main(CrossQ, cp)
