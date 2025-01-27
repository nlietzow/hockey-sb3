import wandb
from hockey import OpponentType, REGISTERED_ENVS
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from utils import (
    CHECKPOINTS_DIR,
    get_eval_callback,
    get_update2_callback,
    get_wandb_callback,
    make_env,
)

assert REGISTERED_ENVS, "Hockey environments are not registered."

TOTAL_TIME_STEPS = 1_000_000
LEARNING_RATE = 1e-4


def make_envs(train: bool):
    if train:
        envs = [make_env(OpponentType(ot)) for ot in OpponentType]
    else:
        envs = [make_env(OpponentType.baseline) for _ in range(len(OpponentType))]
    vec_env = DummyVecEnv(envs)
    vec_env = VecMonitor(vec_env)

    return vec_env


def main():
    run = wandb.init(project="hockey-sb3", sync_tensorboard=True)

    env, eval_env = make_envs(train=True), make_envs(train=False)
    callback = CallbackList(
        [
            get_wandb_callback(run.id),
            get_eval_callback(run.id, eval_env),
            get_update2_callback(env),
        ]
    )
    success = False
    try:
        model = SAC.load(
            CHECKPOINTS_DIR / "baseline.zip",
            env,
            verbose=1,
            learning_rate=LEARNING_RATE,
            tensorboard_log=f"logs/{run.id}",
        )
        model.learn(
            total_timesteps=TOTAL_TIME_STEPS * env.num_envs,
            reset_num_timesteps=False,
            callback=callback,
        )
        success = True
    finally:
        env.close()
        eval_env.close()
        run.finish(exit_code=0 if success else 1)


if __name__ == "__main__":
    main()
