import wandb
from hockey import REGISTERED_ENVS
from sbx import CrossQ
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

from utils import CHECKPOINTS_DIR

assert REGISTERED_ENVS, "Hockey environments are not registered."

POLICY_TYPE = "MlpPolicy"
TOTAL_TIME_STEPS = 100_000


def main(
    n_envs: int = 4,
    model_class: type[CrossQ] = CrossQ,
    checkpoint_path=CHECKPOINTS_DIR / "cross_q" / "model.zip",
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
        eval_env = make_vec_env("Hockey-One-v0-BasicOpponent", n_envs=n_envs)
        callbacks = CallbackList(
            [
                EvalCallback(
                    eval_env=eval_env,
                    n_eval_episodes=10,
                    eval_freq=10_000,
                ),
                WandbCallback(model_save_path=f"models/{run_id}"),
            ]
        )

        parameters = model_class.load(checkpoint_path).get_parameters()
        for _ in range(10):
            model = CrossQ(
                CrossQ.policy_aliases[POLICY_TYPE],
                env=env,
                verbose=False,
                tensorboard_log=f"logs/{run.id}",
            )
            model.set_parameters(parameters)
            model.learn(
                total_timesteps=TOTAL_TIME_STEPS,
                callback=callbacks,
                reset_num_timesteps=False,
            )

            parameters = model.get_parameters()
            env.env_method("update_opponent", parameters)

        success = True
    finally:
        run.finish(exit_code=int(not success))


if __name__ == "__main__":
    main()
