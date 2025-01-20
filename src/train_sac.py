import wandb
from hockey.hockey_env import HockeyEnv_BasicOpponent
from stable_baselines3 import SAC
from wandb.integration.sb3 import WandbCallback


def main():
    env = HockeyEnv_BasicOpponent()
    env.seed(42)

    model = SAC(CONFIG["policy_type"], env, verbose=1)
    try:
        model.learn(
            total_timesteps=CONFIG["total_timesteps"],
            callback=WandbCallback(
                model_save_path=f"models/{run.id}",
                verbose=2,
            ),
        )
        model.save("sac_hockey")
    except Exception as e:
        print(f"Training failed: {e}")
    finally:
        env.close()
        run.finish()


if __name__ == '__main__':
    CONFIG = {
        "env": "HockeyEnv_BasicOpponent",
        "algorithm": "SAC",
        "policy_type": "MlpPolicy",
        "total_timesteps": 100_000,
    }
    run = wandb.init(
        project="hockey-sb3",
        config=CONFIG,
    )
    main()
