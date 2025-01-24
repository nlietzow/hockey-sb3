import gymnasium as gym
import wandb
from hockey import REGISTERED_ENVS
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

assert REGISTERED_ENVS, "Hockey environments are not registered."

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 500_000,
    "env_name": "Hockey-One-v0",
}
run = wandb.init(
    project="hockey-sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)


def make_env():
    _env = gym.make(config["env_name"])
    _env = Monitor(_env)  # record stats such as returns
    return _env


env = DummyVecEnv([make_env])
model = SAC(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
run.finish()
