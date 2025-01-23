import logging

import wandb
from hockey.hockey_env import HockeyEnv_BasicOpponent
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

logging.basicConfig(level=logging.INFO)

# Configuration for the training
CONFIG = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 100_000,
    "env_name": "HockeyEnv_BasicOpponent",
}

# Initialize WandB
run = wandb.init(
    project="hockey-sb3",
    config=CONFIG,
    monitor_gym=True,  # Automatically upload gym monitor stats
    save_code=True,  # Save the code to the WandB run
)


# Function to create and monitor the environment
def make_env():
    _env = HockeyEnv_BasicOpponent()  # Initialize your custom environment
    _env = Monitor(_env)  # Wrap it to record stats like rewards
    return _env


# Wrap the environment
env = DummyVecEnv([make_env])  # Use DummyVecEnv for compatibility with stable-baselines3

# Set a consistent random seed for reproducibility
set_random_seed(42)

# Initialize the SAC model
model = SAC(CONFIG["policy_type"], env, verbose=1)

# Train the model
try:
    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,  # Save gradients every 100 steps
            model_save_path=f"models/{run.id}",  # Save the model to this path
            verbose=2,  # Log detailed information to the terminal
        ),
    )
    model.save(f"models/sac_hockey_{run.id}")
    logging.info("Model training and saving completed successfully.")
except Exception as e:
    logging.error(f"Training failed: {e}", exc_info=True)
finally:
    env.close()  # Ensure the environment is closed properly
    run.finish()  # Finalize the WandB run
