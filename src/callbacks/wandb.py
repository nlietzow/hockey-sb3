from wandb.integration.sb3 import WandbCallback


def get_wandb_callback(run_id: str, verbose: int = 2):
    return WandbCallback(
        model_save_path=f"models/{run_id}",
        verbose=verbose,
    )
