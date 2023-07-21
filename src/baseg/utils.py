from datetime import datetime
from pathlib import Path
import numpy as np

def exp_name_timestamp(name: str) -> str:
    """Generates a name for the experiment starting with the given name and
    appending the current date and time for uniqueness.

    Args:
        name (str): The name of the experiment.

    Returns:
        str: The name of the experiment with the current date and time appended.
    """
    now = datetime.now()
    return f"{name}_{now.strftime('%Y%m%d_%H%M%S')}"


def find_best_checkpoint(ckpt_path: Path, metric: str, mode: str = "min") -> Path:
    """Finds the best checkpoint in the given path based on the given metric.

    Args:
        ckpt_path (Path): The path to the checkpoint directory.
        metric (str): The metric to use for comparison.
        mode (str, optional): The mode to use for comparison. Defaults to "min".

    Returns:
        Path: The path to the best checkpoint.
    """
    assert ckpt_path.exists(), f"Checkpoint path does not exist: {ckpt_path}"
    assert mode in ["min", "max"], f"Invalid mode: {mode}"
    # find the best checkpoint
    best_ckpt = None
    best_value = None
    for ckpt in ckpt_path.glob("*.ckpt"):
        if ckpt.stem == "last":
            continue
        value = float(ckpt.stem.split("=")[-1])
        if best_value is None or (mode == "min" and value < best_value) or (mode == "max" and value > best_value):
            best_ckpt = ckpt
            best_value = value
    assert best_ckpt is not None, f"No checkpoint found in: {ckpt_path}"
    return best_ckpt


def mask2rgb(image: np.ndarray, palette: dict) -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for k, v in palette.items():
        lut[k, :] = v
    return lut[image]