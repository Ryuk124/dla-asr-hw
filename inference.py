import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)
    device = config.inferencer.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    text_encoder = instantiate(config.text_encoder)
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)
    model = instantiate(config.model, n_tokens=len(text_encoder)).to(device)
    metrics = {
        "inference": [
            instantiate(metric, text_encoder=text_encoder)
            for metric in config.metrics.get("inference", [])
        ]
    }
    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        text_encoder=text_encoder,
        batch_transforms=batch_transforms,
        save_path=ROOT_PATH / config.inferencer.save_path,
        metrics=metrics,
    )
    logs = inferencer.run_inference()
    for part, values in logs.items():
        for key, value in values.items():
            print(f"    {part}_{key}: {value}")
    print(f"Predictions saved to {inferencer.save_path}")


if __name__ == "__main__":
    main()
