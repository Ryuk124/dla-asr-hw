import torch

from src.utils.io_utils import ROOT_PATH


def load_encoder(model, checkpoint_path):
    checkpoint = torch.load(
        ROOT_PATH / checkpoint_path, map_location="cpu", weights_only=False
    )
    state_dict = {
        key.removeprefix("encoder."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("encoder.")
    }
    model.encoder.load_state_dict(state_dict, strict=True)
