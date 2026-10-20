from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]) -> dict[str, Any]:
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    spectrograms: list[torch.Tensor] = [
        item["spectrogram"].squeeze(0)
        if len(item["spectrogram"].shape) == 3
        else item["spectrogram"]
        for item in dataset_items
    ]
    text_encoded: list[torch.Tensor] = [
        item["text_encoded"].squeeze(0)
        if len(item["text_encoded"].shape) == 2
        else item["text_encoded"]
        for item in dataset_items
    ]

    spectrogram_padded = pad_sequence(
        [spectrogram.transpose(0, 1) for spectrogram in spectrograms],
        batch_first=True,
        padding_value=0,
        padding_side="right",
    ).transpose(1, 2)  # [B, C, T_max_audio]

    spectrogram_lens = torch.tensor(
        [spectrogram.shape[-1] for spectrogram in spectrograms], dtype=torch.long
    )  # [B]

    text_encoded_padded = pad_sequence(
        text_encoded,
        batch_first=True,
        padding_value=0,
        padding_side="right",
    ).long()  # [B, T_max_text]

    text_encoded_lens = torch.tensor(
        [item.shape[-1] for item in text_encoded], dtype=torch.long
    )  # [B]

    # For debug and demonstration only. Readme says that we need to show it in logs
    audios = pad_sequence(
        [item["audio"].transpose(0, 1) for item in dataset_items],
        batch_first=True,
        padding_value=0,
        padding_side="right",
    ).transpose(1, 2)  # [B, 1, T_max_raw_audio]

    audio_lens = torch.tensor(
        [item["audio"].shape[-1] for item in dataset_items], dtype=torch.long
    )  # [B]

    batch = {
        "audio": audios,
        "audio_lens": audio_lens,
        "spectrogram": spectrogram_padded,
        "spectrogram_length": spectrogram_lens,
        "text_encoded": text_encoded_padded,
        "text_encoded_length": text_encoded_lens,
        "text": [item["text"] for item in dataset_items],
        "audio_path": [item["audio_path"] for item in dataset_items],
        "has_transcription": [
            item.get("has_transcription", True) for item in dataset_items
        ],
    }

    return batch
