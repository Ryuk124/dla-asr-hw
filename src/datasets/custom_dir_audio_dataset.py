from pathlib import Path

import soundfile as sf

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        audio_dir = Path(audio_dir)
        transcription_dir = Path(transcription_dir) if transcription_dir else None
        data = []

        for path in sorted(audio_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in (
                ".mp3",
                ".wav",
                ".flac",
                ".m4a",
            ):
                continue

            info = sf.info(path)
            entry = {
                "path": str(path),
                "text": "",
                "audio_len": info.frames / info.samplerate,
            }

            if transcription_dir is not None:
                transc_path = transcription_dir / (path.stem + ".txt")
                if transc_path.is_file():
                    entry["text"] = transc_path.read_text(encoding="utf-8").strip()

            data.append(entry)

        if not data:
            raise ValueError(f"No audio files found in {audio_dir}")

        super().__init__(data, *args, **kwargs)
