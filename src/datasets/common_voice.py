import csv
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.text_encoder import CTCTextEncoder
from src.utils.io_utils import ROOT_PATH


class CommonVoiceDataset(BaseDataset):
    def __init__(self, data_dir, split = "train", **kwargs):
        data_dir = Path(data_dir).expanduser()
        self._data_dir = (ROOT_PATH / data_dir).resolve()
        if not self._data_dir.is_dir():
            raise FileNotFoundError(
                f"Common Voice directory not found: {self._data_dir}"
            )

        index = self._get_or_load_index(split)
        index = [
            {**entry, "path": str(self._data_dir / "clips" / Path(entry["path"]).name)}
            for entry in index
        ]
        super().__init__(index, **kwargs)

    def _get_or_load_index(self, split):
        index_path = self._data_dir / f"{split}_index.json"
        if index_path.is_file():
            with index_path.open(encoding="utf-8") as file:
                return json.load(file)

        tsv_path = self._data_dir / f"{split}.tsv"
        if not tsv_path.is_file():
            raise FileNotFoundError(
                f"Common Voice split {split!r} requires {index_path} or {tsv_path}."
            )

        index = self._build_index(tsv_path)
        temporary_path = index_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(index, file, ensure_ascii=False, indent=2)
        temporary_path.replace(index_path)
        return index

    def _build_index(self, tsv_path):
        index = []
        with tsv_path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file, delimiter="\t")
            if not {"path", "sentence"}.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"{tsv_path} must contain 'path' and 'sentence' columns."
                )

            for entry in tqdm(reader, desc=f"Indexing Common Voice {tsv_path.stem}"):
                text = CTCTextEncoder.normalize_text(entry["sentence"] or "")
                if not text.strip():
                    continue
                audio_path = self._data_dir / "clips" / entry["path"]
                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f"Common Voice audio not found: {audio_path}"
                    )
                index.append(
                    {
                        "path": audio_path.name,
                        "text": text,
                        "audio_len": sf.info(audio_path).duration,
                    }
                )

        if not index:
            raise ValueError(f"No usable transcripts found in {tsv_path}.")
        return index
