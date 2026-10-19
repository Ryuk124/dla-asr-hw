import argparse
import hashlib
import json
from pathlib import Path
from string import ascii_lowercase

import sentencepiece as spm

from src.text_encoder import BPETextEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("tokenizers/librispeech_bpe_1024")
    )
    args = parser.parse_args()

    sentences = []
    sources = {}
    digest = hashlib.sha256()
    for data_dir in args.data_dirs:
        if data_dir.name not in (
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ):
            raise ValueError(
                f"Expected a LibriSpeech train split directory: {data_dir}"
            )
        files = sorted(data_dir.glob("*/*/*.trans.txt"))
        if not files:
            raise FileNotFoundError(f"No transcripts found in {data_dir}")
        count = 0
        for path in files:
            for line in path.read_text().splitlines():
                fields = line.split(maxsplit=1)
                if len(fields) != 2:
                    raise ValueError(f"Invalid transcript in {path}: {line!r}")
                text = BPETextEncoder.normalize_text(fields[1])
                if text:
                    sentences.append(text)
                    digest.update((text + "\n").encode())
                    count += 1
        sources[str(data_dir)] = count

    prefix = args.output_prefix
    for suffix in (".model", ".vocab", ".json"):
        if Path(str(prefix) + suffix).exists():
            raise FileExistsError(f"Tokenizer already exists: {prefix}{suffix}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        required_chars=ascii_lowercase,
        normalization_rule_name="identity",
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        shuffle_input_sentence=False,
        num_threads=1,
    )
    metadata = {
        "model_type": "bpe",
        "vocab_size": args.vocab_size,
        "ctc_vocab_size": args.vocab_size + 1,
        "blank_id": 0,
        "sentencepiece_id_offset": 1,
        "sources": sources,
        "sentences": len(sentences),
        "corpus_sha256": digest.hexdigest(),
        "sentencepiece_version": spm.__version__,
    }
    Path(str(prefix) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
