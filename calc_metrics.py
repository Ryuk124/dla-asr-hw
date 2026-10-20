import argparse
import json
import re
from pathlib import Path

import editdistance


def normalize_text(text):
    return " ".join(re.sub(r"[^a-z ]", "", text.lower()).split())


def calc_metrics(references, predictions):
    references = Path(references)
    predictions = Path(predictions)
    if not references.is_dir() or not predictions.is_dir():
        raise ValueError("References and predictions must be directories")
    reference_files = {path.stem: path for path in references.glob("*.txt")}
    prediction_files = {path.stem: path for path in predictions.glob("*.txt")}
    if not reference_files:
        raise ValueError(f"No transcriptions found in {references}")
    missing = reference_files.keys() - prediction_files.keys()
    extra = prediction_files.keys() - reference_files.keys()
    if missing or extra:
        raise ValueError(
            f"Utterance IDs do not match: {len(missing)} missing, {len(extra)} extra. "
            f"Missing: {sorted(missing)[:5]}; extra: {sorted(extra)[:5]}"
        )

    word_errors = char_errors = words = chars = 0
    for utterance_id, path in reference_files.items():
        reference = normalize_text(path.read_text(encoding="utf-8"))
        prediction = normalize_text(
            prediction_files[utterance_id].read_text(encoding="utf-8")
        )
        word_errors += editdistance.eval(reference.split(), prediction.split())
        char_errors += editdistance.eval(reference, prediction)
        words += len(reference.split())
        chars += len(reference)
    if words == 0 or chars == 0:
        raise ValueError("References contain no words after normalization")
    return {
        "utterances": len(reference_files),
        "WER": word_errors / words,
        "CER": char_errors / chars,
        "word_errors": word_errors,
        "words": words,
        "char_errors": char_errors,
        "chars": chars,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = calc_metrics(args.references, args.predictions)
    print(f"WER: {metrics['WER']:.2%}")
    print(f"CER: {metrics['CER']:.2%}")
    print(f"Utterances: {metrics['utterances']}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
