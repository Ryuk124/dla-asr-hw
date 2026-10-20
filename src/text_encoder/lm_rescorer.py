import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class NBestRescorer:
    def __init__(
        self,
        model_dir,
        neural_weight=0.3,
        ngram_weight=1.0,
        word_bonus=0.0,
        batch_size=64,
        device="cuda",
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_dir = Path(model_dir)
        self.neural_weight = neural_weight
        self.ngram_weight = ngram_weight
        self.word_bonus = word_bonus
        self.batch_size = batch_size
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = None
        self.tokenizer = None

    def load_model(self):
        if self.model is not None:
            return

        import sentencepiece as spm

        previous_quirks = os.environ.get("SB_DISABLE_QUIRKS")
        os.environ["SB_DISABLE_QUIRKS"] = ",".join(
            filter(
                None,
                [
                    previous_quirks,
                    "allow_tf32,disable_jit_profiling,disable_cudnn_benchmarking",
                ],
            )
        )
        try:
            from speechbrain.lobes.models.transformer.TransformerLM import TransformerLM
        finally:
            if previous_quirks is None:
                os.environ.pop("SB_DISABLE_QUIRKS", None)
            else:
                os.environ["SB_DISABLE_QUIRKS"] = previous_quirks

        tokenizer = spm.SentencePieceProcessor(
            model_file=str(self.model_dir / "tokenizer.ckpt")
        )
        if tokenizer.vocab_size() != 5000 or (
            tokenizer.bos_id(),
            tokenizer.eos_id(),
        ) != (1, 2):
            raise ValueError("Expected the SpeechBrain LibriSpeech 5000-token LM")
        with torch.random.fork_rng(devices=[]):
            model = TransformerLM(
                vocab=5000,
                d_model=768,
                nhead=12,
                num_encoder_layers=12,
                num_decoder_layers=0,
                d_ffn=3072,
                dropout=0.0,
                activation=torch.nn.GELU,
                normalize_before=False,
            )
            model.load_state_dict(
                torch.load(
                    self.model_dir / "lm.ckpt", map_location="cpu", weights_only=True
                )
            )
        self.model = model.eval().requires_grad_(False).to(self.device)
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def score(self, texts):
        if not texts:
            return []
        self.load_model()
        tokens = [self.tokenizer.encode(text.upper()) for text in texts]
        if any(len(ids) + 1 > 2500 for ids in tokens):
            raise ValueError("A hypothesis exceeds the LM context of 2500 tokens")
        order = sorted(range(len(tokens)), key=lambda i: len(tokens[i]))
        scores = [0.0] * len(texts)

        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            lengths = torch.tensor(
                [len(tokens[i]) + 1 for i in indices], device=self.device
            )
            batch = pad_sequence(
                [torch.tensor([1] + tokens[i] + [2]) for i in indices],
                batch_first=True,
            ).to(self.device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            max_length = inputs.size(1)
            logits = self.model(inputs)
            token_scores = -F.cross_entropy(
                logits.transpose(1, 2).float(), targets, reduction="none"
            )
            mask = (
                torch.arange(max_length, device=self.device)[None, :] < lengths[:, None]
            )
            batch_scores = token_scores.masked_fill(~mask, 0.0).sum(-1).cpu().tolist()
            for index, score in zip(indices, batch_scores):
                scores[index] = score
        return scores

    def __call__(self, hypotheses):
        if not hypotheses:
            return []
        if any(not candidates for candidates in hypotheses):
            raise ValueError("Each utterance must have at least one hypothesis")
        texts = list(dict.fromkeys(h["text"] for group in hypotheses for h in group))
        neural_scores = (
            dict(zip(texts, self.score(texts)))
            if self.neural_weight != 0
            else dict.fromkeys(texts, 0.0)
        )
        predictions = []
        for candidates in hypotheses:
            best = max(
                candidates,
                key=lambda h: (
                    h["acoustic_score"]
                    + self.ngram_weight * (h["beam_score"] - h["acoustic_score"])
                    + self.neural_weight * neural_scores[h["text"]]
                    + self.word_bonus * len(h["text"].split())
                ),
            )
            predictions.append(best["text"])
        return predictions
