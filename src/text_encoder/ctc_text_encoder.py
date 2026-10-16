import heapq
import re
from string import ascii_lowercase

from pyctcdecode import build_ctcdecoder

import numpy as np
import torch

# TODO add BPE, LM support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        labels = [self.ind2char[i] for i in range(len(self.ind2char))]
        labels = ["" if token == self.EMPTY_TOK else token.upper() for token in labels]

        self.lm_decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=(
                "/storage/lam-tts-resources-2/experiments/VTTS-4688/"
                "librispeech_lm/3-gram.pruned.1e-7.arpa"
            ),
            alpha=0.5,
            beta=1.0,
        )

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.tensor(
                [self.char2ind[char] for char in text], dtype=torch.long
            ).unsqueeze(0)
        except KeyError:
            unknown_chars = {char for char in text if char not in self.char2ind}
            raise AttributeError(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        result = ""

        for i, ind in enumerate(inds):
            if self.ind2char[ind] == self.EMPTY_TOK:
                continue

            if i > 0 and ind == inds[i - 1]:
                continue

            result += self.ind2char[ind]

        return result.strip()

    # My own beam search implementation. More basic realisation was in commit 7ec7c8f4660ed8bd099a38da1a6f3fde647ab527
    def beam_search(
        self, logprobs, beam_size: int = 1, token_top_k: int | None = None
    ) -> str:
        """Decode one utterance with CTC prefix beam search.

        Args:
            logprobs: Tensor of shape [T, vocab_size], without padded frames.
                Values are log probabilities, including the blank token.
            beam_size: Number of prefixes retained after each frame.
            token_top_k: Maximum number of top-scoring tokens considered per
                frame, plus blank if absent. None considers the full vocabulary.
                Pruning is approximate and can change the decoded text.

        Returns:
            str: The most probable retained prefix converted to text.

        Raises:
            ValueError: If a beam/pruning size is invalid or the input shape
                does not match the vocabulary.
        """
        if (
            not isinstance(beam_size, int)
            or isinstance(beam_size, bool)
            or beam_size < 1
        ):
            raise ValueError("beam_size must be a positive integer")
        if token_top_k is not None and (
            not isinstance(token_top_k, int)
            or isinstance(token_top_k, bool)
            or token_top_k < 1
        ):
            raise ValueError("token_top_k must be a positive integer or None")
        if logprobs.ndim != 2 or logprobs.shape[1] != len(self):
            raise ValueError("logprobs must have shape [T, vocab_size]")

        logprobs = logprobs.detach().float().cpu().numpy()
        blank_id = self.char2ind[self.EMPTY_TOK]
        empty_scores = (float("-inf"), float("-inf"))
        beams = {(): (0.0, float("-inf"))}
        full_vocabulary = range(len(self))
        logaddexp = np.logaddexp

        for frame_lps in logprobs:
            if token_top_k is None or token_top_k >= len(self):
                token_ids = full_vocabulary
            else:
                token_ids = np.argpartition(frame_lps, -token_top_k)[-token_top_k:]
                # Keep blank and a stable token order for deterministic ties.
                token_ids = sorted(set(token_ids.tolist()) | {blank_id})
            candidates = [
                (token_id, float(frame_lps[token_id])) for token_id in token_ids
            ]
            next_beams = {}

            for prefix, (log_p_blank, log_p_nonblank) in beams.items():
                log_p_total = logaddexp(log_p_blank, log_p_nonblank)
                last_token = prefix[-1] if prefix else None

                for token_id, token_logprob in candidates:
                    if token_id == blank_id:
                        old_blank, old_nonblank = next_beams.get(prefix, empty_scores)
                        next_beams[prefix] = (
                            logaddexp(old_blank, log_p_total + token_logprob),
                            old_nonblank,
                        )
                        continue

                    # A repeated token extends the prefix only after blank.
                    source_score = (
                        log_p_blank if token_id == last_token else log_p_total
                    )
                    new_prefix = prefix + (token_id,)
                    old_blank, old_nonblank = next_beams.get(new_prefix, empty_scores)
                    next_beams[new_prefix] = (
                        old_blank,
                        logaddexp(old_nonblank, source_score + token_logprob),
                    )

                    if token_id == last_token:
                        old_blank, old_nonblank = next_beams.get(prefix, empty_scores)
                        next_beams[prefix] = (
                            old_blank,
                            logaddexp(old_nonblank, log_p_nonblank + token_logprob),
                        )

            beams = dict(
                heapq.nlargest(
                    beam_size,
                    next_beams.items(),
                    key=lambda item: logaddexp(*item[1]),
                )
            )

        best_prefix = max(beams, key=lambda prefix: logaddexp(*beams[prefix]))
        return self.decode(best_prefix)

    def beam_search_lm(self, logprobs, beam_size: int = 1) -> str:
            logprobs = logprobs.detach().float().cpu().numpy()
            text = self.lm_decoder.decode(
                logprobs,
                beam_width=beam_size,
            )
            return self.normalize_text(text)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
