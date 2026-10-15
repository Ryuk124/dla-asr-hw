import re
from string import ascii_lowercase

import torch
import numpy as np

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

    def beam_search(self, logprobs, beam_size: int = 1) -> str:
        beams = {
            (): (0.0, float("-inf"))  # prefx ids: (logp_blank, logp_nonblank)
        }
        logprobs = logprobs.detach().float().cpu().numpy()

        for frame_lps in logprobs:
            next_beams = {}
            for prefix, (log_p_blank, log_p_nonblank) in beams.items():
                for token_id, token_logprob in enumerate(frame_lps):
                    if self.ind2char[token_id] == self.EMPTY_TOK:
                        old_blank, old_nonblank = next_beams.get(
                            prefix, (float("-inf"), float("-inf"))
                        )
                        blank_score = (
                            np.logaddexp(log_p_blank, log_p_nonblank) + token_logprob
                        )
                        next_beams[prefix] = (
                            np.logaddexp(old_blank, blank_score),
                            old_nonblank,
                        )
                    else:
                        if not prefix or prefix[-1] != token_id:
                            new_prefix = prefix + (token_id,)
                            score = (
                                np.logaddexp(log_p_blank, log_p_nonblank)
                                + token_logprob
                            )

                            old_blank, old_nonblank = next_beams.get(
                                new_prefix, (float("-inf"), float("-inf"))
                            )
                            next_beams[new_prefix] = (
                                old_blank,
                                np.logaddexp(old_nonblank, score),
                            )
                        else:
                            new_prefix = prefix + (token_id,)

                            old_blank, old_nonblank = next_beams.get(
                                new_prefix, (float("-inf"), float("-inf"))
                            )
                            next_beams[new_prefix] = (
                                old_blank,
                                np.logaddexp(old_nonblank, log_p_blank + token_logprob),
                            )

                            old_blank, old_nonblank = next_beams.get(
                                prefix, (float("-inf"), float("-inf"))
                            )
                            next_beams[prefix] = (
                                old_blank,
                                np.logaddexp(
                                    old_nonblank, log_p_nonblank + token_logprob
                                ),
                            )

            beams = dict(
                sorted(
                    next_beams.items(),
                    key=lambda item: np.logaddexp(*item[1]),
                    reverse=True,
                )[:beam_size]
            )

        best_prefix = max(beams, key=lambda prefix: np.logaddexp(*beams[prefix]))
        return self.decode(best_prefix)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
