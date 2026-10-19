import sentencepiece as spm
import torch

from src.text_encoder.ctc_text_encoder import CTCTextEncoder
from src.utils.io_utils import ROOT_PATH


class BPETextEncoder(CTCTextEncoder):
    def __init__(self, model_path, use_lm=False, **kwargs):
        self.tokenizer = spm.SentencePieceProcessor(
            model_file=str(ROOT_PATH / model_path)
        )
        if self.tokenizer.unk_id() != 0 or any(
            token_id != -1
            for token_id in (
                self.tokenizer.bos_id(),
                self.tokenizer.eos_id(),
                self.tokenizer.pad_id(),
            )
        ):
            raise ValueError("Expected unk_id=0 and disabled BOS, EOS, PAD")

        alphabet = [
            self.tokenizer.id_to_piece(i)
            for i in range(self.tokenizer.get_piece_size())
        ]
        super().__init__(alphabet=alphabet, use_lm=use_lm, **kwargs)

    def encode(self, text) -> torch.Tensor:
        token_ids = self.tokenizer.encode(self.normalize_text(text))
        return torch.tensor([i + 1 for i in token_ids], dtype=torch.long).unsqueeze(0)

    def decode(self, inds) -> str:
        token_ids = [int(i) - 1 for i in inds if int(i) != 0]
        return self.normalize_text(self.tokenizer.decode(token_ids))

    @staticmethod
    def normalize_text(text: str):
        return " ".join(CTCTextEncoder.normalize_text(text).split())
