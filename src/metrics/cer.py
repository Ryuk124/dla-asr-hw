import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.beam_search import BeamSearchMetric
from src.metrics.utils import calc_cer

# TODO add lm versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    @torch.no_grad()
    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: list[str], **kwargs
    ):
        cers = []
        predictions = torch.argmax(log_probs, dim=-1).cpu().numpy()
        lengths = log_probs_length.detach().cpu().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BeamSearchMetric):
    """Compute mean utterance CER from shared beam predictions."""

    @torch.no_grad()
    def __call__(
        self,
        log_probs: Tensor,
        log_probs_length: Tensor,
        text: list[str],
        beam_predictions=None,
        **kwargs,
    ):
        predictions = self.get_predictions(
            log_probs, log_probs_length, beam_predictions
        )
        errors = [
            calc_cer(self.text_encoder.normalize_text(target), prediction)
            for target, prediction in zip(text, predictions)
        ]
        return sum(errors) / len(errors)
