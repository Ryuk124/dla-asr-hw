import torch

from src.metrics.base_metric import BaseMetric


class BeamSearchMetric(BaseMetric):
    """Share decoded prefixes between metrics using the same decoder settings.

    Args:
        text_encoder: Encoder providing beam_search.
        beam_size: Number of retained prefixes.
        token_top_k: Per-frame token limit, or None for the full vocabulary.
        *args: Positional arguments for BaseMetric.
        **kwargs: Keyword arguments for BaseMetric.
    """

    def __init__(
        self,
        text_encoder,
        beam_size,
        *args,
        token_top_k=None,
        rescore=False,
        nbest_size=100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        self.token_top_k = token_top_k
        self.rescore = rescore
        self.nbest_size = nbest_size

    @property
    def decoding_key(self):
        """Return the settings identifying a shared prediction list."""
        return self.beam_size, self.token_top_k, self.rescore, self.nbest_size

    @torch.no_grad()
    def get_predictions(self, log_probs, log_probs_length, beam_predictions=None):
        """Reuse batch predictions or decode when called outside the trainer.

        Args:
            log_probs: Tensor of shape [B, T, vocab_size].
            log_probs_length: Valid output lengths of shape [B].
            beam_predictions: Optional batch-local mapping from decoder settings
                to decoded text lists.

        Returns:
            list[str]: One prediction per utterance, excluding padded frames.
        """
        if beam_predictions is not None and self.decoding_key in beam_predictions:
            return beam_predictions[self.decoding_key]

        predictions = log_probs.detach().float().cpu()
        lengths = log_probs_length.detach().cpu().tolist()
        if self.rescore:
            hypotheses = [
                self.text_encoder.beam_search_nbest(
                    sample[:length], self.beam_size, self.nbest_size
                )
                for sample, length in zip(predictions, lengths)
            ]
            return self.text_encoder.rescore(hypotheses)
        return [
            self.text_encoder.beam_search_lm(
                sample[:length],
                beam_size=self.beam_size,
            )
            for sample, length in zip(predictions, lengths)
        ]


@torch.no_grad()
def prepare_beam_predictions(batch, metric_funcs):
    """Attach one decoded text list per beam configuration to the current batch.

    Args:
        batch: Model batch containing log_probs and log_probs_length. Modified
            in place to hold beam_predictions for metrics and logging.
        metric_funcs: Active metrics, sharing the pipeline's text encoder.

    Raises:
        ValueError: If beam metrics use different text encoders.
    """
    beam_metrics = [
        metric for metric in metric_funcs if isinstance(metric, BeamSearchMetric)
    ]
    # Never reuse predictions from a previous pass through the same batch dict.
    batch.pop("beam_predictions", None)
    if not beam_metrics:
        return
    if any(
        metric.text_encoder is not beam_metrics[0].text_encoder
        for metric in beam_metrics
    ):
        raise ValueError("Shared beam metrics must use the same text encoder")

    predictions = batch["log_probs"].detach().float().cpu()
    lengths = batch["log_probs_length"].detach().cpu()
    decoded = {}
    for metric in beam_metrics:
        if metric.decoding_key not in decoded:
            decoded[metric.decoding_key] = metric.get_predictions(predictions, lengths)
    batch["beam_predictions"] = decoded
