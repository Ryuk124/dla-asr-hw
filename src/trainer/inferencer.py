import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.metrics.beam_search import BeamSearchMetric
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text_encoder (CTCTextEncoder): text encoder.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        self.config = config
        self.cfg_trainer = config.inferencer
        self.device = device
        self.model = model
        self.batch_transforms = batch_transforms or {}
        self.text_encoder = text_encoder
        self.evaluation_dataloaders = dataloaders
        self.save_path = Path(save_path) if save_path is not None else None
        self.metrics = (metrics or {}).get("inference", [])
        self.evaluation_metrics = MetricTracker(*[m.name for m in self.metrics])
        self.decoding = self.cfg_trainer.get("decoding", {})
        self.method = self.decoding.get("method", "greedy")
        if self.method not in ("greedy", "beam", "beam_lm", "rescore"):
            raise ValueError(f"Unknown decoding method: {self.method}")
        self.decoder = BeamSearchMetric(
            text_encoder,
            beam_size=self.decoding.get("beam_size", 100),
            token_top_k=self.decoding.get("token_top_k"),
            rescore=self.method == "rescore",
            nbest_size=self.decoding.get("nbest_size", 100),
        )
        if self.decoder.beam_size < 1:
            raise ValueError("beam_size must be positive")
        if (
            self.method == "rescore"
            and not 1 <= self.decoder.nbest_size <= self.decoder.beam_size
        ):
            raise ValueError("Expected 1 <= nbest_size <= beam_size")
        if not skip_model_load:
            checkpoint = self.cfg_trainer.get("from_pretrained")
            if checkpoint is None:
                raise ValueError("Provide inferencer.from_pretrained")
            self._from_pretrained(checkpoint)

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        return {
            part: self._inference_part(part, dataloader)
            for part, dataloader in self.evaluation_dataloaders.items()
        }

    def _decode(self, log_probs, lengths, greedy):
        if self.method == "greedy":
            return greedy, {}
        if self.method == "beam":
            predictions = [
                self.text_encoder.beam_search(
                    sample[:length],
                    beam_size=self.decoder.beam_size,
                    token_top_k=self.decoder.token_top_k,
                )
                for sample, length in zip(log_probs, lengths.tolist())
            ]
            return predictions, {}
        predictions = self.decoder.get_predictions(log_probs, lengths)
        return predictions, {self.decoder.decoding_key: predictions}

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        batch.update(self.model(**batch))
        log_probs = batch["log_probs"].detach().float().cpu()
        lengths = batch["log_probs_length"].detach().cpu()
        greedy = [
            self.text_encoder.ctc_decode(tokens[:length])
            for tokens, length in zip(log_probs.argmax(-1).tolist(), lengths.tolist())
        ]
        predictions, decoded = self._decode(log_probs, lengths, greedy)
        has_transcription = batch.get("has_transcription", [True] * len(predictions))
        indices = [i for i, present in enumerate(has_transcription) if present]
        if indices and self.metrics and metrics is not None:
            labeled = {
                "log_probs": log_probs[indices],
                "log_probs_length": lengths[indices],
                "text": [batch["text"][i] for i in indices],
                "beam_predictions": {
                    key: [texts[i] for i in indices] for key, texts in decoded.items()
                },
            }
            for metric in self.metrics:
                if isinstance(metric, BeamSearchMetric):
                    labeled["beam_predictions"][metric.decoding_key] = (
                        metric.get_predictions(
                            labeled["log_probs"],
                            labeled["log_probs_length"],
                            labeled["beam_predictions"],
                        )
                    )
                metrics.update(metric.name, metric(**labeled), n=len(indices))
            self.labeled_count += len(indices)

        if self.save_path is not None:
            output_dir = self.save_path / part
            with (output_dir / "predictions.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                for i, prediction in enumerate(predictions):
                    utterance_id = Path(batch["audio_path"][i]).stem
                    with (output_dir / f"{utterance_id}.txt").open(
                        "x", encoding="utf-8"
                    ) as text_file:
                        text_file.write(prediction + "\n")
                    output = {
                        "id": utterance_id,
                        "audio_path": str(batch["audio_path"][i]),
                        "reference": batch["text"][i] if has_transcription[i] else None,
                        "prediction": prediction,
                        "greedy": greedy[i],
                        "beam_predictions": [
                            {
                                "beam_size": key[0],
                                "rescore": key[2],
                                "nbest_size": key[3],
                                "text": texts[i],
                            }
                            for key, texts in decoded.items()
                        ],
                    }
                    handle.write(json.dumps(output, ensure_ascii=False) + "\n")
        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        self.labeled_count = 0
        if self.save_path is not None:
            output_dir = self.save_path / part
            output_dir.mkdir(exist_ok=True, parents=True)
            if any(output_dir.iterdir()):
                raise FileExistsError(f"Output directory is not empty: {output_dir}")
            (output_dir / "predictions.jsonl").touch()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=part)):
                self.process_batch(batch_idx, batch, self.evaluation_metrics, part)
        return self.evaluation_metrics.result() if self.labeled_count else {}
