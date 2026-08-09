"""Regression tests for distributed evaluation and checkpoint rotation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from omnivoice.training.checkpoint import save_checkpoint
from omnivoice.training.trainer import OmniTrainer


class _EvalModel:
    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, **_batch):
        return SimpleNamespace(loss=torch.tensor(2.0))


class _EvalAccelerator:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.logged = None

    def gather(self, _local_stats):
        # rank 0: loss_sum=2 over 1 batch; rank 1: loss_sum=8 over 3 batches.
        return torch.tensor([2.0, 1.0, 8.0, 3.0])

    def log(self, metrics, step):
        self.logged = (metrics, step)

    def wait_for_everyone(self):
        return None


class _CheckpointModel:
    def save_pretrained(self, checkpoint_dir, **_kwargs):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        (Path(checkpoint_dir) / "model.bin").write_bytes(b"model")


class _CheckpointTokenizer:
    def save_pretrained(self, checkpoint_dir):
        (Path(checkpoint_dir) / "tokenizer.json").write_text("{}", encoding="utf-8")


class _CheckpointAccelerator:
    is_main_process = True

    def save_state(self, checkpoint_dir):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def unwrap_model(self, model):
        return model

    def save(self, *_args, **_kwargs):
        return None


class TrainingRuntimeTests(unittest.TestCase):
    def test_evaluation_weights_ranks_by_batch_count(self) -> None:
        """Aggregate global loss sums/counts instead of averaging rank-local means."""
        trainer = OmniTrainer.__new__(OmniTrainer)
        trainer.eval_dataloader = [{}]
        trainer.model = _EvalModel()
        trainer.accelerator = _EvalAccelerator()
        trainer.global_step = 7

        metrics = trainer.evaluate()

        self.assertAlmostEqual(metrics["eval/loss"], 2.5)
        self.assertEqual(trainer.accelerator.logged, ({"eval/loss": 2.5}, 7))

    def test_checkpoint_rotation_ignores_non_numeric_checkpoint_directories(self) -> None:
        """Keep named checkpoint directories without crashing numeric rotation."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "checkpoint-1").mkdir()
            (output / "checkpoint-final").mkdir()

            save_checkpoint(
                _CheckpointAccelerator(),
                _CheckpointModel(),
                _CheckpointTokenizer(),
                str(output),
                step=10,
                keep_last_n=1,
            )

            self.assertFalse((output / "checkpoint-1").exists())
            self.assertTrue((output / "checkpoint-10").is_dir())
            self.assertTrue((output / "checkpoint-final").is_dir())


if __name__ == "__main__":
    unittest.main()
