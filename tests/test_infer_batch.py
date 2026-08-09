"""Batch inference failure-handling regression tests."""

import tempfile
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

from omnivoice.cli import infer_batch


class _Queue:
    def put(self, _value):
        return None


class _Manager:
    def Queue(self):
        return _Queue()


class _FailingExecutor:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False

    def submit(self, *_args, **_kwargs):
        future = Future()
        future.set_exception(RuntimeError("worker failed"))
        return future


class InferBatchFailureTests(unittest.TestCase):
    def test_failed_future_makes_cli_exit_nonzero_without_process_group_kill(self) -> None:
        """A partial batch failure must fail the command after safe executor shutdown."""
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                res_dir=directory,
                test_list="input.jsonl",
                lang_id=None,
                nj_per_gpu=1,
                model="model",
                warmup=0,
                batch_size=1,
                batch_duration=1000.0,
            )
            parser = SimpleNamespace(parse_args=lambda: args)
            sample = {"id": "bad", "text": "hello"}

            with patch.object(infer_batch, "get_parser", return_value=parser), patch.object(
                infer_batch, "get_best_device_with_count", return_value=("cpu", 1)
            ), patch.object(infer_batch.mp, "Manager", return_value=_Manager()), patch.object(
                infer_batch, "read_test_list", return_value=[sample]
            ), patch.object(
                infer_batch, "cluster_samples_by_batch_size", side_effect=lambda subset, *_args: [subset]
            ), patch.object(
                infer_batch, "ProcessPoolExecutor", _FailingExecutor
            ), patch.object(
                infer_batch, "as_completed", side_effect=lambda futures: futures
            ), patch.object(
                infer_batch, "tqdm", side_effect=lambda iterable, **_kwargs: iterable
            ), patch.object(infer_batch.os, "killpg", create=True) as killpg:
                with self.assertRaises(SystemExit) as raised:
                    infer_batch.main()

            self.assertEqual(raised.exception.code, 1)
            killpg.assert_not_called()


if __name__ == "__main__":
    unittest.main()
