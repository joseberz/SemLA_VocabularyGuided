import logging

import numpy as np
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime

from detectron2.utils.logger import log_every_n


class AccumLRScheduler(HookBase):
    """
    Step LR scheduler only when an optimizer step actually happened.
    Expects trainer._trainer.did_step to be set by the inner trainer.
    """
    def __init__(self, optimizer, scheduler):
        self._optimizer = optimizer
        self._scheduler = scheduler

    def after_step(self):
        # only step scheduler if optimizer really stepped this iter
        inner = self.trainer._trainer
        if getattr(inner, "did_step", False):
            self._scheduler.step()

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss_original(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar("loss/val", mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # Val-Loss braucht train-mode, sonst liefert das Modell Predictions (list) statt Loss-Dict.
        was_training = self._model.training
        self._model.eval()

        try:
            with torch.no_grad():
                metrics_dict = self._model(data)

            # manche Modelle geben direkt Tensor zurück
            if isinstance(metrics_dict, torch.Tensor):
                return float(metrics_dict.detach().cpu().item())

            # safety: falls doch Predictions kommen, fail fast mit Hinweis
            if isinstance(metrics_dict, list):
                raise TypeError(
                    "Model returned a list (predictions) instead of a loss dict. "
                    "This usually means the model is still in eval mode somewhere."
                )

            # normaler Fall: dict[str, Tensor/float]
            metrics_dict = {
                k: (v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v))
                for k, v in metrics_dict.items()
            }
            return float(sum(metrics_dict.values()))
        finally:
            # ursprünglichen Zustand wiederherstellen
            self._model.train(was_training)

    def _get_loss_original(self, data):
        # How loss is calculated on train_loop
        with torch.no_grad():
            metrics_dict = self._model(data)

        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced


    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

    def _do_loss_eval_old(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        with inference_context(self._model), torch.no_grad():
            for idx, inputs in enumerate(self._data_loader):
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0

                start_compute_time = time.perf_counter()
                loss_batch = self._get_loss_original(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                losses.append(loss_batch)
        mean_loss = float(np.mean(losses)) if len(losses) else 0.0
        #self.trainer.storage.put_scalar('validation_loss', mean_loss)
        self.trainer.storage.put_scalar("loss/val", mean_loss)
        comm.synchronize()

        return losses