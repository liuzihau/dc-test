"""Local, restart-safe metrics for the vendored Zebra training run.

This callback intentionally lives outside the vendored author source tree.  It
records the synchronized metrics that the released LightningModule already
logs, and it restores RNG state after validation so the extra validation
passes do not change the subsequent training-corruption sequence.
"""

from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


TRAIN_COLUMNS = (
    "optimizer_step",
    "epoch",
    "batch_idx",
    "train_loss",
    "main_elbo",
    "np_prev",
    "np_next",
    "learning_rate",
    "wall_time_seconds",
)

VALIDATION_METRICS = (
    "val/nll",
    "val/ppl",
    "val/bpd",
    "val/nll_var",
    "val/loss_bucket_0",
    "val/loss_bucket_1",
    "val/loss_bucket_2",
    "val/loss_bucket_3",
)

VALIDATION_COLUMNS = (
    "optimizer_step",
    "epoch",
    *(name.replace("/", "_") for name in VALIDATION_METRICS),
    "wall_time_seconds",
)


def _as_float(value: Any) -> float:
  if value is None:
    return math.nan
  if isinstance(value, torch.Tensor):
    if value.numel() != 1:
      return math.nan
    value = value.detach().cpu().item()
  try:
    return float(value)
  except (TypeError, ValueError):
    return math.nan


def _rewrite_through_step(path: Path, columns: Iterable[str], step: int) -> None:
  """Drop uncommitted rows beyond a resumed checkpoint's optimizer step."""
  if not path.exists():
    return
  with path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
  kept = []
  for row in rows:
    try:
      row_step = int(float(row["optimizer_step"]))
    except (KeyError, TypeError, ValueError):
      continue
    if row_step <= step:
      kept.append(row)
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=tuple(columns))
    writer.writeheader()
    writer.writerows(kept)


class AuthorLocalMetricsCallback(Callback):
  """Persist per-update train loss and per-validation aggregate metrics."""

  def __init__(self, output_dir: str, train_every_n_steps: int = 1,
               preserve_rng_around_validation: bool = True,
               validation_step_override: Optional[int] = None):
    super().__init__()
    if train_every_n_steps < 1:
      raise ValueError("train_every_n_steps must be positive")
    self.output_dir = Path(output_dir)
    self.train_every_n_steps = int(train_every_n_steps)
    self.preserve_rng_around_validation = bool(preserve_rng_around_validation)
    self.validation_step_override = (
        None if validation_step_override is None else int(validation_step_override))
    self._start_time = time.monotonic()
    self._rng_state: Optional[Dict[str, Any]] = None
    self._last_logged_step = 0
    self._pending_losses = []

  @property
  def train_path(self) -> Path:
    return self.output_dir / "train.csv"

  @property
  def validation_path(self) -> Path:
    return self.output_dir / "validation.csv"

  def _ensure_csv(self, path: Path, columns: Iterable[str]) -> None:
    self.output_dir.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
      return
    with path.open("w", newline="", encoding="utf-8") as stream:
      csv.writer(stream).writerow(tuple(columns))

  def _append(self, path: Path, columns: Iterable[str], row: Mapping[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as stream:
      writer = csv.DictWriter(stream, fieldnames=tuple(columns))
      writer.writerow(row)
      stream.flush()

  def setup(self, trainer, pl_module, stage: str) -> None:
    del pl_module
    if stage != "fit" or not trainer.is_global_zero:
      return
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self._ensure_csv(self.train_path, TRAIN_COLUMNS)
    self._ensure_csv(self.validation_path, VALIDATION_COLUMNS)

  def on_train_start(self, trainer, pl_module) -> None:
    del pl_module
    if not trainer.is_global_zero:
      return
    # Checkpoint restoration happens after `setup`.  Truncating there would see
    # global_step=0 and erase committed history.  `on_train_start` observes the
    # restored step and safely removes only rows from uncommitted replay work.
    resume_step = int(trainer.global_step)
    self._last_logged_step = resume_step
    if resume_step > 0:
      _rewrite_through_step(self.train_path, TRAIN_COLUMNS, resume_step)
      _rewrite_through_step(self.validation_path, VALIDATION_COLUMNS, resume_step)

  def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                         batch_idx: int) -> None:
    del pl_module, batch
    if not trainer.is_global_zero:
      return
    step = int(trainer.global_step)
    loss = trainer.callback_metrics.get("trainer/loss")
    if loss is None:
      loss = outputs.get("loss") if isinstance(outputs, Mapping) else outputs
    self._pending_losses.append({
        'train_loss': _as_float(loss),
        'main_elbo': _as_float(trainer.callback_metrics.get('components/current_elbo')),
        'np_prev': _as_float(trainer.callback_metrics.get('components/np_prev_1')),
        'np_next': _as_float(trainer.callback_metrics.get('components/np_next_1'))})
    if step <= self._last_logged_step:
      return
    averaged = {key: sum(r[key] for r in self._pending_losses) / len(self._pending_losses)
                for key in self._pending_losses[0]}
    self._pending_losses.clear()
    self._last_logged_step = step
    if step % self.train_every_n_steps:
      return
    learning_rate = math.nan
    if trainer.optimizers and trainer.optimizers[0].param_groups:
      learning_rate = float(trainer.optimizers[0].param_groups[0]["lr"])
    self._append(
        self.train_path,
        TRAIN_COLUMNS,
        {
            "optimizer_step": step,
            "epoch": int(trainer.current_epoch),
            "batch_idx": int(batch_idx),
            **averaged,
            "learning_rate": learning_rate,
            "wall_time_seconds": time.monotonic() - self._start_time,
        },
    )

  def on_validation_start(self, trainer, pl_module) -> None:
    del trainer, pl_module
    if not self.preserve_rng_around_validation:
      return
    self._rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

  def on_validation_end(self, trainer, pl_module) -> None:
    del pl_module
    if not trainer.sanity_checking and trainer.is_global_zero:
      metrics = trainer.callback_metrics
      row: Dict[str, Any] = {
          "optimizer_step": (self.validation_step_override
                             if self.validation_step_override is not None
                             else int(trainer.global_step)),
          "epoch": int(trainer.current_epoch),
          "wall_time_seconds": time.monotonic() - self._start_time,
      }
      for name in VALIDATION_METRICS:
        row[name.replace("/", "_")] = _as_float(metrics.get(name))
      self._append(self.validation_path, VALIDATION_COLUMNS, row)

    # Every DDP rank restores its own generator states.  Detaching validation
    # from the training RNG stream makes the monitored and unmonitored training
    # trajectories equivalent, apart from non-bitwise CUDA effects.
    if self._rng_state is not None:
      random.setstate(self._rng_state["python"])
      np.random.set_state(self._rng_state["numpy"])
      torch.set_rng_state(self._rng_state["torch_cpu"])
      if self._rng_state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(self._rng_state["torch_cuda"])
      self._rng_state = None
