"""Dynamic learning-rate scheduler (Sec. IV-B).

"To avoid catastrophic unlearning, we adopt a dynamic learning rate scheduler
for training ... At the end of the i-th epoch, the learning rate is
discounted with factor beta_eta only if the agent performance is above a
certain threshold"::

    eta_{i+1} = max(beta_eta * eta_i, eta_min)   if success rate > 0.99
                eta_i                             otherwise

with ``eta_0 = 3.5e-4``, ``eta_min = 1e-6`` and ``beta_eta = 0.7``.

Within an epoch the reference implementation scales the base rate by
``sqrt(remaining fraction of the epoch)`` (``LearningRateSchedule`` in
``meda_utils.py``, applied per ``PPO2.learn`` call); ``intra_epoch="constant"``
keeps it flat instead.

Stable-Baselines3 evaluates ``lr_schedule(progress_remaining)`` before every
update, where ``progress_remaining`` refers to the current ``learn()`` call.
:class:`DynamicLearningRate` converts that into the progress of the current
epoch, so a single model can be trained epoch by epoch without resetting its
timestep counter.  SB3 updates the progress *after* collecting a rollout,
whereas PPO2 computed it before (``frac = 1 - (update - 1) / n_updates``);
passing the rollout size to :meth:`start_epoch` reproduces PPO2 exactly: the
first update of an epoch uses the full base rate and the last one
``sqrt(1 / n_updates)`` of it (instead of 0).
"""

from __future__ import annotations

import math


class DynamicLearningRate:
    def __init__(
        self,
        lr0: float = 3.5e-4,
        lr_min: float = 1.0e-6,
        decay: float = 0.7,
        success_threshold: float = 0.99,
        intra_epoch: str = "sqrt",
    ) -> None:
        if intra_epoch not in ("sqrt", "constant"):
            raise ValueError(f"unknown intra_epoch mode {intra_epoch!r}")
        self.base_rate = float(lr0)
        self.lr_min = float(lr_min)
        self.decay = float(decay)
        self.success_threshold = float(success_threshold)
        self.intra_epoch = intra_epoch
        # Bookkeeping for mapping SB3 progress onto epoch progress.
        self._epoch_start = 0
        self._epoch_steps = 1
        self._rollout = 0
        self._learn_total = 1

    # ----------------------------------------------------------- epochs
    def start_epoch(self, num_timesteps: int, epoch_steps: int, rollout_steps: int = 0) -> None:
        """Call before ``model.learn(epoch_steps, reset_num_timesteps=False)``.

        ``rollout_steps`` (``n_envs * n_steps``) is the number of steps SB3
        has already added to its counter when it evaluates the schedule.
        """
        self._epoch_start = int(num_timesteps)
        self._epoch_steps = max(int(epoch_steps), 1)
        self._rollout = max(int(rollout_steps), 0)
        self._learn_total = int(num_timesteps) + self._epoch_steps

    def end_epoch(self, success_rate: float) -> bool:
        """Apply the paper's update rule; returns True if the rate was decayed."""
        if success_rate > self.success_threshold:
            self.base_rate = max(self.decay * self.base_rate, self.lr_min)
            return True
        return False

    # ----------------------------------------------------- SB3 interface
    def epoch_fraction_remaining(self, progress_remaining: float) -> float:
        # SB3: progress_remaining = 1 - num_timesteps / total, total = start + steps
        num_timesteps = (1.0 - float(progress_remaining)) * self._learn_total
        done = (num_timesteps - self._rollout - self._epoch_start) / self._epoch_steps
        return min(max(1.0 - done, 0.0), 1.0)

    def __call__(self, progress_remaining: float) -> float:
        if self.intra_epoch == "constant":
            return self.base_rate
        return self.base_rate * math.sqrt(self.epoch_fraction_remaining(progress_remaining))
