"""Instrumental-variable variant of the windowed wheel-efficiency estimator.

``WheelEstimator`` regresses the force implied by two consecutive velocity
measurements on the command of the same step.  Under feedback, the command at
step t is computed from the measurement at step t, and the implied force is built
from the difference between the measurements at t+1 and t.  Both therefore carry
the measurement noise of step t, and ordinary least squares no longer averages
that noise away: the efficiency estimate is biased, more so as the noise grows.

The previous command u[t-1] was fixed before the measurement at step t existed,
yet it stays close to u[t] because commands change slowly.  Using it as the
instrument removes the shared noise term:

    eta = sum(u[t-1] * F[t]) / sum(u[t-1] * u[t])

Assumptions: measurement noise is independent from one control step to the next,
and consecutive commands are correlated.  Everything else in the estimator is
inherited unchanged, including the heuristic alarm threshold of the adaptive
window.
"""

from contextlib import contextmanager

import numpy as np


# 相邻两步指令的相关系数低于此值时，认为工具变量太弱，保持上一次的估计。
MIN_INSTRUMENT_CORRELATION = 0.5


def instrumental_variable_estimator(base_class):
    """Return a subclass of ``base_class`` (any copy of WheelEstimator) that uses u[t-1] as instrument."""

    class InstrumentalVariableEstimator(base_class):
        def ls(self, commands, forces, size):
            command, force = commands[-size:], forces[-size:]
            energy = np.sum(command ** 2, axis=0)
            excited = energy >= size * self.settings["min_command_rms"] ** 2
            instrument, regressor, target = command[:-1], command[1:], force[1:]
            cross = np.sum(instrument * regressor, axis=0)
            scale = np.sqrt(np.sum(instrument ** 2, axis=0) * np.sum(regressor ** 2, axis=0))
            # 与原实现的小输入保护一致：条件不满足的一侧不更新，并计入 skipped_side_updates。
            usable = excited & (cross >= MIN_INSTRUMENT_CORRELATION * scale) & (scale > 0)
            estimate = self.raw.copy()
            estimate[usable] = np.sum(instrument * target, axis=0)[usable] / cross[usable]
            return estimate, usable

    InstrumentalVariableEstimator.__name__ = f"InstrumentalVariable{base_class.__name__}"
    return InstrumentalVariableEstimator


@contextmanager
def estimator_swapped(module, estimator_class):
    """Make ``module.simulate`` build ``estimator_class`` for the duration of the block."""
    original = module.WheelEstimator
    module.WheelEstimator = estimator_class
    try:
        yield
    finally:
        module.WheelEstimator = original
