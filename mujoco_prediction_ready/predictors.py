"""Streaming effective-gain predictors; RGB can schedule adaptation, not label mu."""
from collections import Counter
import numpy as np
from control_core import Physics, WheelEstimator, integrate

MODES = ['fixed', 'history_long', 'history_adaptive', 'vision_history']


class OnlinePredictor:
    def __init__(self, mode, config):
        if mode not in MODES:
            raise ValueError(mode)
        self.mode, self.config = mode, config
        self.physics, self.dt = Physics(**config['physics']), config['dt']
        estimator_mode = {'fixed': 'fixed', 'history_long': 'window_160',
                          'history_adaptive': 'adaptive', 'vision_history': 'adaptive'}[mode]
        self.estimator = WheelEstimator(estimator_mode, self.physics, self.dt, config['estimator'],
                                        config['measurement_noise']['std'][3:5])
        self.initial_labels, self.baseline = [], None
        self.changed_count, self.visual_fired = 0, False

    def observe_image(self, rgb_prediction):
        if self.mode != 'vision_history' or rgb_prediction is None or self.visual_fired:
            return False
        label = rgb_prediction['label']
        if self.baseline is None:
            if label != 'unknown':
                self.initial_labels.append(label)
            if len(self.initial_labels) >= 8:
                self.baseline = Counter(self.initial_labels).most_common(1)[0][0]
            return False
        changed = label != 'unknown' and label != self.baseline
        self.changed_count = self.changed_count + 1 if changed else 0
        if self.changed_count < 3:
            return False
        self.visual_fired = True
        estimator = self.estimator
        # 不清空历史，不根据外观写入增益；仅用已完成的短窗转移重新估计。
        if len(estimator.commands) >= estimator.minimum:
            commands, forces = np.asarray(estimator.commands), np.asarray(estimator.forces)
            estimator.raw, _ = estimator.ls(commands, forces, estimator.minimum)
            estimator.last_alarm = estimator.updates
            estimator.window = estimator.minimum
            estimator.consecutive = 0
        return True

    def predict(self, observation, planned_commands):
        """Freeze the current gain at this origin; consume no future observation/image."""
        gain = self.estimator.used().copy()
        state = np.asarray(observation).copy()
        for command in planned_commands:
            state = integrate(state, gain * command, self.physics, self.dt)
        return state[3:5]

    def update(self, observation, command, next_observation):
        self.estimator.update(observation[3:5], command, next_observation[3:5])
