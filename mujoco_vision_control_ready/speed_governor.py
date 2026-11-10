"""Observation-only, bounded speed-cap policy. Settings fixed before comparisons."""

MODES = ['baseline', 'motion_only', 'vision_motion']
SETTINGS = {
    'visual_caution_seconds': 1.0,
    'physical_hold_seconds': 1.0,
    'visual_speed_cap': .55,
    'motion_speed_cap': .40,
    'slip_speed_cap': .25,
    'cap_fall_rate': .60,
    'cap_rise_rate': .30,
    'entry_speed_margin': .05,
}


class SpeedGovernor:
    """No terrain, physical parameters, case name or disturbance schedule input."""

    def __init__(self, mode, max_speed, settings=None):
        if mode not in MODES:
            raise ValueError(mode)
        self.mode, self.maximum = mode, max_speed
        self.settings = dict(SETTINGS if settings is None else settings)
        self.cap = float(max_speed)
        self.time = 0.
        self.visual_seen = False
        self.visual_until = self.motion_until = self.slip_until = -1.
        self.previous_risk = False

    def update(self, dt, observed_speed, visual_change, motion_anomaly, slip_evidence):
        cfg = self.settings
        if self.mode == 'baseline':
            self.time += dt
            return {'cap': self.maximum, 'target': self.maximum, 'reason': 'baseline'}
        if self.mode == 'vision_motion' and visual_change and not self.visual_seen:
            self.visual_seen = True
            self.visual_until = self.time + cfg['visual_caution_seconds']
        if motion_anomaly:
            self.motion_until = self.time + cfg['physical_hold_seconds']
        if slip_evidence:
            self.slip_until = self.time + cfg['physical_hold_seconds']
        target, reason = self.maximum, 'normal'
        if self.time < self.visual_until:
            target, reason = cfg['visual_speed_cap'], 'temporary_visual_caution'
        if self.time < self.motion_until:
            target, reason = min(target, cfg['motion_speed_cap']), 'motion_anomaly'
        if self.time < self.slip_until:
            target, reason = min(target, cfg['slip_speed_cap']), 'slip_evidence'
        risk = target < self.maximum
        if risk and not self.previous_risk:
            # 去掉未使用的上限余量；不直接跳到低于观测速度的上限。
            self.cap = min(self.cap, max(0., observed_speed) + cfg['entry_speed_margin'])
        rate = cfg['cap_fall_rate'] if target < self.cap else cfg['cap_rise_rate']
        delta = max(-rate * dt, min(rate * dt, target - self.cap))
        self.cap = max(0., min(self.maximum, self.cap + delta))
        self.previous_risk = risk
        self.time += dt
        return {'cap': self.cap, 'target': target, 'reason': reason}
