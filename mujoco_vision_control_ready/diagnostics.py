"""Causal evidence fusion; no access to terrain IDs, truth, friction or fault labels."""

from collections import Counter, deque
import numpy as np


class EvidenceMonitor:
    def __init__(self, wheel_radius=.1, half_track=.2):
        self.radius, self.track = wheel_radius, half_track
        self.residuals = deque(maxlen=25)
        self.slips = deque(maxlen=20)
        self.initial_labels = []
        self.baseline_label = None
        self.changed_frames = 0
        self.visual_change = False
        self.motion_count = 0
        self.motion_anomaly = False
        self.slip_evidence = False
        self.motion_score = 0.
        self.slip_score = 0.

    def observe_motion(self, current_observation, next_observation, nominal_prediction, wheel_rates_observed):
        # innovation使用下一步带噪观测，不使用事后评分用的真实下一状态。
        self.residuals.append(np.asarray(next_observation)[3:5] - nominal_prediction)
        expected_wheel_speed = next_observation[3] + np.array([-1, 1]) * self.track * next_observation[4]
        self.slips.append(self.radius * np.asarray(wheel_rates_observed) - expected_wheel_speed)
        if len(self.residuals) == self.residuals.maxlen:
            self.motion_score = float(np.max(np.abs(np.mean(self.residuals, axis=0)) / [.002, .006]))
            self.motion_count = self.motion_count + 1 if self.motion_score > 1 else 0
            self.motion_anomaly = self.motion_count >= 5
        if len(self.slips) == self.slips.maxlen:
            self.slip_score = float(np.max(np.mean(np.abs(self.slips), axis=0)) / .2)
            self.slip_evidence = self.slip_score > 1

    def observe_image(self, prediction):
        label = prediction["label"]
        if self.baseline_label is None:
            if label != "unknown":
                self.initial_labels.append(label)
            if len(self.initial_labels) >= 8:
                self.baseline_label = Counter(self.initial_labels).most_common(1)[0][0]
        else:
            changed = label != "unknown" and label != self.baseline_label
            self.changed_frames = self.changed_frames + 1 if changed else 0
            if self.changed_frames >= 3:
                self.visual_change = True
        if self.slip_evidence:
            status = "visual_change_and_slip_evidence" if self.visual_change else "slip_evidence_without_visual_change"
        elif self.motion_anomaly:
            status = "visual_change_with_motion_anomaly" if self.visual_change else "motion_anomaly_cause_unknown"
        elif self.visual_change:
            status = "appearance_changed_physical_effect_unconfirmed"
        else:
            status = "no_current_evidence"
        return {"motion_score": self.motion_score, "slip_score": self.slip_score,
                "visual_change": self.visual_change, "motion_anomaly": self.motion_anomaly,
                "slip_evidence": self.slip_evidence, "status": status}
