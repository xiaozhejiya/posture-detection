from __future__ import annotations

from typing import Any, Dict

from .models import AlertEvent, PostureAnalysis


class AlertSystem:
    def __init__(self, config: Dict[str, Any]):
        self.cooldown_sec = float(config.get("cooldown_sec", 10.0))
        self.last_alert_time = -1e9

    def update(self, analysis: PostureAnalysis) -> AlertEvent:
        if analysis.status in {"normal", "invalid"}:
            return AlertEvent(False, analysis.status, "", analysis.timestamp)

        if analysis.timestamp - self.last_alert_time < self.cooldown_sec:
            return AlertEvent(False, analysis.status, analysis.message, analysis.timestamp, cooling_down=True)

        self.last_alert_time = analysis.timestamp
        return AlertEvent(True, analysis.status, analysis.message, analysis.timestamp)

