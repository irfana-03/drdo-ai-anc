"""
controller.py — Context-aware adaptive processing strategy selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ProcessingStrategy:
    name: str
    description: str
    use_deepfilternet: bool
    use_lms: bool
    transient_attenuation: bool
    speech_priority: bool
    suppression_strength: float  # 0.0–1.0


STRATEGIES: Dict[str, ProcessingStrategy] = {
    "STATIONARY": ProcessingStrategy(
        name="Stationary Suppression",
        description="DeepFilterNet + Residual LMS",
        use_deepfilternet=True,
        use_lms=True,
        transient_attenuation=False,
        speech_priority=False,
        suppression_strength=0.85,
    ),
    "DYNAMIC": ProcessingStrategy(
        name="Dynamic Tracking",
        description="DeepFilterNet + Normal suppression",
        use_deepfilternet=True,
        use_lms=False,
        transient_attenuation=False,
        speech_priority=False,
        suppression_strength=0.70,
    ),
    "IMPULSIVE": ProcessingStrategy(
        name="Impulsive Recovery",
        description="Transient attenuation + DeepFilterNet recovery",
        use_deepfilternet=True,
        use_lms=False,
        transient_attenuation=True,
        speech_priority=False,
        suppression_strength=0.90,
    ),
    "SPEECH": ProcessingStrategy(
        name="Speech Priority",
        description="Speech-priority enhancement",
        use_deepfilternet=True,
        use_lms=False,
        transient_attenuation=False,
        speech_priority=True,
        suppression_strength=0.45,
    ),
    "OTHER": ProcessingStrategy(
        name="General Enhancement",
        description="DeepFilterNet baseline",
        use_deepfilternet=True,
        use_lms=False,
        transient_attenuation=False,
        speech_priority=False,
        suppression_strength=0.60,
    ),
}


class AdaptiveController:
    """Select processing strategy from detected acoustic context."""

    def get_strategy(self, context: str) -> ProcessingStrategy:
        return STRATEGIES.get(context.upper(), STRATEGIES["OTHER"])

    def pipeline_stages(self, context: str, engine_status: Dict[str, str]) -> List[Dict[str, str]]:
        """Return pipeline stage statuses for UI display."""
        strategy = self.get_strategy(context)
        stages = [
            {"name": "MICROPHONE", "status": engine_status.get("audio", "READY")},
            {"name": "FEATURE EXTRACTION", "status": "PROCESSING" if engine_status.get("running") else "READY"},
            {"name": "AI CONTEXT ENGINE", "status": "PROCESSING" if context else "READY"},
            {"name": "IMPULSE DETECTOR", "status": "PROCESSING" if engine_status.get("running") else "READY"},
            {"name": "ADAPTIVE CONTROLLER", "status": "PROCESSING" if context else "READY"},
            {
                "name": "DEEPFILTERNET",
                "status": engine_status.get("deepfilternet", "NOT FOUND"),
            },
            {
                "name": "LMS RESIDUAL FILTER",
                "status": "PROCESSING" if strategy.use_lms and engine_status.get("running") else (
                    "READY" if strategy.use_lms else "READY"
                ),
            },
            {
                "name": "SPEECH PRIORITY",
                "status": "PROCESSING" if strategy.speech_priority and engine_status.get("running") else "READY",
            },
            {"name": "OUTPUT", "status": engine_status.get("audio", "READY")},
        ]
        if engine_status.get("deepfilternet") == "NOT FOUND":
            stages[5]["status"] = "WARNING"
        return stages
