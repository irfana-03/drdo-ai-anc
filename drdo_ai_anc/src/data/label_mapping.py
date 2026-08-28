"""
label_mapping.py — Transparent mapping from dataset-specific labels to
acoustic context classes: STATIONARY, DYNAMIC, IMPULSIVE, SPEECH, OTHER.

Mappings are documented with reasons. No invented military/defence classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

CONTEXT_CLASSES = ["STATIONARY", "DYNAMIC", "IMPULSIVE", "SPEECH", "OTHER"]


@dataclass(frozen=True)
class LabelMappingEntry:
    dataset: str
    original_label: str
    mapped_context: str
    reason: str


# ---------------------------------------------------------------------------
# DEMAND — environment code is the label (Thiemann et al., ICA 2013)
# Nature/office backgrounds are quasi-stationary; transport/street/domestic
# activity environments exhibit stronger temporal variation.
# ---------------------------------------------------------------------------
DEMAND_ENVIRONMENT_MAP: Dict[str, tuple[str, str]] = {
    "NFIELD": (
        "STATIONARY",
        "Open field — steady ambient wind/grass noise (nature category).",
    ),
    "NRIVER": (
        "STATIONARY",
        "River — continuous flowing-water ambient noise (nature category).",
    ),
    "NPARK": (
        "STATIONARY",
        "Park — relatively steady outdoor ambient noise (nature category).",
    ),
    "OOFFICE": (
        "STATIONARY",
        "Office — low-variation HVAC/room tone (office category).",
    ),
    "OHALLWAY": (
        "STATIONARY",
        "Hallway — steady building ambient noise (office category).",
    ),
    "OMEETING": (
        "DYNAMIC",
        "Meeting room — intermittent speech/activity over room tone (office).",
    ),
    "DKITCHEN": (
        "DYNAMIC",
        "Kitchen — appliances and intermittent domestic activity (domestic).",
    ),
    "DLIVING": (
        "DYNAMIC",
        "Living room — TV, movement, intermittent domestic sounds (domestic).",
    ),
    "DWASHING": (
        "DYNAMIC",
        "Washing machine — periodic mechanical cycling (domestic).",
    ),
    "PCAFETER": (
        "DYNAMIC",
        "Cafeteria — crowd chatter and intermittent activity (public).",
    ),
    "PRESTO": (
        "DYNAMIC",
        "Restaurant — variable crowd and service activity (public).",
    ),
    "SCAFE": (
        "DYNAMIC",
        "Street cafe — traffic and pedestrian variability (public).",
    ),
    "PSTATION": (
        "DYNAMIC",
        "Public station — announcements, crowds, trains (public).",
    ),
    "SPSQUARE": (
        "DYNAMIC",
        "Public square — variable pedestrian and urban activity (street).",
    ),
    "STRAFFIC": (
        "DYNAMIC",
        "Street traffic — moving vehicles, non-stationary road noise (street).",
    ),
    "TBUS": (
        "DYNAMIC",
        "Bus interior — engine vibration and route-dependent changes (transport).",
    ),
    "TCAR": (
        "DYNAMIC",
        "Car interior — engine/road noise varying with speed (transport).",
    ),
    "TMETRO": (
        "DYNAMIC",
        "Metro — rail rumble, stops, and cabin dynamics (transport).",
    ),
}

# ---------------------------------------------------------------------------
# SONYC-UST — coarse urban sound classes (Cartwright et al., DCASE 2020)
# ---------------------------------------------------------------------------
SONYC_COARSE_MAP: Dict[str, tuple[str, str]] = {
    "engine": (
        "DYNAMIC",
        "Vehicle/engine sounds — time-varying mechanical noise.",
    ),
    "machinery-impact": (
        "IMPULSIVE",
        "Construction impact machinery — transient impulsive events.",
    ),
    "non-machinery-impact": (
        "IMPULSIVE",
        "Non-machinery impacts — short transient acoustic events.",
    ),
    "powered-saw": (
        "DYNAMIC",
        "Powered saws — sustained but modulated mechanical noise.",
    ),
    "alert-signal": (
        "IMPULSIVE",
        "Horns, sirens, alarms — short alert transients.",
    ),
    "music": (
        "OTHER",
        "Music — does not fit stationary/dynamic/impulsive/speech taxonomy.",
    ),
    "human-voice": (
        "SPEECH",
        "Human voice — speech-related acoustic context.",
    ),
    "dog": (
        "OTHER",
        "Dog barking — biological non-speech event, mapped to OTHER.",
    ),
}

# Priority when multiple SONYC coarse labels are present (highest first)
SONYC_LABEL_PRIORITY = ["SPEECH", "IMPULSIVE", "DYNAMIC", "OTHER", "STATIONARY"]

# ---------------------------------------------------------------------------
# CHiME-3 — noisy speech recordings
# ---------------------------------------------------------------------------
CHIME3_MAP: Dict[str, tuple[str, str]] = {
    "noisy_speech": (
        "SPEECH",
        "CHiME-3 contains real noisy speech in cafeteria/bus/street environments.",
    ),
    "BUS": ("SPEECH", "CHiME-3 bus environment — primary content is speech."),
    "CAF": ("SPEECH", "CHiME-3 cafe environment — primary content is speech."),
    "PED": ("SPEECH", "CHiME-3 pedestrian area — primary content is speech."),
    "STR": ("SPEECH", "CHiME-3 street junction — primary content is speech."),
}


def map_demand_label(environment_code: str) -> Optional[LabelMappingEntry]:
    """Map DEMAND environment code to acoustic context."""
    code = environment_code.upper().split("_")[0]
    if code not in DEMAND_ENVIRONMENT_MAP:
        return None
    mapped, reason = DEMAND_ENVIRONMENT_MAP[code]
    return LabelMappingEntry("demand", code, mapped, reason)


def map_sonyc_coarse_labels(
    present_coarse: List[str],
) -> Optional[LabelMappingEntry]:
    """Map SONYC coarse labels (multi-label) to a single context class."""
    if not present_coarse:
        return LabelMappingEntry(
            "sonyc_ust",
            "no_label",
            "OTHER",
            "No positive coarse label in annotation — mapped to OTHER.",
        )

    mapped_contexts: List[str] = []
    reasons: List[str] = []
    for coarse in present_coarse:
        if coarse in SONYC_COARSE_MAP:
            mapped, reason = SONYC_COARSE_MAP[coarse]
            mapped_contexts.append(mapped)
            reasons.append(f"{coarse}→{mapped}: {reason}")

    if not mapped_contexts:
        return LabelMappingEntry(
            "sonyc_ust",
            ",".join(present_coarse),
            "OTHER",
            "Unknown coarse label(s) — mapped to OTHER.",
        )

    # Resolve multi-label by priority
    for priority in SONYC_LABEL_PRIORITY:
        if priority in mapped_contexts:
            return LabelMappingEntry(
                "sonyc_ust",
                ",".join(present_coarse),
                priority,
                "; ".join(reasons),
            )

    return LabelMappingEntry(
        "sonyc_ust",
        ",".join(present_coarse),
        mapped_contexts[0],
        "; ".join(reasons),
    )


def map_chime3_label(label: str) -> LabelMappingEntry:
    """Map CHiME-3 environment or file type to acoustic context."""
    key = label.upper()
    if key in CHIME3_MAP:
        mapped, reason = CHIME3_MAP[key]
    else:
        mapped, reason = CHIME3_MAP["noisy_speech"]
    return LabelMappingEntry("chime3", label, mapped, reason)


def get_all_mapping_entries() -> List[LabelMappingEntry]:
    """Return every documented mapping for label_mapping.csv."""
    entries: List[LabelMappingEntry] = []
    for code, (mapped, reason) in DEMAND_ENVIRONMENT_MAP.items():
        entries.append(LabelMappingEntry("demand", code, mapped, reason))
    for coarse, (mapped, reason) in SONYC_COARSE_MAP.items():
        entries.append(LabelMappingEntry("sonyc_ust", coarse, mapped, reason))
    for label, (mapped, reason) in CHIME3_MAP.items():
        entries.append(LabelMappingEntry("chime3", label, mapped, reason))
    return entries
