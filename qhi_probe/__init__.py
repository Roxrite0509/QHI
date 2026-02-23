"""
QHI-Probe: Quantified Hallucination Index for Clinical LLMs
============================================================
Lightweight hallucination severity scoring via sparse entity-conditioned
probing of frozen LLM hidden states.

    QHI = Uncertainty × Risk × ViolationProb × 5    [0–25]

Quick start:
    from qhi_probe import QHIProbeSystem, ClinicalSample
    system = QHIProbeSystem()
    system.train(samples)
    score = system.score(sample)
    print(score)

GitHub: https://github.com/pranav-qhi-probe/qhi-probe
Paper:  QHI-Probe: Lightweight Hallucination Severity Scoring (2025)
"""

from qhi_probe.model import QHIProbeSystem, ClinicalSample, QHIScore

__version__ = "0.1.0"
__author__  = "Pranav"
__license__ = "MIT"

__all__ = [
    "QHIProbeSystem",
    "ClinicalSample",
    "QHIScore",
]
