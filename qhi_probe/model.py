"""
qhi_probe/model.py
==================
QHI-Probe: Quantified Hallucination Index
Public API — internals are abstracted behind QHIProbeSystem.

Usage:
    from qhi_probe import QHIProbeSystem, ClinicalSample
    system = QHIProbeSystem()
    system.train(samples)
    score = system.score(sample)

Formula:
    QHI = Uncertainty × Risk × ViolationProb × 5    [Range: 0–25]

Gates (ISO 14971 aligned):
    QHI < 5    → AUTO_USE
    5 ≤ QHI < 20 → REVIEW
    QHI ≥ 20  → BLOCK
"""

import numpy as np
import time
import hashlib
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional

warnings.filterwarnings("ignore")

# ── Public Data Structures ────────────────────────────────────────────────────

@dataclass
class ClinicalSample:
    """
    One clinical LLM output to be scored by QHI-Probe.

    Args:
        text        : The LLM-generated clinical text to evaluate
        entities    : List of medical entities detected in the text
                      (drug names, diagnoses, procedures, lab values)
        true_label  : Ground truth — 0 = clean, 1 = hallucinated
        true_severity: Ground truth QHI severity (0.0–25.0); use 0.0 if unknown
        source      : Dataset source tag (e.g. 'medqa', 'medmcqa')
    """
    text: str
    entities: List[str]
    true_label: int
    true_severity: float = 0.0
    source: str = "unknown"


@dataclass
class QHIScore:
    """
    Full QHI scoring result for one clinical sample.

    Attributes:
        uncertainty_score     : Probe-C output — model uncertainty (0–1)
        risk_score            : Probe-R output — clinical risk level (1–5)
        causal_violation_prob : Probe-V output — violation probability (0–1)
        qhi                   : Final QHI score (0–25)
        gate                  : Operational gate: AUTO_USE / REVIEW / BLOCK
        entity_count          : Number of medical entities found
        inference_time_ms     : Probe inference time (milliseconds)
        explanation           : Human-readable explanation of the score
    """
    uncertainty_score: float
    risk_score: float
    causal_violation_prob: float
    qhi: float
    gate: str
    entity_count: int
    inference_time_ms: float
    explanation: str = ""

    def to_dict(self) -> Dict:
        return {
            "uncertainty_score":      round(self.uncertainty_score, 4),
            "risk_score":             round(self.risk_score, 4),
            "causal_violation_prob":  round(self.causal_violation_prob, 4),
            "qhi":                    round(self.qhi, 4),
            "gate":                   self.gate,
            "entity_count":           self.entity_count,
            "inference_time_ms":      round(self.inference_time_ms, 3),
            "explanation":            self.explanation
        }

    def __repr__(self):
        bar = "█" * int(self.qhi) + "░" * (25 - int(self.qhi))
        return (
            f"\n{'='*52}\n"
            f"  QHI Score: {self.qhi:.2f}/25  [{bar}]\n"
            f"  Gate:      {self.gate}\n"
            f"  ├─ Uncertainty:  {self.uncertainty_score:.4f}\n"
            f"  ├─ Risk:         {self.risk_score:.4f}\n"
            f"  └─ Violation:    {self.causal_violation_prob:.4f}\n"
            f"  {self.explanation}\n"
            f"{'='*52}"
        )


# ── Internal Modules (abstracted) ─────────────────────────────────────────────
# The actual probe implementations are in _internals.py (not public)
# This file exposes only the clean API

from qhi_probe._internals import (
    _HiddenStateExtractor,
    _ProbeC,
    _ProbeR,
    _ProbeV,
)


# ── Main System ───────────────────────────────────────────────────────────────

class QHIProbeSystem:
    """
    QHI-Probe: Lightweight hallucination severity scoring for clinical LLMs.

    No secondary LLM required. Runs on CPU. <1ms inference.

    Example:
        >>> from qhi_probe import QHIProbeSystem, ClinicalSample
        >>> system = QHIProbeSystem()
        >>> system.train(train_samples)
        >>> score = system.score(sample)
        >>> print(score)
    """

    GATES = {"AUTO_USE": 5.0, "BLOCK": 20.0}

    GATE_DESCRIPTIONS = {
        "AUTO_USE": "Low hallucination risk — safe to use without manual review.",
        "REVIEW":   "Moderate risk — clinician verification recommended before use.",
        "BLOCK":    "High hallucination risk — output blocked, escalate to human expert.",
    }

    def __init__(self, hidden_dim: int = 256, verbose: bool = True):
        """
        Args:
            hidden_dim : Dimensionality of hidden state projections.
                         Use 256 for synthetic/demo mode.
                         Use 768 for BERT-base, 1024 for BioMedLM.
            verbose    : Print training progress.
        """
        self.hidden_dim = hidden_dim
        self.verbose = verbose
        self._extractor = _HiddenStateExtractor(hidden_dim=hidden_dim)
        self._probe_c = _ProbeC()
        self._probe_r = _ProbeR()
        self._probe_v = _ProbeV()
        self.trained = False
        self.training_stats: Dict = {}
        self._n_train_samples = 0

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, samples: List[ClinicalSample]) -> Dict:
        """
        Train all three probes on labeled clinical samples.

        Args:
            samples : List of ClinicalSample with true_label and true_severity set.

        Returns:
            Dict with probe-level AUC/accuracy metrics and training time.

        Note:
            Minimum recommended: 200+ samples for stable probe training.
            For production: 1000+ samples with balanced hallucination labels.
        """
        from sklearn.model_selection import train_test_split

        if len(samples) < 50:
            raise ValueError(
                f"Need at least 50 samples to train QHI-Probe. Got {len(samples)}. "
                "Use data/loader.py to load MedQA or MedMCQA datasets."
            )

        if self.verbose:
            print(f"\n{'='*58}")
            print(f"  QHI-PROBE TRAINING")
            print(f"  Samples: {len(samples)} | Hidden dim: {self.hidden_dim}")
            print(f"{'='*58}")

        t0 = time.time()

        # Extract hidden state representations for all samples
        X  = np.array([self._extractor.extract(s) for s in samples])
        yh = np.array([s.true_label for s in samples])        # hallucination labels
        ys = np.array([s.true_severity for s in samples])     # severity values

        # Derive risk labels (5 bins over severity range 0–25)
        yr = np.clip((ys / 5.0).astype(int), 0, 4)
        # Violation labels: critical severity (>15) = definite causal violation
        yv = (ys > 15.0).astype(int)

        # Stratified split for evaluation
        idx = np.arange(len(samples))
        tr, te = train_test_split(idx, test_size=0.2, random_state=42,
                                  stratify=yh)

        # Train probes
        from sklearn.metrics import roc_auc_score

        self._probe_c.fit(X[tr], yh[tr])
        c_auc = roc_auc_score(yh[te],
                              self._probe_c.predict_proba_batch(X[te]))

        self._probe_r.fit(X[tr], yr[tr])
        r_acc = (self._probe_r.predict_batch(X[te]) == yr[te]).mean()

        if len(np.unique(yv[tr])) > 1:
            self._probe_v.fit(X[tr], yv[tr])
            _vte = yv[te]
        else:
            self._probe_v.fit(X[tr], yh[tr])
            _vte = yh[te]
        v_auc = roc_auc_score(_vte,
                               self._probe_v.predict_proba_batch(X[te]))

        t_total = time.time() - t0
        self.trained = True
        self._n_train_samples = len(samples)

        self.training_stats = {
            "probe_c_auc": round(c_auc, 4),
            "probe_r_acc": round(r_acc, 4),
            "probe_v_auc": round(v_auc, 4),
            "train_time_s": round(t_total, 3),
            "n_samples": len(samples),
            "n_hallucinated": int(yh.sum()),
            "n_clean": int((1 - yh).sum()),
        }

        if self.verbose:
            print(f"  Probe-C (Uncertainty)  AUC : {c_auc:.4f}")
            print(f"  Probe-R (Risk)         Acc : {r_acc:.4f}")
            print(f"  Probe-V (Violation)    AUC : {v_auc:.4f}")
            print(f"  Training time               : {t_total:.3f}s")
            print(f"{'='*58}\n")

        return self.training_stats

    # ── Inference ─────────────────────────────────────────────────────────────

    def score(self, sample: ClinicalSample) -> QHIScore:
        """
        Score a single clinical sample.

        Args:
            sample : ClinicalSample to evaluate.

        Returns:
            QHIScore with QHI value, gate, and component scores.

        Raises:
            RuntimeError if system has not been trained yet.
        """
        if not self.trained:
            raise RuntimeError(
                "QHIProbeSystem must be trained before scoring. "
                "Call .train(samples) first."
            )

        t0 = time.time()
        hs = self._extractor.extract(sample)

        uncertainty = self._probe_c.predict(hs)
        risk        = self._probe_r.predict_score(hs)
        violation   = self._probe_v.predict(hs)

        qhi = float(np.clip(uncertainty * risk * violation * 5.0, 0.0, 25.0))
        gate = self._compute_gate(qhi)
        explanation = self._explain(uncertainty, risk, violation, gate)

        return QHIScore(
            uncertainty_score=uncertainty,
            risk_score=risk,
            causal_violation_prob=violation,
            qhi=qhi,
            gate=gate,
            entity_count=len(sample.entities),
            inference_time_ms=(time.time() - t0) * 1000,
            explanation=explanation
        )

    def score_batch(self, samples: List[ClinicalSample]) -> List[QHIScore]:
        """Score multiple samples. Returns list of QHIScore objects."""
        return [self.score(s) for s in samples]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _compute_gate(self, qhi: float) -> str:
        if qhi < self.GATES["AUTO_USE"]:
            return "AUTO_USE"
        elif qhi < self.GATES["BLOCK"]:
            return "REVIEW"
        return "BLOCK"

    def _explain(self, u: float, r: float, v: float, gate: str) -> str:
        parts = []
        if u > 0.7:
            parts.append("model is uncertain about this output")
        if r > 3.5:
            parts.append("high clinical risk domain")
        if v > 0.6:
            parts.append("likely causal/factual violation detected")
        if not parts:
            parts.append("all signals within safe range")
        return f"[{gate}] {'; '.join(parts).capitalize()}."

    def summary(self) -> str:
        """Print a summary of the trained system."""
        if not self.trained:
            return "System not trained yet."
        return (
            f"QHI-Probe System | Trained on {self._n_train_samples} samples\n"
            f"  Probe-C AUC: {self.training_stats['probe_c_auc']}\n"
            f"  Probe-R Acc: {self.training_stats['probe_r_acc']}\n"
            f"  Probe-V AUC: {self.training_stats['probe_v_auc']}\n"
        )
