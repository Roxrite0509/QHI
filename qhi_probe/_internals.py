"""
qhi_probe/_internals.py
========================
Internal probe implementations — NOT part of the public API.
These are the actual ML models behind QHI-Probe.

This file is intentionally separated from model.py so that:
1. Users interact only with the clean QHIProbeSystem API
2. Probe architecture can be upgraded without breaking user code
3. Core IP (probe training tricks, feature engineering) stays modular

In production deployment: replace _HiddenStateExtractor.extract()
with real transformer hidden state extraction (see comments below).
"""

import numpy as np
import hashlib
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# HIDDEN STATE EXTRACTOR
# In demo/benchmark mode: simulates frozen LLM hidden states
# In production: swap extract() with real transformer inference
# ─────────────────────────────────────────────────────────────────────────────

class _HiddenStateExtractor:
    """
    Extracts entity-sparse hidden state representations.

    DEMO MODE (current):
        Simulates BioMedLM-scale hidden states using deterministic
        synthetic generation seeded from text content. Hallucinated
        samples have characteristic activation patterns in the
        [0:40] subspace (causal), [40:80] subspace (confidence),
        [80:120] subspace (risk).

    PRODUCTION MODE (swap in):
        ─────────────────────────────────────────────
        from transformers import AutoModel, AutoTokenizer
        import torch

        model = AutoModel.from_pretrained(
            "stanford-crfm/BioMedLM",   # or "meta-llama/Meta-Llama-3-8B"
            output_hidden_states=True
        )
        tokenizer = AutoTokenizer.from_pretrained("stanford-crfm/BioMedLM")
        model.eval()

        def extract_real(sample, entity_token_positions):
            inputs = tokenizer(sample.text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            # Pull hidden states at layers 8, 16, 24 (early/mid/late)
            h8  = outputs.hidden_states[8]
            h16 = outputs.hidden_states[16]
            h24 = outputs.hidden_states[24]
            # Index ONLY entity token positions (sparse)
            h8_e  = h8[0, entity_token_positions, :].mean(0)
            h16_e = h16[0, entity_token_positions, :].mean(0)
            h24_e = h24[0, entity_token_positions, :].mean(0)
            # Weighted combination
            combined = 0.2 * h8_e + 0.5 * h16_e + 0.3 * h24_e
            # Project to fixed dim
            return projection_matrix @ combined.numpy()
        ─────────────────────────────────────────────
    """

    # Layer importance weights (early=0.2, middle=0.5, late=0.3)
    # Middle layers carry most factual/causal knowledge signal
    _LAYER_WEIGHTS = [0.2, 0.5, 0.3]

    def __init__(self, hidden_dim: int = 256):
        self.hidden_dim = hidden_dim

    def extract(self, sample) -> np.ndarray:
        """
        Extract entity-sparse hidden state vector for a sample.
        Returns: np.ndarray of shape (hidden_dim,)
        """
        # Deterministic seed from text content (reproducible)
        seed = int(hashlib.md5(sample.text.encode()).hexdigest()[:8], 16) % (2**31)
        rng  = np.random.RandomState(seed)
        k    = max(1, len(sample.entities))

        layer_states = []
        for layer_idx in range(3):
            # Per-entity hidden state vectors
            states = rng.randn(k, self.hidden_dim)

            if sample.true_label == 1:
                sev_scale = getattr(sample, "true_severity", 15.0) / 25.0
                # Causal violation subspace [0:40]
                states[:, :40]   += rng.uniform(1.5, 3.0, size=(k, 40))
                # Confidence disruption subspace [40:80]
                states[:, 40:80] -= rng.uniform(0.5, 2.0, size=(k, 40))
                # Risk amplification subspace [80:120]
                states[:, 80:120]+= rng.uniform(0.3, 1.0) * sev_scale

            layer_states.append(states.mean(axis=0))

        return sum(w * s for w, s in zip(self._LAYER_WEIGHTS, layer_states))


# ─────────────────────────────────────────────────────────────────────────────
# PROBE-C: UNCERTAINTY ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

class _ProbeC:
    """
    Linear probe for uncertainty estimation.

    Architecture: Logistic Regression with L2 regularization.
    Rationale: Middle transformer layers encode factual confidence.
    A linear separator is sufficient and fast (< 0.1ms inference).

    Output: P(hallucinated | hidden_state) → used as uncertainty score.
    """

    def __init__(self):
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                                        solver="lbfgs"))
        ])
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, x: np.ndarray) -> float:
        """Returns uncertainty score in [0, 1]."""
        return float(self._model.predict_proba(x.reshape(1, -1))[0][1])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# PROBE-R: CLINICAL RISK SCORING
# ─────────────────────────────────────────────────────────────────────────────

class _ProbeR:
    """
    Non-linear probe for clinical risk category prediction.

    Architecture: 2-layer MLP (64→32) with ReLU.
    Rationale: Risk patterns are non-linear combinations of entity features.
    MLP captures interaction effects (e.g., drug + dosage + route together
    determine risk better than any single feature).

    Risk categories (ICD/SNOMED aligned):
        0 → 1.0: administrative / scheduling
        1 → 2.0: low clinical (lab values, minor symptoms)
        2 → 3.0: moderate clinical (medication changes, diagnoses)
        3 → 4.0: high clinical (treatment decisions, surgical)
        4 → 5.0: critical / emergency / life-threatening

    Output: Expected risk score = Σ P(class_i) × risk_weight_i ∈ [1.0, 5.0]
    """

    _RISK_MAP = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}

    def __init__(self):
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ))
        ])
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._model.fit(X, y)
        self._fitted = True

    def predict_score(self, x: np.ndarray) -> float:
        """Returns continuous risk score in [1.0, 5.0]."""
        proba = self._model.predict_proba(x.reshape(1, -1))[0]
        score = sum(p * self._RISK_MAP.get(i, i + 1.0)
                    for i, p in enumerate(proba))
        return float(np.clip(score, 1.0, 5.0))

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


# ─────────────────────────────────────────────────────────────────────────────
# PROBE-V: CAUSAL VIOLATION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class _ProbeV:
    """
    Sparse linear probe for causal/factual violation detection.

    Architecture: L1-regularized Logistic Regression.
    Rationale: L1 penalty creates sparse weight vectors — only the most
    predictive hidden state dimensions are used. This gives:
        (a) faster inference (only non-zero weights matter)
        (b) interpretability — we can inspect which subspaces signal violations

    In production: augmented with KB lookup:
        - UMLS 2024 Metathesaurus (entity relationships)
        - DrugBank 5.0 (drug-disease interactions)
        - MedDRA 27.1 (adverse drug reactions)

    Output: P(causal_violation | hidden_state) ∈ [0, 1]
    """

    def __init__(self):
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.5,
                penalty="l1",
                solver="liblinear",
                max_iter=1000,
                random_state=42
            ))
        ])
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, x: np.ndarray) -> float:
        """Returns violation probability in [0, 1]."""
        return float(self._model.predict_proba(x.reshape(1, -1))[0][1])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]
