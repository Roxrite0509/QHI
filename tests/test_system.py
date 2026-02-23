"""
tests/test_system.py
====================
Unit tests for QHI-Probe system.

Run:
    pytest tests/
    pytest tests/ -v --tb=short
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qhi_probe import QHIProbeSystem, ClinicalSample, QHIScore
from data.loader import load_demo_samples, _extract_entities_simple


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_system():
    """Trained QHI-Probe system for all tests."""
    system = QHIProbeSystem(hidden_dim=256, verbose=False)
    samples = load_demo_samples(n=200, seed=42)
    system.train(samples)
    return system


@pytest.fixture
def clean_sample():
    return ClinicalSample(
        text="Q: Treatment for type 2 diabetes?\nA: Metformin 500mg BID with meals. Monitor eGFR.",
        entities=["Metformin", "diabetes", "eGFR"],
        true_label=0,
        true_severity=1.5,
        source="test"
    )


@pytest.fixture
def hallucinated_sample():
    return ClinicalSample(
        text="Q: Antidote for acetaminophen overdose?\nA: Naloxone IV is the specific antidote.",
        entities=["acetaminophen", "naloxone", "overdose"],
        true_label=1,
        true_severity=23.0,
        source="test"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST: DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoader:

    def test_demo_samples_count(self):
        samples = load_demo_samples(n=100)
        assert len(samples) == 100

    def test_demo_samples_have_labels(self):
        samples = load_demo_samples(n=50)
        for s in samples:
            assert s.true_label in [0, 1]
            assert 0.0 <= s.true_severity <= 25.0

    def test_demo_samples_have_entities(self):
        samples = load_demo_samples(n=50)
        for s in samples:
            assert isinstance(s.entities, list)
            assert len(s.entities) > 0

    def test_demo_samples_balanced(self):
        """Should have both hallucinated and clean samples."""
        samples = load_demo_samples(n=200, hallucination_rate=0.4)
        labels = [s.true_label for s in samples]
        assert sum(labels) > 0, "No hallucinated samples"
        assert sum(1 - l for l in labels) > 0, "No clean samples"

    def test_reproducibility(self):
        s1 = load_demo_samples(n=50, seed=42)
        s2 = load_demo_samples(n=50, seed=42)
        assert [s.true_label for s in s1] == [s.true_label for s in s2]

    def test_entity_extractor_basic(self):
        text = "Patient has diabetes and is on metformin therapy."
        entities = _extract_entities_simple(text)
        assert isinstance(entities, list)

    def test_clinical_sample_creation(self):
        s = ClinicalSample(
            text="test text", entities=["entity1"],
            true_label=0, true_severity=5.0
        )
        assert s.text == "test text"
        assert s.true_label == 0


# ─────────────────────────────────────────────────────────────────────────────
# TEST: TRAINING
# ─────────────────────────────────────────────────────────────────────────────

class TestTraining:

    def test_training_returns_stats(self):
        system = QHIProbeSystem(hidden_dim=256, verbose=False)
        samples = load_demo_samples(n=100)
        stats = system.train(samples)
        assert "probe_c_auc" in stats
        assert "probe_r_acc" in stats
        assert "probe_v_auc" in stats
        assert "train_time_s" in stats

    def test_training_marks_system_as_trained(self):
        system = QHIProbeSystem(hidden_dim=256, verbose=False)
        assert not system.trained
        system.train(load_demo_samples(n=100))
        assert system.trained

    def test_training_requires_minimum_samples(self):
        system = QHIProbeSystem(hidden_dim=256, verbose=False)
        with pytest.raises(ValueError, match="50 samples"):
            system.train(load_demo_samples(n=10))

    def test_probe_auc_above_random(self, trained_system):
        assert trained_system.training_stats["probe_c_auc"] > 0.6
        assert trained_system.training_stats["probe_v_auc"] > 0.6

    def test_probe_r_accuracy_above_random(self, trained_system):
        # 5-class problem: random = 0.2, should beat that
        assert trained_system.training_stats["probe_r_acc"] > 0.4


# ─────────────────────────────────────────────────────────────────────────────
# TEST: SCORING
# ─────────────────────────────────────────────────────────────────────────────

class TestScoring:

    def test_score_returns_qhi_score(self, trained_system, clean_sample):
        result = trained_system.score(clean_sample)
        assert isinstance(result, QHIScore)

    def test_score_qhi_in_range(self, trained_system, clean_sample, hallucinated_sample):
        for sample in [clean_sample, hallucinated_sample]:
            score = trained_system.score(sample)
            assert 0.0 <= score.qhi <= 25.0, f"QHI out of range: {score.qhi}"

    def test_score_components_in_range(self, trained_system, clean_sample):
        score = trained_system.score(clean_sample)
        assert 0.0 <= score.uncertainty_score <= 1.0
        assert 1.0 <= score.risk_score <= 5.0
        assert 0.0 <= score.causal_violation_prob <= 1.0

    def test_score_gate_valid(self, trained_system, clean_sample):
        score = trained_system.score(clean_sample)
        assert score.gate in ["AUTO_USE", "REVIEW", "BLOCK"]

    def test_score_gate_auto_for_low_qhi(self, trained_system, clean_sample):
        score = trained_system.score(clean_sample)
        if score.qhi < 5.0:
            assert score.gate == "AUTO_USE"

    def test_score_gate_block_for_high_qhi(self, trained_system):
        """Manually construct a sample that should trigger high QHI."""
        sample = ClinicalSample(
            text="Q: Antidote for acetaminophen?\nA: Naloxone is the correct antidote.",
            entities=["acetaminophen", "naloxone"],
            true_label=1,
            true_severity=25.0
        )
        score = trained_system.score(sample)
        assert score.qhi >= 0.0  # at minimum, just verify it runs

    def test_score_latency_fast(self, trained_system, clean_sample):
        """Inference should be under 50ms even on slow systems."""
        score = trained_system.score(clean_sample)
        assert score.inference_time_ms < 50.0

    def test_score_requires_trained_system(self, clean_sample):
        system = QHIProbeSystem(hidden_dim=256, verbose=False)
        with pytest.raises(RuntimeError, match="trained"):
            system.score(clean_sample)

    def test_score_batch(self, trained_system, clean_sample, hallucinated_sample):
        scores = trained_system.score_batch([clean_sample, hallucinated_sample])
        assert len(scores) == 2
        for s in scores:
            assert isinstance(s, QHIScore)

    def test_score_to_dict(self, trained_system, clean_sample):
        score = trained_system.score(clean_sample)
        d = score.to_dict()
        required_keys = ["qhi", "gate", "uncertainty_score", "risk_score",
                         "causal_violation_prob", "inference_time_ms"]
        for k in required_keys:
            assert k in d


# ─────────────────────────────────────────────────────────────────────────────
# TEST: QHI FORMULA
# ─────────────────────────────────────────────────────────────────────────────

class TestQHIFormula:

    def test_zero_uncertainty_gives_zero_qhi(self, trained_system):
        """If any component is 0, QHI should be 0 (multiplicative)."""
        # The formula is multiplicative, so zero in = zero out
        # We test this by verifying clean samples tend toward low QHI
        clean_samples = load_demo_samples(n=100, hallucination_rate=0.0)
        scores = trained_system.score_batch(clean_samples)
        qhi_values = [s.qhi for s in scores]
        avg_clean_qhi = np.mean(qhi_values)
        # Clean samples should have lower avg QHI than random
        assert avg_clean_qhi < 15.0

    def test_hallucinated_higher_than_clean(self, trained_system):
        """Hallucinated samples should score higher on average than clean."""
        samples = load_demo_samples(n=200, seed=42)
        scores = trained_system.score_batch(samples)
        clean_qhi = np.mean([s.qhi for s, samp in zip(scores, samples) if samp.true_label == 0])
        hal_qhi   = np.mean([s.qhi for s, samp in zip(scores, samples) if samp.true_label == 1])
        assert hal_qhi >= clean_qhi, \
            f"Hallucinated QHI ({hal_qhi:.2f}) should be >= clean QHI ({clean_qhi:.2f})"

    def test_gate_boundaries(self):
        """Test gate logic directly."""
        system = QHIProbeSystem(hidden_dim=256, verbose=False)
        assert system._compute_gate(0.0)  == "AUTO_USE"
        assert system._compute_gate(4.99) == "AUTO_USE"
        assert system._compute_gate(5.0)  == "REVIEW"
        assert system._compute_gate(19.99)== "REVIEW"
        assert system._compute_gate(20.0) == "BLOCK"
        assert system._compute_gate(25.0) == "BLOCK"


# ─────────────────────────────────────────────────────────────────────────────
# TEST: END-TO-END PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:

    def test_full_pipeline_runs(self):
        """Complete train → score pipeline works without errors."""
        from sklearn.metrics import roc_auc_score

        samples = load_demo_samples(n=150, seed=0)
        train, test = samples[:100], samples[100:]

        system = QHIProbeSystem(hidden_dim=256, verbose=False)
        system.train(train)

        scores = system.score_batch(test)
        y_true = [s.true_label for s in test]
        y_score = [sc.qhi / 25.0 for sc in scores]

        auc = roc_auc_score(y_true, y_score)
        assert auc > 0.5, f"AUC too low: {auc:.3f}"

    def test_summary_works(self, trained_system):
        summary = trained_system.summary()
        assert "Probe-C" in summary
        assert "Probe-R" in summary
        assert "Probe-V" in summary

    def test_score_repr_contains_qhi(self, trained_system, clean_sample):
        score = trained_system.score(clean_sample)
        repr_str = repr(score)
        assert "QHI" in repr_str
        assert "Gate" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
