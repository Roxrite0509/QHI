"""
run_benchmark.py
================
QHI-Probe Benchmark Runner

Usage:
    # Demo mode (no internet needed):
    python run_benchmark.py --dataset demo --n 500

    # Real MedQA-USMLE (requires: pip install datasets):
    python run_benchmark.py --dataset medqa --n 500

    # Real MedMCQA (194k questions):
    python run_benchmark.py --dataset medmcqa --n 1000

    # Combined datasets:
    python run_benchmark.py --dataset medqa medmcqa --n 300

    # Full evaluation with plots:
    python run_benchmark.py --dataset demo --n 1000 --save-results
"""

import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qhi_probe import QHIProbeSystem, ClinicalSample
from data.loader import load_demo_samples, load_combined

try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        classification_report, confusion_matrix
    )
    from sklearn.model_selection import train_test_split
except ImportError:
    print("ERROR: pip install scikit-learn")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def run_random_baseline(test_samples, seed=99):
    rng    = np.random.RandomState(seed)
    y_true = np.array([s.true_label for s in test_samples])
    y_pred = rng.random(len(y_true))
    return y_true, y_pred


def run_confidence_only_baseline(train_samples, test_samples):
    """Single logistic regression on hidden states — no risk/violation decomposition."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from qhi_probe._internals import _HiddenStateExtractor

    extractor = _HiddenStateExtractor(hidden_dim=256)
    X_train = np.array([extractor.extract(s) for s in train_samples])
    X_test  = np.array([extractor.extract(s) for s in test_samples])
    y_train = np.array([s.true_label for s in train_samples])
    y_test  = np.array([s.true_label for s in test_samples])

    model = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    return y_test, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(system: QHIProbeSystem, test_samples, threshold=0.5):
    """Full evaluation of QHI-Probe on test set."""
    scores  = system.score_batch(test_samples)
    y_true  = np.array([s.true_label for s in test_samples])
    y_score = np.array([sc.qhi / 25.0 for sc in scores])  # normalize to [0,1]
    y_pred  = (y_score > threshold).astype(int)

    sev_true = np.array([s.true_severity for s in test_samples])
    qhi_vals = np.array([sc.qhi for sc in scores])

    # Gates
    gates       = [sc.gate for sc in scores]
    n_auto      = sum(1 for g in gates if g == "AUTO_USE")
    n_review    = sum(1 for g in gates if g == "REVIEW")
    n_block     = sum(1 for g in gates if g == "BLOCK")

    # Latency
    latencies   = [sc.inference_time_ms for sc in scores]

    return {
        "auc_roc":            round(roc_auc_score(y_true, y_score), 4),
        "avg_precision":      round(average_precision_score(y_true, y_score), 4),
        "f1":                 round(f1_score(y_true, y_pred), 4),
        "severity_pearson_r": round(float(np.corrcoef(qhi_vals, sev_true)[0, 1]), 4),
        "avg_latency_ms":     round(float(np.mean(latencies)), 3),
        "p95_latency_ms":     round(float(np.percentile(latencies, 95)), 3),
        "gate_auto_use":      n_auto,
        "gate_review":        n_review,
        "gate_block":         n_block,
        "n_test":             len(test_samples),
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(qhi_res, conf_res, rand_res):
    W = 72
    print(f"\n{'='*W}")
    print(f"  {'QHI-PROBE BENCHMARK RESULTS':^{W-4}}")
    print(f"{'='*W}")
    print(f"\n  {'Method':<28} {'AUC-ROC':>9} {'Avg-P':>9} {'F1':>9}  {'GPU?':>6}")
    print(f"  {'-'*64}")

    def row(name, r, gpu="No"):
        print(f"  {name:<28} {r['auc_roc']:>9.4f} {r['avg_precision']:>9.4f} "
              f"{r['f1']:>9.4f}  {gpu:>6}")

    row("Random Baseline",          rand_res)
    row("Confidence-Only Probe",    conf_res)
    row("QHI-Probe (Ours) ★",       qhi_res)
    print(f"  {'-'*64}")

    print(f"\n  QHI-Probe Extra Metrics:")
    print(f"    Severity Correlation (r):  {qhi_res['severity_pearson_r']:.4f}")
    print(f"    Avg Inference Latency:     {qhi_res['avg_latency_ms']:.3f} ms  (CPU only)")
    print(f"    P95 Inference Latency:     {qhi_res['p95_latency_ms']:.3f} ms")
    print(f"    GPU Required:              None ✅")

    n = qhi_res["n_test"]
    print(f"\n  Operational Gate Distribution (n={n}):")
    print(f"    AUTO_USE  (QHI < 5)  :  {qhi_res['gate_auto_use']:>4}  "
          f"({qhi_res['gate_auto_use']/n*100:.1f}%)")
    print(f"    REVIEW   (5–20)      :  {qhi_res['gate_review']:>4}  "
          f"({qhi_res['gate_review']/n*100:.1f}%)")
    print(f"    BLOCK    (QHI ≥ 20)  :  {qhi_res['gate_block']:>4}  "
          f"({qhi_res['gate_block']/n*100:.1f}%)")
    print(f"\n{'='*W}")


def print_example_scores(system, n=8):
    from data.loader import _DEMO_SAMPLES_RAW

    print(f"\n  EXAMPLE QHI SCORES")
    print(f"  {'─'*70}")
    print(f"  {'Text (truncated)':<38} {'Gate':<12} {'QHI':>6}  {'U':>6} {'R':>5} {'V':>6}")
    print(f"  {'─'*70}")

    for q, correct, hallucinated, risk, entities in _DEMO_SAMPLES_RAW[:n]:
        for is_hal in [False, True]:
            text = hallucinated if is_hal else correct
            sample = ClinicalSample(
                text=f"Q: {q}\nA: {text}",
                entities=list(entities),
                true_label=int(is_hal),
                true_severity=risk * 5.0 if is_hal else 0.5
            )
            sc = system.score(sample)
            tag = "[HAL]" if is_hal else "[OK] "
            short = f"{tag} {text[:32]}.."
            print(f"  {short:<38} {sc.gate:<12} {sc.qhi:>6.2f}  "
                  f"{sc.uncertainty_score:>6.3f} {sc.risk_score:>5.2f} "
                  f"{sc.causal_violation_prob:>6.3f}")

    print(f"  {'─'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="QHI-Probe Benchmark — Clinical Hallucination Detection"
    )
    parser.add_argument("--dataset", nargs="+",
                        default=["demo"],
                        choices=["demo", "medqa", "medmcqa", "truthfulqa"],
                        help="Dataset(s) to use")
    parser.add_argument("--n", type=int, default=500,
                        help="Samples per dataset (default: 500)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden state dimension (256=demo, 768=BERT, 1024=BioMedLM)")
    parser.add_argument("--save-results", action="store_true",
                        help="Save results to results/benchmark_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  QHI-PROBE BENCHMARK")
    print(f"  Dataset: {args.dataset} | N per source: {args.n}")
    print(f"  Benchmark: MedQA-USMLE / MedMCQA / Demo")
    print(f"{'='*72}\n")

    # ── Load Data ─────────────────────────────────────────────────────────────
    print("[1/5] Loading dataset...")
    samples = load_combined(sources=args.dataset, n_each=args.n, seed=args.seed)

    labels = [s.true_label for s in samples]
    train_samples, test_samples = train_test_split(
        samples, test_size=0.2, random_state=args.seed, stratify=labels
    )
    print(f"  Train: {len(train_samples)} | Test: {len(test_samples)}")

    # ── Train QHI-Probe ───────────────────────────────────────────────────────
    print("\n[2/5] Training QHI-Probe...")
    system = QHIProbeSystem(hidden_dim=args.hidden_dim, verbose=True)
    train_stats = system.train(train_samples)

    # ── Evaluate QHI-Probe ────────────────────────────────────────────────────
    print("[3/5] Evaluating QHI-Probe...")
    qhi_results = evaluate(system, test_samples)

    # ── Run Baselines ─────────────────────────────────────────────────────────
    print("[4/5] Running baselines...")

    rand_true, rand_pred = run_random_baseline(test_samples, seed=args.seed)
    rand_results = {
        "auc_roc": round(roc_auc_score(rand_true, rand_pred), 4),
        "avg_precision": round(average_precision_score(rand_true, rand_pred), 4),
        "f1": round(f1_score(rand_true, (rand_pred > 0.5).astype(int)), 4),
    }

    conf_true, conf_pred = run_confidence_only_baseline(train_samples, test_samples)
    conf_results = {
        "auc_roc": round(roc_auc_score(conf_true, conf_pred), 4),
        "avg_precision": round(average_precision_score(conf_true, conf_pred), 4),
        "f1": round(f1_score(conf_true, (conf_pred > 0.5).astype(int)), 4),
    }

    # ── Print Results ─────────────────────────────────────────────────────────
    print("\n[5/5] Results:\n")
    print_results_table(qhi_results, conf_results, rand_results)
    print_example_scores(system, n=5)

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.save_results:
        os.makedirs("results", exist_ok=True)
        output = {
            "benchmark": {
                "datasets": args.dataset,
                "n_per_source": args.n,
                "total_train": len(train_samples),
                "total_test": len(test_samples),
            },
            "qhi_probe": {**qhi_results, **train_stats},
            "baselines": {
                "random": rand_results,
                "confidence_only": conf_results
            }
        }
        path = "results/benchmark_results.json"
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Results saved to {path}")


if __name__ == "__main__":
    main()
