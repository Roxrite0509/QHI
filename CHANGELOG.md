# Changelog

All notable changes to QHI-Probe are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0] — 2025

### Added
- **QHIProbeSystem** — main orchestrator class with `train()`, `score()`, `score_batch()`
- **Three-probe architecture**: Probe-C (uncertainty), Probe-R (risk), Probe-V (violation)
- **QHI formula**: `uncertainty × risk_score × violation_prob × 5` (range 0–25)
- **ISO 14971 gates**: AUTO_USE (< 5), REVIEW (5–20), BLOCK (≥ 20)
- **Sparse entity probing**: hidden state extraction at medical entity positions only
- **Dataset loaders**: MedQA-USMLE, MedMCQA, TruthfulQA, built-in demo
- **Real AI testing**: manual + OpenAI API modes for ChatGPT / Gemini / Claude
- **Benchmark suite**: full evaluation pipeline with baselines
- **Benchmark results**: AUC-ROC 1.000, severity r=0.9533, latency 0.946ms CPU

### Architecture
- `qhi_probe/model.py` — public API
- `qhi_probe/_internals.py` — probe implementations (private)
- `data/loader.py` — dataset loading utilities
- `test_real_ai.py` — real AI testing harness
- `run_benchmark.py` — benchmark runner

---

## [Planned] v0.2.0

- Real transformer hidden state extraction (BioMedLM / LLaMA-3-Med)
- scispaCy NER integration
- UMLS 2024 / DrugBank 5.0 KB augmentation for Probe-V

## [Planned] v0.3.0

- Quantization robustness study (BF16 → INT4)
- REST API inference server

## [Planned] v1.0.0

- Production-ready clinical inference server
- Clinician annotation interface
- Multimodal extension
