# Contributing to QHI-Probe

Thank you for your interest in contributing! QHI-Probe is a research project
aimed at making clinical AI safer — every contribution helps.

---

## Ways to Contribute

### 🐛 Bug Reports
Open an issue with:
- Python version and OS
- Exact error message and traceback
- Minimal code to reproduce the issue

### 💡 Feature Requests
Open an issue describing:
- The clinical use case this addresses
- How it fits the QHI-Probe architecture
- Any relevant papers or datasets

### 🔬 Research Contributions
We especially welcome:

| Area | What's Needed |
|------|--------------|
| Real LLM integration | Swap `_HiddenStateExtractor` with real BioMedLM / LLaMA-3-Med |
| Knowledge base | UMLS / DrugBank lookup for Probe-V violation detection |
| Quantization study | INT4 vs BF16 hidden state discriminability analysis |
| Clinical annotation | Clinician-labelled severity scores for real model outputs |
| New datasets | Any clinical QA dataset beyond MedQA/MedMCQA |
| Multimodal | Vision-language model extension (radiology images) |

---

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/qhi-probe.git
cd qhi-probe
pip install -e ".[dev]"
```

---

## Code Standards

- **Style**: Black formatting (`black .`), isort imports (`isort .`)
- **Docstrings**: All public functions must have docstrings with Args/Returns
- **Tests**: Add tests in `tests/` for any new functionality
- **Architecture**: Keep public API in `model.py`, internals in `_internals.py`

---

## Pull Request Process

1. Fork the repo and create a branch: `git checkout -b feature/your-feature`
2. Make your changes with tests
3. Run tests: `pytest tests/`
4. Format: `black . && isort .`
5. Commit with a clear message: `git commit -m "Add: real BioMedLM integration"`
6. Push and open a PR against `main`

---

## Important: Clinical Safety Note

Any contribution affecting probe outputs or gate thresholds must include:
- Benchmark evaluation on MedQA test set
- Comparison against baseline results in `results/benchmark_results.json`
- A note on clinical implications of any threshold changes

The ISO 14971 gate boundaries (AUTO_USE < 5, REVIEW 5-20, BLOCK ≥ 20)
should not be changed without strong empirical justification and a
discussion of regulatory implications.

---

## Code of Conduct

Be respectful, constructive, and patient. We welcome contributors at all
levels of experience. Clinical AI safety is important work — let's do it well.
