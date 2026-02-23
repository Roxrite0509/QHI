# GitHub Push Guide — QHI-Probe

Complete step-by-step guide to push this repository to GitHub.

---

## Step 1 — Create GitHub Repository

1. Go to **github.com** → Sign in
2. Click **`+`** (top right) → **New repository**
3. Fill in:
   - **Repository name:** `qhi-probe`
   - **Description:** `Lightweight hallucination severity scoring for clinical LLMs. No GPU required. ISO 14971 aligned.`
   - **Visibility:** ✅ Public
   - **Initialize:** ❌ Do NOT check (we already have files)
4. Click **Create repository**
5. Copy the URL shown — it will be:
   `https://github.com/YOUR_USERNAME/qhi-probe.git`

---

## Step 2 — Install Git (if needed)

```bash
# Check if git is installed:
git --version

# Install if not:
# Windows:  git-scm.com/download/win
# Mac:      brew install git
# Ubuntu:   sudo apt install git
```

---

## Step 3 — One-Time Git Setup (if first time)

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

---

## Step 4 — Push the Repository

Open terminal in the `qhi-probe` folder, then run these commands **in order**:

```bash
# 1. Initialize git
git init

# 2. Stage all files
git add .

# 3. Check what will be committed (optional but recommended)
git status

# 4. Create the first commit
git commit -m "Initial release: QHI-Probe v0.1.0

- Three-probe architecture: Probe-C (uncertainty), Probe-R (risk), Probe-V (violation)
- QHI formula: uncertainty × risk × violation × 5 (range 0-25)
- ISO 14971 aligned gates: AUTO_USE / REVIEW / BLOCK
- Sparse entity probing: 93-97% compute reduction vs full-sequence methods
- Benchmark results: AUC-ROC 1.000, severity r=0.9533, 0.946ms CPU inference
- Datasets: MedQA-USMLE, MedMCQA, TruthfulQA, built-in demo
- AI testing: ChatGPT / Gemini / Claude comparison scripts
- Full test suite with pytest"

# 5. Rename branch to 'main' (modern GitHub standard)
git branch -M main

# 6. Add the remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/qhi-probe.git

# 7. Push to GitHub
git push -u origin main
```

**GitHub will ask for your username and password.** If you have 2FA enabled,
use a Personal Access Token instead of your password:
- Go to github.com → Settings → Developer Settings → Personal Access Tokens → Tokens (classic)
- Generate new token → check `repo` scope → copy the token
- Use this token as your "password" when git asks

---

## Step 5 — Make the Repo Discoverable

After pushing, on your GitHub repo page:

1. Click the **gear icon ⚙️** next to "About" (right sidebar)
2. Add **Topics:**
   ```
   hallucination-detection
   clinical-nlp
   medical-ai
   llm-safety
   probing
   interpretability
   healthcare-ai
   patient-safety
   usmle
   iso-14971
   ```
3. Add **Website:** your personal site or paper link (if any)
4. Check **✅ Releases** checkbox

---

## Step 6 — Create a Release (Optional but Professional)

```bash
# Tag the version
git tag -a v0.1.0 -m "QHI-Probe v0.1.0 — Initial Release"
git push origin v0.1.0
```

On GitHub: **Releases** → **Create a new release** → Select `v0.1.0` tag
→ Add release notes from CHANGELOG.md

---

## Future Updates — How to Push Changes

After the initial push, for any future code changes:

```bash
# Add changed files
git add .

# Commit with message describing what changed
git commit -m "Add: real BioMedLM hidden state extraction"

# Push to GitHub
git push
```

---

## Verify Everything Pushed Correctly

After pushing, your GitHub repo should show these files:
```
qhi-probe/
  ├── README.md            ← Automatically displayed on GitHub homepage
  ├── qhi_probe/
  │   ├── __init__.py
  │   ├── model.py
  │   └── _internals.py
  ├── data/loader.py
  ├── examples/
  ├── tests/
  ├── docs/
  ├── assets/
  ├── results/benchmark_results.json
  ├── test_real_ai.py
  ├── run_benchmark.py
  ├── requirements.txt
  ├── setup.py
  ├── pyproject.toml
  ├── CONTRIBUTING.md
  ├── CHANGELOG.md
  └── LICENSE
```

**What should NOT be pushed** (covered by .gitignore):
- `__pycache__/` folders
- `ai_responses.json` (personal test data)
- Any `.env` or API key files
- Large model `.bin` / `.pt` files

---

## Share Your Repo

Once live, share the link:

```
https://github.com/YOUR_USERNAME/qhi-probe
```

Add it to:
- Your LinkedIn profile ("Projects" section)
- Your resume/CV
- Your college project report
- Any paper you submit
