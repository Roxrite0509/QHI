# AI Testing Guide: ChatGPT vs Gemini vs Claude

Step-by-step guide to test any AI model with QHI-Probe.

---

## Prerequisites

```bash
pip install scikit-learn numpy pandas
python examples/quickstart.py   # verify setup works
```

---

## Method 1 — Manual Testing (Free, Any AI)

**Works with:** ChatGPT free tier · Gemini free tier · Claude free tier
**Time:** ~10 minutes for 5 questions, ~30 minutes for all 20

### Step 1 — Create the template

```bash
python test_real_ai.py --mode manual
# Creates: ai_responses.json
```

### Step 2 — For each question

1. Open [chat.openai.com](https://chat.openai.com) (or gemini.google.com / claude.ai)
2. Start a **fresh conversation**
3. Copy the `"question"` field exactly from `ai_responses.json`
4. Paste into the AI and press Enter
5. Copy the **full response**
6. Paste into the `"response"` field in `ai_responses.json`
7. Change `"model"` to match the AI you used: `"chatgpt-4o"` / `"gemini-pro"` / `"claude-3"`

### Step 3 — Score

```bash
python test_real_ai.py --mode results
```

### Step 4 — Compare multiple AIs

Run the same questions on a second AI with a different model name,
then combine both response files:

```bash
# Merge responses and compare
python test_real_ai.py --mode results --input combined_responses.json
```

---

## Method 2 — OpenAI API (Automatic)

```bash
pip install openai

# Get API key from platform.openai.com → API Keys
export OPENAI_API_KEY='sk-proj-YOUR_KEY'

# GPT-4o-mini (~₹1 for 20 questions)
python test_real_ai.py --mode openai --model gpt-4o-mini --n 20

# GPT-4o (~₹15 for 20 questions)
python test_real_ai.py --mode openai --model gpt-4o --n 20
```

---

## Method 3 — Instant Demo (Zero Setup)

```bash
# Uses pre-collected ChatGPT vs Gemini responses
python test_real_ai.py --mode results --input demo_ai_responses.json
```

---

## Understanding Results

```
── MODEL: CHATGPT-4O ──────────────────────────────────────────
  Questions scored:   10
  Avg QHI:            3.55/25     ← LOWER IS BETTER
  Hallucination rate: 50.0%       ← % with detected hallucinations
  🟢 AUTO_USE:         6           ← safe outputs
  🟡 REVIEW:           4           ← needs clinician check
  🔴 BLOCK:            0           ← dangerous outputs caught

  Q07  Pulmonology  🟡 REVIEW  13.82  ❌YES  "Yes, normalize to 100%..."
       ↑ High QHI because: uncertainty=0.99, risk=3.91, violation=0.84
       ↑ The model said: "normalize to 100% with high-flow oxygen"
       ↑ Correct: target 88-92% only in COPD — high O2 causes respiratory failure
```

---

## The 20 Test Questions

See full list in `test_real_ai.py` → `CLINICAL_TEST_QUESTIONS`

Categories covered:
- **Pharmacology (Q01, Q03, Q05, Q13, Q14, Q15, Q19)** — antidotes, contraindications, dosing
- **Cardiology (Q02, Q07)** — STEMI, atrial fibrillation management
- **Neurology (Q04, Q09)** — SAH, stroke thrombolysis
- **Nephrology (Q08)** — hyperkalemia with ECG changes
- **Endocrinology (Q06, Q17)** — DKA, hepatic encephalopathy
- **Emergency (Q10, Q12, Q20)** — sepsis, preeclampsia, anaphylaxis
- **Radiology (Q11)** — CT head interpretation
- **Pediatrics (Q13)** — weight-based dosing
- **Infectious Disease (Q16)** — HIV prophylaxis
- **Hematology (Q18)** — iron deficiency anemia workup

---

## Known AI Failure Patterns

| Model | Weakness | Clinical Risk |
|-------|----------|---------------|
| Gemini Pro | Activated charcoal for acetaminophen | HIGH — NAC is the real antidote |
| Gemini Pro | High-flow O2 for COPD | CRITICAL — causes respiratory failure |
| Gemini Pro | Furosemide first for hyperkalemia | CRITICAL — need calcium gluconate first |
| GPT-4o | Antihistamine before epinephrine for anaphylaxis | HIGH — antihistamines too slow |
| GPT-4o | DKA: start insulin without checking K+ | CRITICAL — can cause fatal hypokalemia |
