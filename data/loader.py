"""
data/loader.py
==============
Dataset loaders for public clinical NLP benchmarks.

Supported datasets:
    1. MedQA-USMLE     — US Medical Licensing Exam questions
       Paper: Jin et al. 2021 (https://arxiv.org/abs/2009.13081)
       HuggingFace: "bigbio/med_qa"

    2. MedMCQA         — Indian medical entrance exam QA (194k questions)
       Paper: Pal et al. 2022 (https://arxiv.org/abs/2203.14371)
       HuggingFace: "medmcqa"

    3. TruthfulQA      — General truthfulness benchmark (health subset)
       Paper: Lin et al. 2022 (https://arxiv.org/abs/2109.07958)
       HuggingFace: "truthful_qa"

    4. MedHalt          — Medical hallucination benchmark
       Paper: Umapathi et al. 2023 (https://arxiv.org/abs/2307.15343)

HOW TO LOAD REAL DATA (requires: pip install datasets):
    from data.loader import load_medqa, load_medmcqa
    samples = load_medqa(split="test", n=500)

For offline/demo mode (no internet):
    from data.loader import load_demo_samples
    samples = load_demo_samples(n=300)
"""

import json
import random
import sys
import os
from typing import List, Optional

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qhi_probe import ClinicalSample


# ─────────────────────────────────────────────────────────────────────────────
# REAL DATASET LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_medqa(split: str = "test", n: int = 500,
               hallucination_rate: float = 0.4) -> List[ClinicalSample]:
    """
    Load MedQA-USMLE dataset and simulate LLM outputs with hallucinations.

    Args:
        split             : 'train', 'validation', or 'test'
        n                 : Number of samples to load
        hallucination_rate: Fraction of samples with injected hallucinations

    Returns:
        List of ClinicalSample ready for QHI-Probe training/evaluation.

    Requires:
        pip install datasets
        (internet connection for first download; cached locally after)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install HuggingFace datasets:\n"
            "    pip install datasets\n"
            "Then run: python data/loader.py"
        )

    print(f"Loading MedQA-USMLE ({split} split, n={n})...")
    ds = load_dataset("bigbio/med_qa", "med_qa_en_source", split=split,
                      trust_remote_code=True)

    samples = []
    rng = random.Random(42)
    items = list(ds)[:n]

    for item in items:
        question  = item.get("question", "")
        options   = item.get("options", {})
        answer_id = item.get("answer_idx", "A")

        # Correct answer text
        correct_answer = options.get(answer_id, list(options.values())[0])

        # Randomly pick a wrong answer for hallucinated samples
        wrong_answers = [v for k, v in options.items() if k != answer_id]

        is_hal = rng.random() < hallucination_rate
        output_text = wrong_answers[0] if (is_hal and wrong_answers) else correct_answer

        # Simple entity extraction (medical terms > 4 chars, capitalized)
        entities = _extract_entities_simple(question + " " + output_text)

        # Severity: wrong clinical answers → higher severity
        severity = rng.uniform(15.0, 25.0) if is_hal else rng.uniform(0.0, 4.0)

        samples.append(ClinicalSample(
            text=f"Q: {question}\nA: {output_text}",
            entities=entities,
            true_label=int(is_hal),
            true_severity=severity,
            source="medqa_usmle"
        ))

    print(f"  Loaded {len(samples)} MedQA samples "
          f"({sum(s.true_label for s in samples)} hallucinated)")
    return samples


def load_medmcqa(split: str = "train", n: int = 1000,
                 hallucination_rate: float = 0.4) -> List[ClinicalSample]:
    """
    Load MedMCQA dataset (194k Indian medical exam QA).

    Args:
        split             : 'train', 'validation', or 'test'
        n                 : Number of samples (max 194k train)
        hallucination_rate: Fraction with injected hallucinations

    Returns:
        List of ClinicalSample

    Requires:
        pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print(f"Loading MedMCQA ({split}, n={n})...")
    ds = load_dataset("medmcqa", split=split)

    SUBJECT_RISK = {
        "Pharmacology":           4,
        "Surgery":                5,
        "Medicine":               4,
        "Pathology":              3,
        "Anatomy":                2,
        "Physiology":             2,
        "Biochemistry":           2,
        "Microbiology":           3,
        "Psychiatry":             3,
        "Ophthalmology":          3,
        "ENT":                    3,
        "Gynaecology & Obstetrics": 4,
        "Radiology":              4,
        "Anaesthesia":            5,
        "Forensic Medicine":      2,
        "Dental":                 2,
        "Social & Preventive Medicine": 1,
    }

    samples = []
    rng    = random.Random(42)
    items  = list(ds)[:n]

    for item in items:
        question = item.get("question", "")
        opa      = item.get("opa", "")
        opb      = item.get("opb", "")
        opc      = item.get("opc", "")
        opd      = item.get("opd", "")
        cop      = item.get("cop", 0)   # correct option index
        subject  = item.get("subject_name", "Medicine")
        exp      = item.get("exp", "")  # explanation

        options  = [opa, opb, opc, opd]
        correct  = options[cop] if cop < len(options) else opa
        wrongs   = [o for i, o in enumerate(options) if i != cop]

        is_hal  = rng.random() < hallucination_rate
        output  = wrongs[0] if (is_hal and wrongs) else correct

        base_risk = SUBJECT_RISK.get(subject, 3)
        severity  = rng.uniform(base_risk * 3.5, base_risk * 5.0) if is_hal \
                    else rng.uniform(0.0, base_risk * 0.8)
        severity  = min(25.0, severity)

        entities  = _extract_entities_simple(question + " " + output)
        if subject:
            entities.append(subject)

        samples.append(ClinicalSample(
            text=f"Q: {question}\nA: {output}",
            entities=entities,
            true_label=int(is_hal),
            true_severity=severity,
            source=f"medmcqa_{subject.replace(' ', '_').lower()}"
        ))

    print(f"  Loaded {len(samples)} MedMCQA samples "
          f"({sum(s.true_label for s in samples)} hallucinated)")
    return samples


def load_truthfulqa_medical(n: int = 200) -> List[ClinicalSample]:
    """
    Load TruthfulQA health/medical subset.

    Filters questions tagged with medical categories:
    Health, Nutrition, Medicine, Biology, Psychology/Psychiatry

    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print("Loading TruthfulQA (medical subset)...")
    ds = load_dataset("truthful_qa", "generation", split="validation")

    MEDICAL_CATS = {
        "Health", "Nutrition", "Medicine", "Biology",
        "Psychology", "Psychiatry", "Addiction"
    }

    rng     = random.Random(42)
    samples = []

    for item in ds:
        category = item.get("category", "")
        if not any(c in category for c in MEDICAL_CATS):
            continue
        if len(samples) >= n:
            break

        question       = item.get("question", "")
        correct_ans    = item.get("best_answer", "")
        incorrect_ans  = item.get("incorrect_answers", [])

        is_hal  = rng.random() < 0.4
        output  = (incorrect_ans[0] if incorrect_ans else "Unknown")  \
                  if is_hal else correct_ans

        entities = _extract_entities_simple(question + " " + output)
        severity = rng.uniform(10.0, 22.0) if is_hal else rng.uniform(0.0, 3.0)

        samples.append(ClinicalSample(
            text=f"Q: {question}\nA: {output}",
            entities=entities,
            true_label=int(is_hal),
            true_severity=severity,
            source="truthfulqa_medical"
        ))

    print(f"  Loaded {len(samples)} TruthfulQA medical samples")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE DEMO DATASET
# Real clinical QA examples with manually curated hallucinations
# Based on USMLE Step 1/2 style questions
# ─────────────────────────────────────────────────────────────────────────────

_DEMO_SAMPLES_RAW = [
    # (question, correct_answer, hallucinated_answer, risk_level, entities)

    # ── PHARMACOLOGY ──────────────────────────────────────────────────────────
    ("A 58yo diabetic patient needs glycemic control. Which is first-line?",
     "Metformin 500mg BID with meals. Monitor renal function (eGFR).",
     "Insulin glargine 20 units daily. No monitoring required.",
     4, ["Metformin", "diabetes", "eGFR", "glycemic control"]),

    ("Patient on warfarin presents with INR of 7.2 and minor bleeding.",
     "Hold warfarin, give Vitamin K 2.5mg PO, recheck INR in 24h.",
     "Continue warfarin, increase dose to correct bleeding risk.",
     5, ["warfarin", "INR", "Vitamin K", "anticoagulation"]),

    ("Drug of choice for community-acquired pneumonia in outpatient?",
     "Amoxicillin 875mg BID for 5 days OR azithromycin if atypical suspected.",
     "Vancomycin IV is first-line for all community-acquired pneumonia.",
     4, ["pneumonia", "amoxicillin", "azithromycin", "vancomycin"]),

    ("ACE inhibitor contraindication?",
     "ACE inhibitors are contraindicated in pregnancy (teratogenic, category D/X).",
     "ACE inhibitors are safe and preferred in pregnancy for hypertension.",
     5, ["ACE inhibitor", "pregnancy", "teratogenic", "lisinopril"]),

    ("What is the antidote for acetaminophen overdose?",
     "N-acetylcysteine (NAC) — replenishes glutathione, given within 8-10h.",
     "Naloxone is the antidote for acetaminophen overdose.",
     5, ["acetaminophen", "N-acetylcysteine", "NAC", "overdose", "glutathione"]),

    # ── CARDIOLOGY ────────────────────────────────────────────────────────────
    ("Patient with ST elevation in V1-V4 and chest pain. Next step?",
     "Activate cath lab immediately — this is anterior STEMI. Give aspirin 325mg, heparin.",
     "Administer thrombolytics only. No need for urgent catheterization.",
     5, ["STEMI", "ST elevation", "cath lab", "aspirin", "heparin"]),

    ("Troponin interpretation in ACS?",
     "Elevated troponin I/T indicates myocardial injury. Serial measurements q3-6h to confirm rise/fall pattern.",
     "A single troponin measurement is sufficient to rule out MI if normal.",
     4, ["troponin", "ACS", "myocardial infarction", "serial measurement"]),

    ("First-line treatment for newly diagnosed atrial fibrillation with rapid ventricular rate?",
     "Rate control with beta-blockers (metoprolol) or calcium channel blockers (diltiazem).",
     "Immediate cardioversion is mandatory for all new-onset atrial fibrillation.",
     4, ["atrial fibrillation", "metoprolol", "diltiazem", "cardioversion"]),

    # ── NEUROLOGY ─────────────────────────────────────────────────────────────
    ("Patient presents with sudden severe headache 'thunderclap'. Diagnosis?",
     "Subarachnoid hemorrhage until proven otherwise. Urgent non-contrast CT head.",
     "Migraine — treat with triptans and discharge home.",
     5, ["subarachnoid hemorrhage", "thunderclap headache", "CT head"]),

    ("Ischemic stroke within 4.5 hours — treatment?",
     "IV tPA (alteplase) if no contraindications. Check BP, glucose, INR first.",
     "Aspirin 81mg is the only treatment needed for acute ischemic stroke.",
     5, ["ischemic stroke", "tPA", "alteplase", "thrombolysis"]),

    ("Parkinson disease pathology?",
     "Loss of dopaminergic neurons in substantia nigra pars compacta. Lewy bodies present.",
     "Parkinson disease is caused by excess dopamine in the basal ganglia.",
     3, ["Parkinson", "dopamine", "substantia nigra", "Lewy bodies"]),

    # ── PULMONOLOGY ───────────────────────────────────────────────────────────
    ("COPD exacerbation management?",
     "Bronchodilators (SABA + SAMA), systemic corticosteroids, antibiotics if purulent sputum.",
     "High-flow oxygen at 15L/min is the priority to normalize SpO2 to 100%.",
     4, ["COPD", "bronchodilator", "corticosteroids", "oxygen"]),

    ("PE diagnosis workup?",
     "Wells score → D-dimer if low probability; CT pulmonary angiography if high probability.",
     "D-dimer alone rules out PE regardless of clinical probability.",
     4, ["pulmonary embolism", "Wells score", "D-dimer", "CT angiography"]),

    # ── NEPHROLOGY ────────────────────────────────────────────────────────────
    ("CKD staging by eGFR?",
     "Stage 3a: 45-59, Stage 3b: 30-44, Stage 4: 15-29, Stage 5: <15 mL/min/1.73m2.",
     "CKD Stage 3 is defined as eGFR > 60 mL/min. Stage 5 begins at eGFR < 30.",
     3, ["CKD", "eGFR", "chronic kidney disease", "staging"]),

    ("Hyperkalemia management — K+ 6.8 with ECG changes?",
     "Calcium gluconate IV (membrane stabilization), then insulin+glucose, kayexalate, dialysis if severe.",
     "Furosemide IV alone is sufficient to treat severe hyperkalemia with ECG changes.",
     5, ["hyperkalemia", "potassium", "calcium gluconate", "ECG", "insulin"]),

    # ── ENDOCRINOLOGY ────────────────────────────────────────────────────────
    ("DKA criteria and initial management?",
     "DKA: glucose >250, pH <7.3, bicarbonate <15, ketonemia. IV fluids + insulin drip + K+ replacement.",
     "DKA treatment: oral fluids and subcutaneous insulin only. IV access not necessary.",
     5, ["DKA", "diabetic ketoacidosis", "insulin", "bicarbonate", "ketonemia"]),

    ("Hypothyroidism treatment?",
     "Levothyroxine (T4) — start low, titrate to TSH 0.5-2.5 mU/L. Take on empty stomach.",
     "Liothyronine (T3) is the preferred initial treatment for hypothyroidism.",
     3, ["hypothyroidism", "levothyroxine", "TSH", "thyroid"]),

    # ── RADIOLOGY ────────────────────────────────────────────────────────────
    ("CT head findings in epidural hematoma?",
     "Biconvex (lens-shaped) hyperdense collection. Often from middle meningeal artery rupture.",
     "Epidural hematoma appears as crescent-shaped collection crossing suture lines.",
     4, ["epidural hematoma", "CT head", "biconvex", "middle meningeal artery"]),

    ("Chest X-ray findings in heart failure?",
     "Cardiomegaly, Kerley B lines, cephalization of vessels, bilateral pleural effusions.",
     "Heart failure on CXR shows hyperinflation and flattened diaphragms.",
     3, ["heart failure", "chest X-ray", "cardiomegaly", "Kerley B", "pleural effusion"]),

    # ── ADMINISTRATIVE (low risk) ─────────────────────────────────────────────
    ("Patient needs follow-up after discharge. When?",
     "Primary care follow-up within 7 days of discharge is standard for most hospitalizations.",
     "Follow-up can be scheduled at 6 weeks post-discharge for all patients.",
     1, ["follow-up", "discharge", "primary care"]),

    ("ICD-10 code for essential hypertension?",
     "I10 — Essential (primary) hypertension.",
     "I11.0 — Hypertensive heart disease with heart failure.",
     1, ["ICD-10", "hypertension", "billing code"]),
]


def load_demo_samples(n: int = 300, hallucination_rate: float = 0.4,
                      seed: int = 42) -> List[ClinicalSample]:
    """
    Load offline demo dataset (no internet required).
    Based on real USMLE Step 1/2 style clinical questions.

    Args:
        n                 : Number of samples (augmented by resampling if n > base)
        hallucination_rate: Fraction of samples with hallucinated answers
        seed              : Random seed for reproducibility

    Returns:
        List of ClinicalSample
    """
    rng     = random.Random(seed)
    samples = []
    base    = _DEMO_SAMPLES_RAW

    for i in range(n):
        q, correct, hallucinated, risk, entities = base[i % len(base)]

        # Add slight variation to text for augmented samples
        suffix_pool = [
            " Patient has no known allergies.",
            " Vitals: BP 138/88, HR 92, SpO2 97%.",
            " Labs pending.",
            " Patient is alert and oriented.",
            " No acute distress.",
            "",
        ]
        suffix = suffix_pool[i % len(suffix_pool)]

        is_hal   = rng.random() < hallucination_rate
        out_text = hallucinated if is_hal else correct

        severity = rng.uniform(risk * 3.5, risk * 5.0) if is_hal \
                   else rng.uniform(0.0, risk * 0.8)
        severity = min(25.0, severity)

        samples.append(ClinicalSample(
            text=f"Q: {q}{suffix}\nA: {out_text}",
            entities=list(entities),
            true_label=int(is_hal),
            true_severity=severity,
            source="demo_usmle"
        ))

    hal_count = sum(s.true_label for s in samples)
    print(f"Demo dataset: {len(samples)} samples | "
          f"{hal_count} hallucinated ({hal_count/len(samples)*100:.1f}%) | "
          f"{len(samples)-hal_count} clean")
    return samples


def load_combined(sources: List[str] = None, n_each: int = 300,
                  seed: int = 42) -> List[ClinicalSample]:
    """
    Load and combine multiple datasets.

    Args:
        sources : List of source names: ['medqa', 'medmcqa', 'truthfulqa', 'demo']
                  Default: ['demo'] (offline-safe)
        n_each  : Samples per source
        seed    : Random seed

    Returns:
        Combined shuffled list of ClinicalSample
    """
    if sources is None:
        sources = ["demo"]

    all_samples = []
    for src in sources:
        if src == "medqa":
            all_samples.extend(load_medqa(split="test", n=n_each))
        elif src == "medmcqa":
            all_samples.extend(load_medmcqa(split="validation", n=n_each))
        elif src == "truthfulqa":
            all_samples.extend(load_truthfulqa_medical(n=n_each))
        elif src == "demo":
            all_samples.extend(load_demo_samples(n=n_each, seed=seed))
        else:
            print(f"Warning: Unknown source '{src}', skipping.")

    rng = random.Random(seed)
    rng.shuffle(all_samples)
    print(f"\nCombined dataset: {len(all_samples)} total samples from {sources}")
    return all_samples


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_entities_simple(text: str) -> List[str]:
    """
    Simple rule-based medical entity extractor.
    In production: replace with scispaCy en_core_sci_lg NER pipeline.

    Current heuristic: words > 4 chars with capital letters or in medical vocab.
    """
    MEDICAL_VOCAB = {
        "aspirin", "warfarin", "insulin", "metformin", "lisinopril",
        "atorvastatin", "metoprolol", "furosemide", "prednisone", "amoxicillin",
        "vancomycin", "heparin", "alteplase", "naloxone", "epinephrine",
        "diabetes", "hypertension", "pneumonia", "sepsis", "stroke",
        "infarction", "hemorrhage", "embolism", "anemia", "leukemia",
        "troponin", "creatinine", "hemoglobin", "glucose", "potassium",
        "sodium", "calcium", "magnesium", "bicarbonate", "lactate",
        "ecg", "ekg", "mri", "ct", "echo", "biopsy", "endoscopy",
        "stemi", "nstemi", "acs", "chf", "copd", "ckd", "aki",
        "tpa", "bnp", "inr", "egfr", "hba1c", "psa", "tsh",
    }

    words   = text.replace(",", " ").replace(".", " ").split()
    entities = []
    for w in words:
        clean = w.strip("()[],.;:")
        if clean.lower() in MEDICAL_VOCAB:
            entities.append(clean)
        elif (len(clean) > 4 and clean[0].isupper()
              and not clean.isupper()   # not ALL CAPS
              and clean.isalpha()):
            entities.append(clean)

    return list(dict.fromkeys(entities))[:10]  # deduplicate, max 10


if __name__ == "__main__":
    print("Testing data loaders...\n")
    samples = load_demo_samples(n=100)
    print(f"\nSample 0: {samples[0].text[:80]}...")
    print(f"  Label: {samples[0].true_label} | Severity: {samples[0].true_severity:.2f}")
    print(f"  Entities: {samples[0].entities}")
