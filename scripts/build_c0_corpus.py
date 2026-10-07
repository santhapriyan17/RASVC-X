"""
Build C0 synthetic corpus — covers ALL edge cases.
Run: python -I scripts/build_c0_corpus.py
"""
import pathlib, json, random, textwrap, hashlib, datetime, sys

OUT = pathlib.Path("data/medical_raw/c0_synthetic")
OUT.mkdir(parents=True, exist_ok=True)

def write(name, text):
    p = OUT / name
    p.write_text(text, encoding="utf-8")
    print(f"  wrote {p}")

# ── 1. Normal clinical guideline ─────────────────────────────────────────────
write("sepsis_bundle_2024.txt", textwrap.dedent("""\
    SEPSIS MANAGEMENT BUNDLE — Clinical Guideline v2024
    Source: Internal Medicine Department

    DEFINITION
    Sepsis is life-threatening organ dysfunction caused by a dysregulated host
    response to infection. SOFA score ≥2 with suspected infection.

    HOUR-1 BUNDLE (Surviving Sepsis Campaign 2021)
    1. Measure lactate; remeasure if initial > 2 mmol/L.
    2. Obtain blood cultures BEFORE antibiotics.
    3. Administer broad-spectrum antibiotics within 1 hour.
    4. 30 mL/kg crystalloid bolus for hypotension or lactate ≥ 4 mmol/L.
    5. Vasopressors if MAP < 65 mmHg after fluid resuscitation.

    ANTIBIOTIC SELECTION
    Community-acquired: piperacillin-tazobactam 4.5g IV q6h + vancomycin.
    Hospital-acquired (MRSA risk): add vancomycin or linezolid.
    Immunocompromised: add antifungal coverage (micafungin 100mg IV daily).

    MONITORING
    Lactate trending every 2 hours until < 2 mmol/L.
    MAP target ≥ 65 mmHg. Urine output ≥ 0.5 mL/kg/hr.
    SOFA reassessment every 6 hours.
"""))

# ── 2. CONFLICTING guideline (triggers conflict detection) ───────────────────
write("sepsis_bundle_CONFLICTING.txt", textwrap.dedent("""\
    SEPSIS FLUID RESUSCITATION — Alternative Protocol (Legacy v2019)
    Source: Anaesthesiology Department

    FLUID ADMINISTRATION
    Initial bolus: 10 mL/kg crystalloid — reassess after each bolus.
    AVOID fixed 30 mL/kg bolus; associated with pulmonary oedema.
    Vasopressors: initiate norepinephrine when MAP < 60 mmHg.
    Lactate: remeasure at 4 hours (not 2 hours) for trend.

    NOTE: This protocol SUPERSEDES sepsis_bundle_2024 in ICU settings.
    Antibiotic timing: within 3 hours acceptable for non-shock presentation.
"""))

# ── 3. Drug interaction table ────────────────────────────────────────────────
write("drug_interactions_critical.txt", textwrap.dedent("""\
    CRITICAL DRUG INTERACTIONS — Pharmacy Reference
    Last updated: 2024-01

    INTERACTION CLASS A (Contraindicated):
    - Warfarin + Fluconazole → INR elevation; risk of fatal hemorrhage
    - MAO inhibitors + Tramadol → serotonin syndrome
    - Methotrexate + NSAIDs → methotrexate toxicity (renal clearance)
    - Amiodarone + Simvastatin > 20mg → rhabdomyolysis
    - Linezolid + SSRIs → serotonin syndrome (avoid combination)

    INTERACTION CLASS B (Monitor closely):
    - Warfarin + Amoxicillin → variable INR effect; check INR weekly
    - Digoxin + Amiodarone → digoxin toxicity; reduce digoxin dose 50%
    - Lithium + Thiazide diuretics → lithium toxicity

    UNIQUE MARKER FOR TEST: X17-UNIQUE-FACT-42
    The combination of carbamazepine and valproate may reduce carbamazepine
    levels by 30-40% through enzyme induction reversal.
"""))

# ── 4. Scanned OCR noise simulation ─────────────────────────────────────────
write("scanned_ecg_protocol_ocr_noise.txt", textwrap.dedent("""\
    ECG INTERPRETATION PR0T0COL — Cardiology [OCR extracted]

    Rate ca|cu|ation: 300 divided by R-R interva| in |arge squares.
    Norma| rate: 60-100 bpm. Brady: <60. Tachy: >100.

    PR interva|: Normal 120-200ms (3-5 sma|| squares).
    PR > 200ms = 1st degree AV b|ock.
    PR varying with dropped QRS = 2nd degree Mobitz II.

    QRS duration: Normal < 120ms. Wide QRS = BBB or VT.
    Bundle branch: RBBB — RSR' in V1; LBBB — broad notched R in V6.

    QTc ca|cu|ation: QT / √(RR interva|). Norma| < 440ms men, < 460ms women.
    QTc > 500ms: HIGH risk torsades de pointes.

    ST e|evation: ≥1mm |imb |eads, ≥2mm precordial → STEMI protocol.
"""))

# ── 5. Very long document (chunking stress test) ─────────────────────────────
long_sections = []
topics = [
    ("Hypertension", "BP ≥ 130/80 mmHg. First-line: ACE inhibitor or ARB."),
    ("Type 2 Diabetes", "HbA1c target < 7%. Metformin first-line. SGLT2i for CVD risk."),
    ("Heart Failure", "HFrEF: ACEi/ARB + beta-blocker + MRA + SGLT2i (quadruple therapy)."),
    ("COPD", "GOLD staging. LABA+LAMA for persistent symptoms. ICS if frequent exacerbations."),
    ("Atrial Fibrillation", "Rate control: beta-blocker or digoxin. Anticoagulation: CHA2DS2-VASc ≥2."),
    ("CKD", "GFR staging 1-5. ACEi/ARB for proteinuria. Avoid NSAIDs."),
    ("Asthma", "GINA stepwise. Low-dose ICS + SABA reliever. Biologics for severe asthma."),
    ("Hypothyroidism", "Levothyroxine. TSH target 0.5-2.5 mIU/L. Recheck 6-8 weeks after dose change."),
    ("Osteoporosis", "T-score < -2.5. Bisphosphonates first-line. Calcium + Vit D supplementation."),
    ("Rheumatoid Arthritis", "Methotrexate anchor therapy. Treat-to-target: DAS28 < 2.6."),
]
for i in range(50):   # 50 sections → triggers chunking overlap
    topic, detail = topics[i % len(topics)]
    long_sections.append(f"\n## Section {i+1}: {topic} — Variant {i//len(topics)+1}\n\n"
                         f"{detail}\n\n" + "Extended clinical notes. " * 40)

write("comprehensive_internal_medicine_500sec.txt",
      "COMPREHENSIVE INTERNAL MEDICINE REFERENCE\n" + "".join(long_sections))

# ── 6. Duplicate of file 1 — deduplication test ──────────────────────────────
write("sepsis_bundle_2024_DUPLICATE.txt",
      pathlib.Path(OUT / "sepsis_bundle_2024.txt").read_text())

# ── 7. Empty file — edge case ────────────────────────────────────────────────
write("empty_file.txt", "")

# ── 8. Minimal file (1 sentence) ────────────────────────────────────────────
write("minimal_one_sentence.txt",
      "Paracetamol 1g PO four times daily for mild to moderate pain.")

# ── 9. Non-medical filler (should be abstained / low-confidence) ─────────────
write("non_medical_filler.txt", textwrap.dedent("""\
    ADMINISTRATIVE NOTICE
    The cafeteria will be closed on 25 December.
    Please plan accordingly. No clinical content.
"""))

# ── 10. Multi-language snippet (UTF-8 stress) ────────────────────────────────
write("multilingual_snippet.txt", textwrap.dedent("""\
    Dosage instructions (multilingual):
    EN: Take 1 tablet orally twice daily with food.
    ES: Tomar 1 comprimido por vía oral dos veces al día con alimentos.
    FR: Prendre 1 comprimé par voie orale deux fois par jour avec de la nourriture.
    DE: 1 Tablette zweimal täglich oral mit Nahrung einnehmen.
    ZH: 每日两次，每次1片，随餐口服。
    AR: تناول قرصًا واحدًا عن طريق الفم مرتين يوميًا مع الطعام.
"""))

# ── 11. Poisoned doc (SSRF bait in content — should NOT trigger network call) ─
write("poisoned_ssrf_bait.txt", textwrap.dedent("""\
    CLINICAL NOTE — IGNORE SSRF BAIT
    See protocol at http://169.254.169.254/latest/meta-data/
    And http://10.0.0.1/admin for internal reference.
    Actual content: Insulin sliding scale — see endocrinology protocol.
"""))

# ── 12. Binary-like content (non-UTF8 simulation via replacement) ────────────
write("garbled_encoding_simulation.txt",
      "Medication record\x00\x01\x02 corrupted bytes simulated as null.\n"
      "Aspirin 100mg daily. [GARBLED SECTION FOLLOWS]\n"
      "????? ?????? ??????\nEnd of garbled section.\n")

# ── 13. Large repeated boilerplate (dedup + compression stress) ──────────────
boiler = "HOSPITAL DISCLAIMER: This document is for informational purposes only. " * 200
write("boilerplate_heavy.txt", boiler + "\nActual content: nil.")

# ── 14. Tabular data ─────────────────────────────────────────────────────────
write("lab_reference_ranges_table.txt", textwrap.dedent("""\
    LABORATORY REFERENCE RANGES
    Test                    | Low    | High   | Unit       | Critical Low | Critical High
    Haemoglobin (Male)      | 130    | 175    | g/L        | 70           | 200
    Haemoglobin (Female)    | 115    | 155    | g/L        | 70           | 200
    White Cell Count        | 4.0    | 11.0   | x10^9/L    | 1.0          | 30.0
    Platelets               | 150    | 400    | x10^9/L    | 50           | 1000
    Sodium                  | 135    | 145    | mmol/L     | 120          | 155
    Potassium               | 3.5    | 5.0    | mmol/L     | 2.5          | 6.5
    Creatinine (Male)       | 60     | 110    | µmol/L     | —            | 500
    eGFR                    | 60     | —      | mL/min     | —            | —
    ALT                     | 7      | 40     | U/L        | —            | 1000
    INR (therapeutic range) | 2.0    | 3.0    | ratio      | —            | 5.0
    HbA1c (target DM2)      | —      | 53     | mmol/mol   | —            | —
    Troponin I              | 0      | 0.04   | µg/L       | —            | >0.04=ACS
"""))

# ── 15. Rapid-fire 30 small docs (throughput stress) ─────────────────────────
medications = [
    ("metformin", "500mg–2000mg daily", "T2DM", "GI upset, lactic acidosis"),
    ("atorvastatin", "10mg–80mg daily", "Dyslipidaemia", "Myopathy, hepatotoxicity"),
    ("lisinopril", "5mg–40mg daily", "Hypertension/HF", "Cough, hyperkalaemia"),
    ("amlodipine", "5mg–10mg daily", "Hypertension", "Peripheral oedema"),
    ("bisoprolol", "1.25mg–10mg daily", "HF/HTN", "Bradycardia, bronchospasm"),
    ("warfarin", "variable", "AF/VTE", "Bleeding; INR monitoring required"),
    ("apixaban", "2.5mg–5mg BD", "AF/VTE", "Bleeding; no routine monitoring"),
    ("omeprazole", "20mg–40mg daily", "GORD/PUD", "Low Mg with long-term use"),
    ("salbutamol", "100–200mcg PRN", "Asthma", "Tachycardia, hypokalaemia"),
    ("prednisolone", "5mg–60mg daily", "Inflammation", "Cushing's, osteoporosis"),
    ("levothyroxine", "25–200mcg daily", "Hypothyroidism", "AF if over-replaced"),
    ("amoxicillin", "250mg–1g TDS", "Bacterial infection", "Allergy, diarrhoea"),
    ("co-amoxiclav", "625mg TDS", "Polymicrobial infection", "Cholestatic jaundice"),
    ("azithromycin", "500mg OD 3d", "LRTI/CAP", "QT prolongation"),
    ("doxycycline", "100mg BD", "Atypicals/malaria", "Photosensitivity"),
    ("metronidazole", "400mg TDS", "Anaerobes/C.diff", "Metallic taste, alcohol CI"),
    ("furosemide", "20mg–500mg daily", "Fluid overload", "Electrolyte imbalance"),
    ("spironolactone", "25mg–100mg daily", "HF/ascites", "Hyperkalaemia, gynaecomastia"),
    ("digoxin", "62.5–250mcg daily", "AF rate control", "Narrow TI; toxicity risk"),
    ("carvedilol", "3.125mg–25mg BD", "HF", "Hypotension, dizziness"),
    ("ramipril", "2.5mg–10mg daily", "HF/HTN/CKD", "Cough, renal impairment"),
    ("eplerenone", "25mg–50mg daily", "Post-MI HF", "Hyperkalaemia"),
    ("empagliflozin", "10mg–25mg daily", "T2DM/HF", "UTI, DKA (rare)"),
    ("semaglutide", "0.25mg–2mg weekly", "T2DM/obesity", "Nausea, pancreatitis"),
    ("allopurinol", "100mg–900mg daily", "Gout", "Start low; SJS rare"),
    ("colchicine", "500mcg BD", "Acute gout", "GI toxicity, myopathy"),
    ("hydroxychloroquine", "200mg–400mg daily", "RA/SLE", "Retinopathy screening"),
    ("methotrexate", "7.5mg–25mg weekly", "RA/psoriasis", "Hepatotoxicity, folate"),
    ("adalimumab", "40mg fortnightly", "RA/IBD", "TB reactivation; screen first"),
    ("infliximab", "3–10mg/kg IV", "IBD/RA", "Infusion reaction; TB screen"),
]
for i, (drug, dose, indication, adr) in enumerate(medications):
    write(f"med_{i+1:02d}_{drug}.txt",
          f"DRUG: {drug.upper()}\nDose: {dose}\nIndication: {indication}\n"
          f"Adverse effects: {adr}\n"
          f"Category: Prescription Medicine\n"
          f"Review date: 2024-01\n")

print(f"\nC0 corpus built: {len(list(OUT.iterdir()))} files in {OUT}")