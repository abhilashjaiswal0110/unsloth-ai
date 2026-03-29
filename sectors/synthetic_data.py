"""
Synthetic Data Generator for Sector-Specific RLHF + GRPO Training
=================================================================
Generates high-quality QA pairs for healthcare, public utility, and insurance
domains. Each example follows prompts + reference answer format required by
GRPO training with multi-objective reward functions.

Output format per example:
  {
    "prompt": "<question>",
    "answer": "<reference answer>",
    "sector": "healthcare|public_utility|insurance",
    "category": "<sub-category>",
    "difficulty": "easy|medium|hard",
    "requires_reasoning": true|false
  }
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Healthcare Domain Data
# ─────────────────────────────────────────────────────────────────────────────

HEALTHCARE_QA: list[dict[str, Any]] = [
    # ── Clinical Reasoning ────────────────────────────────────────────────────
    {
        "prompt": "A 58-year-old male presents with sudden onset chest pain radiating to the left arm, diaphoresis, and shortness of breath. His ECG shows ST-elevation in leads II, III, and aVF. What is the most likely diagnosis, and what is the immediate management plan?",
        "answer": "The most likely diagnosis is an acute inferior ST-elevation myocardial infarction (STEMI). Immediate management includes: 1) Activate cardiac catheterization lab for primary PCI within 90 minutes. 2) Administer aspirin 325mg chewed immediately. 3) Administer heparin (unfractionated or LMWH). 4) P2Y12 inhibitor (clopidogrel, ticagrelor, or prasugrel). 5) Nitroglycerin for pain if BP permits (systolic > 90mmHg). 6) Morphine for refractory pain. 7) Beta-blocker if no contraindications. Monitor for complications including right ventricular infarction, bradycardia, and heart block.",
        "category": "clinical_reasoning",
        "difficulty": "hard",
        "requires_reasoning": True,
    },
    {
        "prompt": "A patient is prescribed metformin 500mg twice daily for newly diagnosed Type 2 diabetes. Their current eGFR is 28 mL/min/1.73m². Is this prescription appropriate? Explain your reasoning.",
        "answer": "No, this prescription is not appropriate. Metformin is contraindicated when eGFR is below 30 mL/min/1.73m² due to the risk of lactic acidosis. With an eGFR of 28, the kidneys cannot adequately clear metformin, leading to drug accumulation. The prescriber should: 1) Discontinue metformin. 2) Consider alternative agents such as DPP-4 inhibitors (with dose adjustment), GLP-1 receptor agonists, or insulin. 3) Consult nephrology for management of diabetes in the setting of Stage 4 CKD. 4) Monitor renal function every 3 months.",
        "category": "pharmacology",
        "difficulty": "medium",
        "requires_reasoning": True,
    },
    {
        "prompt": "What are the key differences between Type 1 and Type 2 diabetes mellitus in terms of pathophysiology, typical age of onset, and treatment approach?",
        "answer": "Type 1 Diabetes: Autoimmune destruction of pancreatic beta cells leading to absolute insulin deficiency. Typically presents in childhood/adolescence (though can occur at any age). Treatment requires exogenous insulin therapy, carbohydrate counting, and continuous glucose monitoring. Type 2 Diabetes: Characterized by insulin resistance and progressive beta-cell dysfunction. Typically presents in adults over 40 (increasingly seen in younger populations due to obesity). Treatment starts with lifestyle modifications (diet, exercise, weight loss), progresses to oral medications (metformin first-line), then injectable therapies including GLP-1 agonists and insulin if needed. Key distinguishing features: Type 1 patients are typically lean with acute onset and ketosis-prone; Type 2 patients often have metabolic syndrome with gradual onset.",
        "category": "medical_knowledge",
        "difficulty": "easy",
        "requires_reasoning": False,
    },
    {
        "prompt": "A 72-year-old woman on warfarin for atrial fibrillation presents with INR of 8.5 and no active bleeding. What is the appropriate management?",
        "answer": "For an INR of 8.5 without active bleeding: 1) Hold warfarin immediately. 2) Administer oral Vitamin K 2.5-5mg to reverse anticoagulation. 3) Recheck INR in 6-12 hours. 4) Investigate cause of supratherapeutic INR (dietary changes, drug interactions — especially antibiotics, amiodarone, NSAIDs, or herbal supplements). 5) Monitor for signs of bleeding (hematuria, melena, ecchymosis, headache). 6) Resume warfarin at a reduced dose once INR is below 3.0. 7) If signs of bleeding develop, escalate to IV Vitamin K and consider fresh frozen plasma or 4-factor PCC. Do NOT give vitamin K IV unless there is life-threatening hemorrhage due to risk of anaphylaxis.",
        "category": "clinical_reasoning",
        "difficulty": "hard",
        "requires_reasoning": True,
    },
    {
        "prompt": "Calculate the BMI for a patient who weighs 95 kg and is 1.75 meters tall. Classify the result according to WHO categories.",
        "answer": "BMI = weight (kg) / height² (m²) = 95 / (1.75)² = 95 / 3.0625 = 31.02 kg/m². According to WHO classification: Underweight < 18.5, Normal 18.5-24.9, Overweight 25.0-29.9, Obese Class I 30.0-34.9, Obese Class II 35.0-39.9, Obese Class III ≥ 40.0. This patient has a BMI of 31.02, which falls into Obese Class I. Recommended interventions include dietary counseling, structured exercise program, behavioral therapy, and potentially pharmacotherapy if lifestyle changes are insufficient.",
        "category": "calculation",
        "difficulty": "easy",
        "requires_reasoning": True,
    },
    {
        "prompt": "A patient presents with a hemoglobin A1c of 9.2%. They are currently on metformin 1000mg twice daily. What additional therapy would you recommend and why?",
        "answer": "An A1c of 9.2% on maximum-dose metformin indicates inadequate glycemic control (target < 7% for most adults). Recommended second-line additions depend on patient factors: 1) SGLT2 inhibitor (empagliflozin, dapagliflozin) — preferred if patient has cardiovascular disease, heart failure, or CKD, provides additional cardiorenal benefits. 2) GLP-1 receptor agonist (semaglutide, liraglutide) — preferred if weight loss is a priority or patient has atherosclerotic cardiovascular disease. 3) If A1c > 9% with symptoms, consider initiating basal insulin (glargine or degludec) as the gap from target is > 2%. 4) DPP-4 inhibitor (sitagliptin) if cost is a barrier. Given the A1c is 2.2% above target, dual therapy or even triple therapy may be needed. Follow up in 3 months to reassess.",
        "category": "treatment_planning",
        "difficulty": "medium",
        "requires_reasoning": True,
    },
    {
        "prompt": "What are the ABCDE criteria for melanoma detection in dermatology?",
        "answer": "The ABCDE criteria for melanoma detection are: A — Asymmetry: One half of the mole does not match the other half. B — Border: Edges are irregular, ragged, notched, or blurred. C — Color: The color is not uniform; may include shades of brown, black, pink, red, white, or blue. D — Diameter: The mole is larger than 6mm (about the size of a pencil eraser), though melanomas can be smaller. E — Evolving: The mole is changing in size, shape, color, or has new symptoms such as bleeding, itching, or crusting. Any lesion meeting one or more criteria should be evaluated with dermoscopy and potentially excisional biopsy. The 'ugly duckling sign' (a mole that looks different from all others) is an additional screening tool.",
        "category": "medical_knowledge",
        "difficulty": "easy",
        "requires_reasoning": False,
    },
    {
        "prompt": "A 45-year-old patient with chronic kidney disease Stage 3b (eGFR 38) needs pain management for osteoarthritis. NSAIDs are contraindicated. Recommend a safe pain management strategy.",
        "answer": "For osteoarthritis pain in CKD Stage 3b (eGFR 38): 1) First-line: Acetaminophen (paracetamol) up to 2g/day (reduced from standard 4g due to renal impairment) — safest analgesic in CKD. 2) Topical agents: Topical NSAIDs (diclofenac gel) have minimal systemic absorption and may be used cautiously. Capsaicin cream for localized pain. 3) Non-pharmacological: Physical therapy, weight management, joint strengthening exercises, heat/cold therapy, assistive devices. 4) If inadequate: Tramadol at reduced dose (50mg Q12H, renally adjusted) with monitoring for CNS depression. 5) Consider duloxetine 30-60mg daily (hepatically cleared, safe in CKD). 6) Avoid: All oral NSAIDs (further nephrotoxicity), high-dose opioids (accumulation of active metabolites). 7) Intra-articular corticosteroid injections for acute flares. 8) Referral to pain management or rheumatology if refractory.",
        "category": "treatment_planning",
        "difficulty": "hard",
        "requires_reasoning": True,
    },
    {
        "prompt": "Explain the mechanism of action of ACE inhibitors and list three common side effects.",
        "answer": "ACE inhibitors (e.g., lisinopril, enalapril, ramipril) work by blocking angiotensin-converting enzyme, which prevents the conversion of angiotensin I to angiotensin II. This results in: 1) Vasodilation (reduced peripheral vascular resistance). 2) Decreased aldosterone secretion (reducing sodium and water retention). 3) Reduced cardiac preload and afterload. 4) Decreased sympathetic nervous system activity. Three common side effects: 1) Dry cough (occurs in 5-20% of patients due to bradykinin accumulation in the lungs — bradykinin is normally degraded by ACE). 2) Hyperkalemia (due to decreased aldosterone, which normally promotes potassium excretion). 3) Hypotension, especially first-dose hypotension (more common in volume-depleted patients or those on diuretics). Additional serious but less common: angioedema (higher risk in African Americans), acute kidney injury (especially in bilateral renal artery stenosis).",
        "category": "pharmacology",
        "difficulty": "medium",
        "requires_reasoning": False,
    },
    {
        "prompt": "A hospital reports 12 central line-associated bloodstream infections (CLABSIs) over 3,000 central line days in a quarter. Calculate the CLABSI rate per 1,000 central line days and determine if this exceeds the NHSN benchmark of 0.8.",
        "answer": "CLABSI rate = (Number of CLABSIs / Number of central line days) × 1,000 = (12 / 3,000) × 1,000 = 4.0 per 1,000 central line days. This rate of 4.0 significantly exceeds the NHSN benchmark of 0.8 per 1,000 central line days — it is 5 times the acceptable rate. Recommended actions: 1) Immediate root cause analysis of all 12 infections. 2) Audit central line insertion and maintenance bundle compliance. 3) Review hand hygiene compliance rates. 4) Assess chlorhexidine bathing protocol adherence. 5) Evaluate nurse-to-patient staffing ratios. 6) Consider implementing daily 'line necessity' reviews. 7) Report to hospital infection control committee. 8) Staff re-education on aseptic technique. The Standardized Infection Ratio (SIR) would be 4.0/0.8 = 5.0, indicating performance far below national standards.",
        "category": "calculation",
        "difficulty": "medium",
        "requires_reasoning": True,
    },
    {
        "prompt": "A 30-year-old pregnant woman at 28 weeks gestation presents with blood pressure 160/105 mmHg, proteinuria of 3+, and headache. What is the diagnosis and immediate management?",
        "answer": "Diagnosis: Preeclampsia with severe features (BP ≥ 160/110 with proteinuria and symptoms). Immediate management: 1) IV magnesium sulfate 4g loading dose over 20 minutes, then 1g/hr maintenance for seizure prophylaxis. 2) Antihypertensive: IV labetalol 20mg bolus (first-line) or IV hydralazine 5mg, or oral nifedipine 10mg. Target BP < 150/100. 3) Continuous fetal monitoring (CTG). 4) Labs: CBC, LFTs, creatinine, LDH, uric acid, coagulation panel — to assess for HELLP syndrome. 5) Administer betamethasone 12mg IM × 2 doses 24 hours apart for fetal lung maturity (< 34 weeks). 6) Strict fluid balance monitoring (risk of pulmonary edema). 7) MFM (maternal fetal medicine) consultation. 8) Plan for delivery: if ≥ 37 weeks, deliver. At 28 weeks, expectant management may be attempted if condition stabilizes, with delivery recommended if worsening. 9) Monitor for eclampsia, HELLP, placental abruption.",
        "category": "clinical_reasoning",
        "difficulty": "hard",
        "requires_reasoning": True,
    },
    {
        "prompt": "What is the recommended immunization schedule for a healthy infant from birth to 6 months in the United States?",
        "answer": "CDC recommended immunization schedule for birth to 6 months: Birth: Hepatitis B (HepB) — 1st dose. 1 month: Hepatitis B (HepB) — 2nd dose. 2 months: DTaP (diphtheria, tetanus, pertussis) — 1st dose; IPV (polio) — 1st dose; Hib (Haemophilus influenzae type b) — 1st dose; PCV13 (pneumococcal) — 1st dose; Rotavirus (RV) — 1st dose. 4 months: DTaP — 2nd dose; IPV — 2nd dose; Hib — 2nd dose; PCV13 — 2nd dose; Rotavirus — 2nd dose. 6 months: DTaP — 3rd dose; PCV13 — 3rd dose; Rotavirus — 3rd dose (if Rotateq); Influenza — 1st dose (annually from 6 months); Hepatitis B — 3rd dose (can be given 6-18 months). Note: COVID-19 vaccine recommendations may vary. All vaccinations should be documented in the immunization information system (IIS).",
        "category": "preventive_medicine",
        "difficulty": "medium",
        "requires_reasoning": False,
    },
    # Additional healthcare QA pairs for training robustness
    {
        "prompt": "A patient's lab results show: Sodium 128 mEq/L, serum osmolality 265 mOsm/kg, urine osmolality 450 mOsm/kg, urine sodium 45 mEq/L. The patient is euvolemic. What is the most likely diagnosis?",
        "answer": "The findings are consistent with Syndrome of Inappropriate Antidiuretic Hormone (SIADH). Diagnostic criteria met: 1) Hyponatremia (Na 128, normal 135-145). 2) Low serum osmolality (265, normal 275-295) — hypotonic hyponatremia. 3) Inappropriately concentrated urine (osmolality 450 > 100) despite dilute serum. 4) Elevated urine sodium (45 > 40) despite hyponatremia. 5) Euvolemic status (excludes hypovolemic and hypervolemic causes). Common causes to investigate: malignancy (especially small cell lung cancer), CNS disorders, pulmonary disease, medications (SSRIs, carbamazepine, cyclophosphamide). Management: 1) Fluid restriction (< 1L/day first-line). 2) Identify and treat underlying cause. 3) If severe or symptomatic: hypertonic saline (3% NaCl) with rate limited to avoid osmotic demyelination — correct no faster than 8 mEq/L per 24 hours.",
        "category": "clinical_reasoning",
        "difficulty": "hard",
        "requires_reasoning": True,
    },
    {
        "prompt": "What are the four stages of pressure injuries (pressure ulcers) according to the NPUAP classification?",
        "answer": "NPUAP Pressure Injury Staging: Stage 1: Non-blanchable erythema of intact skin. The area may be painful, firm, soft, warmer or cooler compared to adjacent tissue. Darkly pigmented skin may not show visible blanching. Stage 2: Partial-thickness skin loss with exposed dermis. The wound bed is viable, pink/red, and moist. May present as an intact or ruptured serum-filled blister. Adipose tissue and deeper tissues are NOT visible. Stage 3: Full-thickness skin loss. Adipose (fat) tissue is visible in the wound. Granulation tissue and rolled wound edges may be present. Slough and/or eschar may be present. Depth varies by anatomical location. Fascia, muscle, tendon, bone are NOT exposed. Stage 4: Full-thickness skin and tissue loss with exposed or palpable fascia, muscle, tendon, ligament, cartilage, or bone. Slough and/or eschar may be present. Undermining and tunneling often occur. Additional categories: Unstageable (obscured by slough/eschar) and Deep Tissue Pressure Injury (persistent non-blanchable deep red/maroon/purple discoloration).",
        "category": "medical_knowledge",
        "difficulty": "medium",
        "requires_reasoning": False,
    },
    {
        "prompt": "A 65-year-old male with COPD has the following ABG results: pH 7.32, PaCO2 58 mmHg, HCO3 32 mEq/L, PaO2 55 mmHg. Interpret these results.",
        "answer": "ABG Interpretation: pH 7.32 (acidotic, normal 7.35-7.45). PaCO2 58 mmHg (elevated, normal 35-45) — indicates respiratory acidosis. HCO3 32 mEq/L (elevated, normal 22-26) — indicates metabolic compensation. PaO2 55 mmHg (hypoxemic, normal 80-100). Diagnosis: Partially compensated respiratory acidosis with hypoxemia. The elevated CO2 indicates CO2 retention (hypoventilation) typical of COPD. The elevated bicarbonate represents renal compensation (kidneys retaining HCO3 to buffer the acidosis). Since pH remains below normal, compensation is partial, not complete. Expected compensation: For chronic respiratory acidosis, HCO3 should rise 3.5 mEq/L per 10 mmHg rise in CO2. Expected HCO3 = 24 + 3.5 × (58-40)/10 = 24 + 6.3 = 30.3. Actual HCO3 of 32 is close, confirming chronic process. Management: Controlled oxygen therapy (target SpO2 88-92% to avoid suppressing hypoxic drive), bronchodilators, consider NIV (BiPAP) if worsening.",
        "category": "clinical_reasoning",
        "difficulty": "hard",
        "requires_reasoning": True,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Public Utility Domain Data
# ─────────────────────────────────────────────────────────────────────────────

PUBLIC_UTILITY_QA: list[dict[str, Any]] = [
    {
        "prompt": "A residential customer's monthly electricity bill shows 850 kWh consumed at a tiered rate: first 500 kWh at $0.12/kWh, next 300 kWh at $0.15/kWh, and anything above 800 kWh at $0.20/kWh. Calculate the total energy charge before taxes and fees.",
        "answer": "Tiered rate calculation: Tier 1: 500 kWh × $0.12 = $60.00. Tier 2: 300 kWh × $0.15 = $45.00. Tier 3: (850 - 800) = 50 kWh × $0.20 = $10.00. Total energy charge = $60.00 + $45.00 + $10.00 = $115.00. Note: This does not include fixed service charges, demand charges, fuel cost adjustments, taxes, or regulatory fees which typically add 15-30% to the total bill. The tiered structure incentivizes conservation — the marginal cost increases as consumption rises.",
        "category": "billing_calculation",
        "difficulty": "easy",
        "requires_reasoning": True,
    },
    {
        "prompt": "A water utility detects a water main break affecting 2,500 households. The main is a 12-inch ductile iron pipe. Outline the emergency response protocol and customer communication plan.",
        "answer": "Emergency Response Protocol: Phase 1 — Immediate (0-2 hours): 1) Dispatch emergency repair crew and equipment. 2) Isolate the break by closing upstream and downstream valves. 3) Assess impact zone — identify affected households, critical facilities (hospitals, schools, fire stations). 4) Issue boil water advisory if contamination risk exists. 5) Activate emergency operations center if break qualifies as major incident (>1,000 customers). Phase 2 — Repair (2-12 hours): 1) Excavate and assess damage extent. 2) Obtain or verify pipe specifications (12-inch DI). 3) Install repair clamp or replace pipe section. 4) Coordinate with traffic management for road closures. 5) Arrange alternative water supply (tanker trucks) if repair exceeds 8 hours. Phase 3 — Restoration: 1) Slowly repressurize the main to avoid further breaks. 2) Flush the system and collect bacteriological samples. 3) Maintain boil water advisory until lab results confirm safety (24-48 hours). Customer Communication: 1) Automated IVR/SMS blast within 30 minutes of isolation. 2) Post updates on website and social media every 2 hours. 3) Door-to-door notification for vulnerable customers (elderly, medical equipment users). 4) Provide estimated restoration time. 5) Post-incident: bill credits for prolonged outages per tariff regulations.",
        "category": "emergency_response",
        "difficulty": "hard",
        "requires_reasoning": True,
    },
    {
        "prompt": "Explain the concept of net metering for residential solar customers and how it affects their utility bill.",
        "answer": "Net metering is a billing mechanism that credits solar energy system owners for the electricity they add to the grid. How it works: 1) When solar panels produce more electricity than the home consumes, excess power flows back to the grid through the utility meter. 2) The meter effectively 'spins backward,' crediting the customer at the retail electricity rate. 3) When solar production is insufficient (nighttime, cloudy days), the customer draws power from the grid normally. 4) At billing time, the customer pays only for 'net' consumption (grid power used minus solar power exported). Bill impact: If a customer consumes 900 kWh and exports 400 kWh, they are billed for 500 kWh net. If exports exceed consumption in a billing period, credits typically roll forward (usually for 12 months). Most utilities still charge a fixed monthly service/connection fee regardless of net consumption. Some jurisdictions have moved to 'net billing' or 'net metering 2.0' where export credits are at a lower wholesale rate rather than full retail rate, reducing the economic benefit.",
        "category": "policy_explanation",
        "difficulty": "medium",
        "requires_reasoning": False,
    },
    {
        "prompt": "A natural gas utility needs to calculate the annual revenue requirement for a rate case filing. The rate base is $450 million, the authorized ROE is 9.8%, the debt-to-equity ratio is 50/50, the cost of debt is 4.5%, depreciation is $35 million, O&M expenses are $120 million, and taxes are $28 million. Calculate the revenue requirement.",
        "answer": "Revenue Requirement Calculation: Step 1 — Weighted Average Cost of Capital (WACC): Equity component: 50% × 9.8% = 4.90%. Debt component: 50% × 4.5% = 2.25%. WACC = 4.90% + 2.25% = 7.15%. Step 2 — Return on Rate Base: Rate Base × WACC = $450M × 7.15% = $32.175 million. Step 3 — Total Revenue Requirement: Return on Rate Base: $32.175M. + Depreciation: $35.000M. + O&M Expenses: $120.000M. + Taxes: $28.000M. = Total Revenue Requirement: $215.175 million. This represents the minimum revenue the utility needs to collect through customer rates to cover all costs, maintain infrastructure, and provide shareholders their authorized return. The Public Utility Commission reviews this calculation during rate proceedings, and consumer advocates may challenge individual components (particularly the ROE and rate base composition).",
        "category": "regulatory_calculation",
        "difficulty": "hard",
        "requires_reasoning": True,
    },
    {
        "prompt": "What is the difference between a demand charge and an energy charge on a commercial electricity bill?",
        "answer": "Energy Charge: Measured in kilowatt-hours (kWh). Reflects the total amount of electricity consumed over the billing period. Calculated as: total kWh × rate per kWh. Represents the cost of generating and delivering the energy. Example: 10,000 kWh × $0.10/kWh = $1,000. Demand Charge: Measured in kilowatts (kW). Reflects the maximum rate of electricity use at any point during the billing period (typically the highest 15-minute average). Calculated as: peak demand (kW) × rate per kW. Represents the cost of having sufficient infrastructure capacity available. Example: 50 kW peak × $12/kW = $600. Key difference: Energy charges reflect HOW MUCH power you use total; demand charges reflect HOW FAST you use power at peak. A facility that uses power steadily has lower demand charges than one with sharp peaks. This is why demand response programs encourage customers to reduce peak usage (load shifting, thermal storage, staggering equipment start times).",
        "category": "billing_explanation",
        "difficulty": "medium",
        "requires_reasoning": False,
    },
    {
        "prompt": "A utility company is planning a smart meter rollout for 150,000 customers. The cost per meter (including installation) is $285. The expected annual savings per meter are: $18 in reduced meter reading costs, $12 in theft/loss reduction, $8 in improved outage management, and $5 in customer service savings. Calculate the payback period and 10-year NPV at a discount rate of 6%.",
        "answer": "Per-Meter Calculations: Total cost per meter: $285. Annual savings per meter: $18 + $12 + $8 + $5 = $43/year. Simple Payback Period: $285 / $43 = 6.63 years. NPV Calculation (per meter, 10-year horizon, 6% discount rate): NPV = -$285 + Σ[$43 / (1.06)^t] for t = 1 to 10. Using the annuity present value factor: PV factor = [1 - (1.06)^(-10)] / 0.06 = 7.3601. PV of savings = $43 × 7.3601 = $316.48. NPV per meter = $316.48 - $285.00 = $31.48. Program-Level (150,000 meters): Total investment: 150,000 × $285 = $42.75 million. Total 10-year NPV: 150,000 × $31.48 = $4.72 million. The project is financially viable (positive NPV) with a payback period under 7 years. Additional unquantified benefits include improved power quality monitoring, dynamic pricing enablement, and enhanced grid reliability data.",
        "category": "financial_analysis",
        "difficulty": "hard",
        "requires_reasoning": True,
    },
    {
        "prompt": "Explain the process a customer should follow to dispute a utility bill they believe is incorrect.",
        "answer": "Utility Bill Dispute Process: Step 1 — Self-Review: Check meter reading (compare with previous months for anomalies). Verify rate plan is correct. Look for estimated vs. actual reads. Check for billing period length changes. Step 2 — Contact Utility: Call customer service or visit online portal. Request detailed bill breakdown. Ask for a meter re-read (most utilities offer one free re-read per year). Request a meter accuracy test if suspected malfunction. Step 3 — Formal Dispute: Submit written dispute to the utility's billing department. Include account number, disputed amount, and supporting documentation. The utility must acknowledge within 5-10 business days (varies by jurisdiction). Payment of the undisputed portion is still required. Step 4 — Utility Investigation: The utility has 30-60 days to investigate and respond. If the meter tests within ±2% accuracy, the bill stands. If the meter is inaccurate, the utility must recalculate and issue credits. Step 5 — Regulatory Complaint: If unsatisfied with the utility's resolution, file a complaint with the state Public Utility Commission (PUC). The PUC investigates independently and can order corrective action. Step 6 — Formal Hearing: Request a formal hearing before the PUC if informal resolution fails. Right to legal representation. PUC decision is binding on the utility.",
        "category": "customer_service",
        "difficulty": "easy",
        "requires_reasoning": False,
    },
    {
        "prompt": "A municipal water treatment plant processes 20 million gallons per day (MGD). If the chlorine demand is 2.5 mg/L, the desired chlorine residual is 0.5 mg/L, and the chlorine solution concentration is 12.5%, calculate the daily chlorine feed rate in pounds per day.",
        "answer": "Chlorine Feed Rate Calculation: Step 1 — Required chlorine dose: Chlorine dose = Chlorine demand + Desired residual = 2.5 + 0.5 = 3.0 mg/L. Step 2 — Convert to pounds per day: Using the formula: lbs/day = Dose (mg/L) × Flow (MGD) × 8.34 lbs/gal. lbs/day = 3.0 × 20 × 8.34 = 500.4 lbs/day of pure chlorine needed. Step 3 — Account for solution concentration: Since the chlorine solution is 12.5% (0.125): Solution feed rate = 500.4 / 0.125 = 4,003.2 lbs/day of chlorine solution. This equals approximately 480 gallons/day of 12.5% sodium hypochlorite solution (at ~8.34 lbs/gal). The plant should maintain at least 3 days of chemical storage (1,440 gallons) and have backup disinfection capability per EPA requirements.",
        "category": "engineering_calculation",
        "difficulty": "hard",
        "requires_reasoning": True,
    },
    {
        "prompt": "What are the main causes of power outages and how do utilities prioritize restoration?",
        "answer": "Main Causes of Power Outages: 1) Weather events (storms, ice, wind, lightning) — accounts for ~70% of major outages. 2) Equipment failure (transformers, switchgear, conductors aging). 3) Tree contact with power lines. 4) Vehicle accidents damaging poles/equipment. 5) Animal contact with equipment (squirrels, birds). 6) Planned maintenance outages. 7) Grid overload during extreme demand. 8) Cyberattacks or physical security threats. Restoration Priority Framework: Priority 1: Public safety (downed wires, fire/police/EMS facilities). Priority 2: Critical infrastructure (hospitals, water treatment, emergency shelters, 911 centers). Priority 3: Transmission and substation repairs (restores power to largest number fastest). Priority 4: Main distribution feeders (major circuits serving thousands). Priority 5: Lateral lines and tap connections (smaller groups, neighborhoods). Priority 6: Individual service drops (single customers). This framework follows the principle of 'greatest good for greatest number' while ensuring life safety first. Utilities provide estimated restoration times (ETR) that are updated as crews assess damage.",
        "category": "operations",
        "difficulty": "medium",
        "requires_reasoning": False,
    },
    {
        "prompt": "A commercial customer has a peak demand of 250 kW and wants to install a battery energy storage system (BESS) to reduce demand charges. If the demand charge is $15/kW/month and the battery can shave 80 kW of peak demand, calculate the annual savings and simple payback for a $95,000 BESS installation.",
        "answer": "Demand Charge Savings Calculation: Monthly demand reduction: 80 kW. Monthly savings: 80 kW × $15/kW = $1,200/month. Annual savings: $1,200 × 12 = $14,400/year. Simple Payback Period: $95,000 / $14,400 = 6.6 years. Additional considerations: 1) Battery degradation: Capacity typically decreases 2-3% per year, so savings will diminish over time (year 10 savings may be ~$11,500). 2) Energy arbitrage: Additional savings possible by charging during off-peak ($0.06/kWh) and discharging during peak ($0.15/kWh). If cycling 200 kWh daily: $0.09 × 200 × 365 = $6,570/year additional savings. 3) With arbitrage, total annual savings = $14,400 + $6,570 = $20,970. Adjusted payback = $95,000 / $20,970 = 4.5 years. 4) Federal ITC (Investment Tax Credit) of 30% reduces net cost to $66,500, bringing payback to 3.2 years. 5) Useful life of lithium-ion BESS: 10-15 years. Net financial benefit over 10 years (undiscounted): approximately $143,700 in savings vs. $66,500 net cost.",
        "category": "financial_analysis",
        "difficulty": "hard",
        "requires_reasoning": True,
    },
    {
        "prompt": "Explain the role of SCADA systems in modern utility operations.",
        "answer": "SCADA (Supervisory Control and Data Acquisition) in Utility Operations: Core Functions: 1) Real-time monitoring of infrastructure — substations, generators, pipelines, treatment plants, and distribution networks. 2) Remote control — operators can open/close valves, switch breakers, adjust set points without dispatching field crews. 3) Data acquisition — continuous collection of measurements (voltage, pressure, flow, temperature) from thousands of sensors. 4) Alarm management — automated alerts when parameters exceed thresholds. 5) Historical data logging for trending, compliance reporting, and forensic analysis. Components: RTUs (Remote Terminal Units) or PLCs at field sites collect data and execute commands. Communication networks (fiber, radio, cellular) link field devices to control center. Master station software displays system status and enables operator control. Human-Machine Interface (HMI) provides visualization and interaction. Applications by sector: Electric — load management, fault detection, automatic reclosing, voltage regulation. Water — pressure management, leak detection, pump control, water quality monitoring. Gas — pressure regulation, leak detection, compressor control, flow management. Security considerations: SCADA networks must be isolated from public internet (air-gapped or firewalled), follow NERC CIP standards (electric), and undergo regular vulnerability assessments.",
        "category": "technology",
        "difficulty": "medium",
        "requires_reasoning": False,
    },
    {
        "prompt": "A residential customer's water meter registered 15,400 gallons this month versus an average of 6,200 gallons for the same period last year. List systematic troubleshooting steps to identify the cause.",
        "answer": "Systematic Water Consumption Troubleshooting: Step 1 — Meter Verification: Read meter at start, ensure no usage for 2 hours, re-read. If meter moved: active leak exists. If meter has leak indicator (triangle/star), check if spinning when all fixtures off. Step 2 — Toilet Leak Test: Place food coloring in each toilet tank. Wait 15-20 minutes without flushing. If color appears in bowl: flapper valve leak (most common cause — a leaking toilet can waste 200+ gallons/day = 6,000+/month). Step 3 — Check Visible Fixtures: Faucets (dripping), water heater (T&P valve discharge), water softener (stuck in regeneration cycle), humidifier, ice maker line. Step 4 — Underground/Hidden Leaks: Check water pressure (sudden drop may indicate broken line). Look for unexplained wet spots in yard, driveway, or basement. Listen for running water when all fixtures off. Check irrigation system (stuck valve or broken head). Step 5 — Seasonal Factors: New swimming pool (fill = 15,000-30,000 gallons). Increased irrigation due to weather. Additional occupants or guests. Step 6 — Meter Accuracy: If no leaks found, request meter test. Meters can 'over-register' due to air in lines after water main work. The 9,200-gallon increase (148% above average) most commonly indicates a toilet flapper leak or irrigation system malfunction.",
        "category": "troubleshooting",
        "difficulty": "medium",
        "requires_reasoning": True,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Insurance Domain Data
# ─────────────────────────────────────────────────────────────────────────────

INSURANCE_QA: list[dict[str, Any]] = [
    {
        "prompt": "A homeowner has a dwelling coverage of $350,000 with an 80% coinsurance clause. A fire causes $100,000 in damage. At the time of loss, the replacement cost of the home is $500,000. Calculate the insurance payout.",
        "answer": "Coinsurance Calculation: Step 1 — Determine required coverage: Required coverage = Replacement cost × Coinsurance percentage = $500,000 × 80% = $400,000. Step 2 — Calculate coinsurance ratio: Actual coverage / Required coverage = $350,000 / $400,000 = 0.875 (87.5%). Step 3 — Since the homeowner is underinsured (87.5% < 100%): Payout = (Actual coverage / Required coverage) × Loss = 0.875 × $100,000 = $87,500. Step 4 — Subtract deductible (assume standard $1,000): Net payout = $87,500 - $1,000 = $86,500. The homeowner bears the coinsurance penalty of $12,500 ($100,000 - $87,500) plus the deductible. To avoid this penalty, the homeowner should increase dwelling coverage to at least $400,000 (80% of $500,000). Best practice: insure at 100% replacement cost with an inflation guard endorsement.",
        "category": "claims_calculation",
        "difficulty": "hard",
        "requires_reasoning": True,
    },
    {
        "prompt": "Explain the difference between occurrence-based and claims-made professional liability insurance policies.",
        "answer": "Occurrence-Based Policy: Covers incidents that OCCUR during the policy period, regardless of when the claim is filed. If a policyholder had coverage from 2020-2023 and an incident occurred in 2022 but the claim is filed in 2025, the 2022 policy responds. No 'tail' coverage needed. Premiums are generally higher because the insurer's liability extends indefinitely. More common for general liability and property insurance. Claims-Made Policy: Covers claims that are REPORTED during the policy period, regardless of when the incident occurred (subject to retroactive date). Both the incident must occur after the retroactive date AND the claim must be filed during the policy period. If coverage lapses, unreported claims from past incidents are NOT covered. Requires 'tail' coverage (Extended Reporting Period/ERP) when switching carriers or retiring — can cost 150-300% of annual premium. Common for professional liability (E&O, D&O, medical malpractice). Key practical difference: Occurrence policies are 'set and forget.' Claims-made policies require careful management of retroactive dates and tail coverage to avoid coverage gaps.",
        "category": "policy_knowledge",
        "difficulty": "medium",
        "requires_reasoning": False,
    },
    {
        "prompt": "An auto insurance company must determine the premium for a 22-year-old male driver with a clean record in a metropolitan area. The base rate for the territory is $800. Rating factors: age/gender factor 1.45, clean record credit 0.90, urban territory factor 1.25, vehicle symbol factor (2023 sedan) 1.10, multi-policy discount 0.95. Calculate the final annual premium.",
        "answer": "Premium Rating Calculation: Base Rate: $800.00. Step 1 — Apply multiplicative rating factors: $800.00 × 1.45 (age/gender) = $1,160.00. × 0.90 (clean record credit) = $1,044.00. × 1.25 (urban territory) = $1,305.00. × 1.10 (vehicle symbol) = $1,435.50. × 0.95 (multi-policy discount) = $1,363.73. Final Annual Premium: $1,363.73 (rounded to $1,364). Breakdown of impact: Age/gender surcharge: +$360 (+45%) — young males are statistically highest risk. Clean record credit: -$116 (-10%) — no accidents or violations. Urban territory: +$261 (+25%) — higher traffic density, theft rates. Vehicle symbol: +$131 (+10%) — vehicle repair cost and theft attractiveness. Multi-policy discount: -$72 (-5%) — bundling home and auto. Note: In states with gender-neutral rating (CA, HI, MA, MT, NC, PA), the age/gender factor would be replaced with an age-only factor. Insurance scoring (credit-based) may also apply in most states.",
        "category": "premium_calculation",
        "difficulty": "medium",
        "requires_reasoning": True,
    },
    {
        "prompt": "A small business owner is deciding between a Business Owner's Policy (BOP) and separate commercial property and general liability policies. What are the advantages and limitations of a BOP?",
        "answer": "Business Owner's Policy (BOP) Advantages: 1) Cost savings — bundled pricing typically 15-30% less than separate policies. 2) Simplified administration — one policy, one renewal, one premium payment. 3) Built-in coverages often included at no extra cost: business income/extra expense coverage, equipment breakdown, off-premises property coverage, electronic data loss coverage. 4) Broad form property coverage (special cause of loss, which covers everything except specific exclusions). 5) Automatic inflation guard on property values. 6) Easier to understand for small business owners. BOP Limitations: 1) Eligibility restrictions — typically for businesses under $5M annual revenue and under certain square footage thresholds. 2) Not available for high-risk businesses (bars, contractors, manufacturers). 3) Coverage limits may be insufficient for larger operations. 4) Less customization — cannot tailor each coverage component independently. 5) Typically does not include: professional liability (E&O), workers' compensation, commercial auto, cyber liability, employment practices liability (EPLI). 6) Business income coverage may have sub-limits that are inadequate. Recommendation: BOP is ideal for small, low-risk businesses (offices, retail, restaurants under certain sizes). Businesses with complex operations, large property values, or high liability exposure should consider separate, tailored policies.",
        "category": "product_knowledge",
        "difficulty": "medium",
        "requires_reasoning": False,
    },
    {
        "prompt": "An insurance company's loss ratio is 68%, expense ratio is 27%, and investment income ratio is 8%. Calculate the combined ratio and operating ratio. Is this company profitable from underwriting operations?",
        "answer": "Insurance Profitability Ratios: Combined Ratio = Loss Ratio + Expense Ratio = 68% + 27% = 95%. Operating Ratio = Combined Ratio - Investment Income Ratio = 95% - 8% = 87%. Analysis: Combined Ratio of 95%: The company pays out $0.95 in losses and expenses for every $1.00 of premium earned. Since the combined ratio is below 100%, the company IS profitable from underwriting operations alone (underwriting profit of 5 cents per premium dollar). Operating Ratio of 87%: When investment income is included, the company's total operations generate a 13% profit margin. This is a healthy result. Industry context: Average P&C industry combined ratio is ~98-100%. A combined ratio below 95% is considered excellent. Companies can survive above 100% combined ratio if investment income compensates (common in long-tail lines like workers' comp). Key metric relationships: Loss ratio (68%) is below industry average (~70%), suggesting good risk selection and claims management. Expense ratio (27%) is typical for a standard carrier (direct writers may achieve 20-22%).",
        "category": "financial_analysis",
        "difficulty": "hard",
        "requires_reasoning": True,
    },
    {
        "prompt": "What is subrogation in insurance and provide an example of how it works in practice?",
        "answer": "Subrogation is the legal right of an insurance company to pursue a third party that caused a loss to the insured, after the insurer has paid the claim. It allows the insurer to recover the amount paid. Practical Example: 1) Driver A (insured by Company X) is rear-ended by Driver B (insured by Company Y) at a stoplight. 2) Driver A files a claim with Company X for $15,000 in vehicle damage and $5,000 in medical expenses. 3) Company X pays Driver A the $20,000 claim (minus deductible of $500), so Driver A receives $19,500. 4) Company X then exercises its subrogation rights against Driver B (and Company Y). 5) Company X sends a subrogation demand to Company Y for $20,000 (the full amount paid including deductible held from insured). 6) If Company Y accepts liability, they reimburse Company X. 7) Company X then returns Driver A's $500 deductible. Key principles: Insured must be 'made whole' before insurer recovers. Insured cannot accept money from the third party that would prejudice subrogation rights. Subrogation prevents double recovery and places the financial burden on the at-fault party. Common in auto, property (fire caused by contractor), workers' compensation (workplace injury caused by third party), and health insurance (medical bills from car accident).",
        "category": "legal_concepts",
        "difficulty": "easy",
        "requires_reasoning": False,
    },
    {
        "prompt": "A life insurance applicant is a 40-year-old non-smoking male seeking a $1,000,000 20-year level term policy. The base annual premium is $850. Underwriting reveals he has well-controlled hypertension (on medication, BP 128/82) and BMI of 29. He qualifies for Standard Plus (not Preferred) rating with a table rating adjustment of +25%. Calculate his annual and monthly premiums.",
        "answer": "Life Insurance Premium Calculation: Base annual premium (Preferred rate): $850.00. Table rating adjustment: +25% of base. Substandard premium = $850.00 × 0.25 = $212.50. Rated Annual Premium: $850.00 + $212.50 = $1,062.50. Monthly premium (with modal factor of 0.0875 — typical monthly factor): $1,062.50 × 0.0875 = $92.97/month. Note: Monthly payments total $92.97 × 12 = $1,115.64/year, which is 5% more than annual payment due to the modal loading factor. Underwriting rationale: Well-controlled hypertension on medication with BP 128/82 moves from Preferred to Standard Plus (one class down). BMI of 29 is just under the obesity threshold (30) but above Preferred guidelines (typically <27). Combined factors result in a +25% table rating (Table B). Recommendations for applicant: 1) Request re-underwriting in 2-3 years if BP and BMI improve. 2) Consider a policy with a conversion option to whole life. 3) Compare with carriers that have more favorable guidelines for controlled hypertension.",
        "category": "underwriting",
        "difficulty": "hard",
        "requires_reasoning": True,
    },
    {
        "prompt": "What are the key differences between HMO, PPO, EPO, and POS health insurance plans?",
        "answer": "Health Insurance Plan Comparison: HMO (Health Maintenance Organization): Requires primary care physician (PCP) selection. Referral needed for specialists. Network-only coverage (no out-of-network benefits except emergencies). Lowest premiums and copays. Focus on preventive care. Example: Kaiser Permanente. PPO (Preferred Provider Organization): No PCP required. No referrals needed for specialists. Coverage both in-network and out-of-network (higher cost for out-of-network). Higher premiums than HMO. Maximum flexibility in provider choice. Most popular plan type. EPO (Exclusive Provider Organization): No PCP required. No referrals needed. Network-only coverage (similar to HMO restriction). Premiums between HMO and PPO. Combines PPO convenience with HMO cost control. POS (Point of Service): Requires PCP selection (like HMO). Referral needed for in-network specialists. DOES cover out-of-network (like PPO) but at higher cost. Moderate premiums. Hybrid between HMO and PPO. Summary matrix: Flexibility: PPO > POS > EPO > HMO. Cost: HMO < EPO < POS < PPO. Out-of-network: PPO, POS (yes) | HMO, EPO (no). PCP required: HMO, POS (yes) | PPO, EPO (no). Referrals: HMO, POS (yes) | PPO, EPO (no).",
        "category": "product_knowledge",
        "difficulty": "easy",
        "requires_reasoning": False,
    },
    {
        "prompt": "An insurance company has the following claims data for auto liability: 500 policies, 45 claims filed, total incurred losses $892,000, allocated loss adjustment expenses (ALAE) $78,000, and unallocated loss adjustment expenses (ULAE) $45,000. Calculate the frequency, severity, pure premium, and loss adjustment expense ratio.",
        "answer": "Actuarial Metrics Calculation: 1) Claim Frequency = Number of claims / Number of policies = 45 / 500 = 0.09 or 9.0%. Interpretation: 9 out of every 100 policies generate a claim. 2) Claim Severity = Total incurred losses / Number of claims = $892,000 / 45 = $19,822.22 per claim. 3) Pure Premium = Frequency × Severity = 0.09 × $19,822.22 = $1,784.00 per policy. Alternatively: Total losses / Number of policies = $892,000 / 500 = $1,784.00. 4) Total LAE = ALAE + ULAE = $78,000 + $45,000 = $123,000. 5) Loss Adjustment Expense Ratio = Total LAE / Total Incurred Losses = $123,000 / $892,000 = 13.79%. 6) Loaded Pure Premium (including LAE) = Pure Premium × (1 + LAE Ratio) = $1,784.00 × 1.1379 = $2,029.83. To arrive at the gross premium, the company would add expense loading (commissions, overhead, profit margin). If expense loading is 30% and target profit is 5%: Gross premium = $2,029.83 / (1 - 0.30 - 0.05) = $2,029.83 / 0.65 = $3,122.82 per policy.",
        "category": "actuarial",
        "difficulty": "hard",
        "requires_reasoning": True,
    },
    {
        "prompt": "A policyholder's commercial property policy has a $5 million building limit, $2 million contents limit, and a 72-hour waiting period on business income coverage with a 12-month period of indemnity. A fire destroys 40% of the building and 60% of the contents. Business income loss is $50,000/month for 8 months. Calculate the total claim payment assuming a $10,000 deductible.",
        "answer": "Total Claim Calculation: 1) Building Damage: $5,000,000 × 40% = $2,000,000. 2) Contents Damage: $2,000,000 × 60% = $1,200,000. 3) Business Income: $50,000/month × 8 months = $400,000. The 72-hour waiting period applies to the first 3 days of lost income. Assuming 30-day months: Daily income loss = $50,000 / 30 = $1,666.67. Waiting period deduction = $1,666.67 × 3 = $5,000. Adjusted business income = $400,000 - $5,000 = $395,000. The 12-month period of indemnity covers the full 8 months. 4) Subtotal before deductible: Building: $2,000,000. Contents: $1,200,000. Business Income: $395,000. Subtotal: $3,595,000. 5) Deductible: $10,000 (applied once per occurrence). 6) Total Claim Payment: $3,595,000 - $10,000 = $3,585,000. Note: This assumes no coinsurance penalty (building and contents are insured to at least 80% of value), no coverage exclusions apply, and replacement cost valuation (not ACV). Extra expense coverage may provide additional amounts for temporary relocation costs.",
        "category": "claims_calculation",
        "difficulty": "hard",
        "requires_reasoning": True,
    },
    {
        "prompt": "Explain the principle of indemnity in insurance and its exceptions.",
        "answer": "Principle of Indemnity: The fundamental insurance principle that a policyholder should be restored to the same financial position they were in immediately before the loss — no better and no worse. The insured should not profit from a loss. How it applies: 1) Property insurance: Pays the actual cash value (replacement cost minus depreciation) or replacement cost (if endorsed). 2) Liability insurance: Pays the amount the insured is legally obligated to pay (up to policy limits). 3) Prevents moral hazard — if people could profit from losses, they might cause them intentionally. Exceptions to Indemnity: 1) Life insurance: Pays a fixed sum regardless of actual financial loss. A $1M policy pays $1M whether the insured's economic value was higher or lower. 2) Valued policies: Some property policies (fine art, antiques) agree on value at inception and pay that amount regardless of actual loss value. 3) Replacement cost coverage: Pays full replacement cost without deducting depreciation, technically exceeding the 'same position' standard. 4) Agreed amount endorsement: Suspends coinsurance and pays the agreed value. 5) Pair and set clause: May pay more than proportional value when one item of a matched pair/set is lost. 6) New for old coverage: Replaces damaged old items with new ones (exceeds strict indemnity). Related principles: Insurable interest (must have financial stake), utmost good faith (honest disclosure), contribution (multiple policies share loss), proximate cause (direct link between peril and loss).",
        "category": "legal_concepts",
        "difficulty": "medium",
        "requires_reasoning": False,
    },
    {
        "prompt": "A reinsurance treaty has the following structure: quota share of 30% on the first $2M, and excess of loss cover of $3M xs $2M. The cedent retains 70% of the first layer. A $4.5M loss occurs. How is the loss distributed between the cedent and reinsurers?",
        "answer": "Reinsurance Loss Distribution for $4.5M Loss: Layer 1 — Quota Share (first $2M): Cedent's retention (70%): $2,000,000 × 70% = $1,400,000. Quota Share reinsurer (30%): $2,000,000 × 30% = $600,000. Layer 2 — Excess of Loss ($3M xs $2M): The excess amount above $2M = $4,500,000 - $2,000,000 = $2,500,000. This falls within the $3M xs $2M layer (capacity $3M, attachment $2M). XOL reinsurer pays: $2,500,000 (full amount, within $3M limit). Summary: Cedent pays: $1,400,000 (29.6% of total loss). Quota Share reinsurer pays: $600,000 (12.7% of total loss). XOL reinsurer pays: $2,500,000 (55.6% of total loss). Verification: $1,400,000 + $600,000 + $2,500,000 = $4,500,000. If the loss were $5.5M (exceeding the XOL layer): XOL reinsurer would pay max $3M. Remaining $500,000 ($5.5M - $2M - $3M) would fall back to the cedent's net retention. Cedent total: $1,400,000 + $500,000 = $1,900,000.",
        "category": "reinsurance",
        "difficulty": "hard",
        "requires_reasoning": True,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Generation and Saving
# ─────────────────────────────────────────────────────────────────────────────

ALL_SECTORS = {
    "healthcare": HEALTHCARE_QA,
    "public_utility": PUBLIC_UTILITY_QA,
    "insurance": INSURANCE_QA,
}


def _add_metadata(qa_list: list[dict], sector: str) -> list[dict]:
    """Ensure sector field is set on all entries."""
    for item in qa_list:
        item["sector"] = sector
    return qa_list


def generate_sector_dataset(
    sector: str,
    output_dir: str | Path = "data/sectors",
) -> Path:
    """Generate and save a single sector's dataset as JSONL."""
    if sector not in ALL_SECTORS:
        raise ValueError(
            f"Unknown sector '{sector}'. Choose from: {list(ALL_SECTORS.keys())}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_list = _add_metadata(ALL_SECTORS[sector], sector)

    # Shuffle for training variety
    random.seed(42)
    qa_list = list(qa_list)  # copy
    random.shuffle(qa_list)

    # Split 85% train / 15% eval
    split_idx = max(1, int(len(qa_list) * 0.85))
    train_data = qa_list[:split_idx]
    eval_data = qa_list[split_idx:]

    train_path = output_dir / f"{sector}_train.jsonl"
    eval_path = output_dir / f"{sector}_eval.jsonl"

    for path, data in [(train_path, train_data), (eval_path, eval_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        log.info("Wrote %d examples to %s", len(data), path)

    return output_dir


def generate_all_datasets(output_dir: str | Path = "data/sectors") -> Path:
    """Generate datasets for all sectors."""
    output_dir = Path(output_dir)
    for sector in ALL_SECTORS:
        generate_sector_dataset(sector, output_dir)
    log.info("All sector datasets generated in %s", output_dir)
    return output_dir


def load_sector_dataset(
    sector: str, split: str = "train", data_dir: str | Path = "data/sectors"
):
    """Load a sector dataset as a HuggingFace Dataset."""
    from datasets import Dataset

    data_dir = Path(data_dir)
    path = data_dir / f"{sector}_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run generate_sector_dataset('{sector}') first."
        )

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return Dataset.from_list(records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_all_datasets()
    print("Done! Datasets generated in data/sectors/")
