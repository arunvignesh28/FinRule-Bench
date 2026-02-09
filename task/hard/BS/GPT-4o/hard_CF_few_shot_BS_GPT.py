#!/usr/bin/env python3
"""
hard_few_shot_counterfactual_BS_gpt.py

Few-shot with IMPROVED causal and counterfactual explanations for hard multi-error detection.
Includes error analysis for ablation studies (matching few-shot implementation).

Inputs:
 - BS-MIXED-100.md            (mixed dataset: 20% clean, 80% anomaly)
 - BS-MIXED-100-truth.json    (ground-truth JSON)

Outputs:
 - few_shot_cf_eval_results.csv
 - few_shot_cf_eval_summary.log
 - few_shot_cf_eval_metrics.json
 - few_shot_cf_eval_error_analysis.json      (error type classification for ablation)
 - few_shot_cf_eval_error_cases.csv          (detailed error cases)

Model: gpt-4o with token & cost tracking enabled
"""

import json
import re
import time
import csv
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from openai import OpenAI

# ------------------- Configuration -------------------
INPUT_MD = Path("BS-MIXED-100.md")
INPUT_TRUTH = Path("BS-MIXED-100-truth.json")

CSV_OUT = Path("few_shot_cf_eval_results.csv")
LOG_FILE = Path("few_shot_cf_eval_summary.log")
METRICS_OUT = Path("few_shot_cf_eval_metrics.json")
ERROR_ANALYSIS_OUT = Path("few_shot_cf_eval_error_analysis.json")
ERROR_CASES_OUT = Path("few_shot_cf_eval_error_cases.csv")

MODEL = "gpt-4o"
TEMPERATURE = 0.0
DELAY = 1.0
MAX_RETRIES = 4

# Pricing for gpt-4o (in USD per million tokens)
PRICE_INPUT = 2.50    # $2.50 per 1M input tokens
PRICE_OUTPUT = 10.00  # $10.00 per 1M output tokens

VALID_CODES = {"BS01", "BS02", "BS03", "BS04", "BS05"}

client = OpenAI()

# ------------------- Few-Shot Examples with Improved Causal & Counterfactual Reasoning -------------------

FEW_SHOT_COUNTERFACTUAL = """
EXAMPLES WITH DISCRIMINATIVE CAUSAL REASONING:

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 1 - CLEAN BASELINE (NO violations)
═══════════════════════════════════════════════════════════════════════════════

Section | Subsection | Item | Value
Assets | Current_Assets | Cash and cash equivalents | 5000
Assets | Current_Assets | Accounts receivable | 3000
Assets | Non_Current_Assets | Property and equipment | 10000
Assets |  | Total Assets | 18000
Liabilities | Current_Liabilities | Accounts payable | 2000
Liabilities | Non_Current_Liabilities | Long-term debt | 6000
Liabilities |  | Total Liabilities | 8000
Equity | Stockholders_Equity | Retained earnings | 10000
Equity |  | Total Equity | 10000

Analysis: NO

WHY NO VIOLATIONS:
BS01: 18000 = 8000 + 10000 ✓ (verify arithmetic first)
BS02: Current_Assets + Non_Current_Assets present ✓ (check subsections exist)
BS03: "Cash and cash equivalents" exact match ✓ (no synonyms)
BS04: "Retained earnings" exact match ✓ (not "accumulated" or "surplus")
BS05: No treasury stock in assets ✓ (absence is compliant)

CRITICAL COUNTERFACTUAL (train discrimination):
→ If Total Assets were 18001 instead of 18000?
  RESULT: BS01 violation (18001 ≠ 18000)
  Current state: 18000 = 18000, so NO violation
→ If subsections said "Short_Term" instead of "Current_Assets"?
  RESULT: BS02 violation (wrong terminology)
  Current state: Uses exact "Current_Assets", so NO violation

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 2 - BS01 ONLY (equation imbalance)
═══════════════════════════════════════════════════════════════════════════════

Section | Subsection | Item | Value
Assets | Current_Assets | Cash and cash equivalents | 5000
Assets | Current_Assets | Accounts receivable | 3000
Assets | Non_Current_Assets | Property and equipment | 10000
Assets |  | Total Assets | 18500
Liabilities | Current_Liabilities | Accounts payable | 2000
Liabilities | Non_Current_Liabilities | Long-term debt | 6000
Liabilities |  | Total Liabilities | 8000
Equity | Stockholders_Equity | Retained earnings | 10000
Equity |  | Total Equity | 10000

Analysis: YES: [BS01]

WHY BS01 VIOLATED:
Arithmetic: 18500 ≠ (8000 + 10000)
Root cause: Total Assets miscalculated or mistyped
Consequence: Entire statement unreliable

WHY OTHER RULES NOT VIOLATED:
BS02: Current_Assets/Non_Current_Assets subsections present ✓
BS03: "Cash and cash equivalents" exact terminology ✓
BS04: "Retained earnings" exact label ✓
BS05: No treasury stock ✓

DISCRIMINATIVE TRAINING:
→ Common mistake: Flagging BS04 because "numbers don't add up"
  WRONG: BS04 is about terminology, not arithmetic
  Correct reasoning: Check equity label first, it says "Retained earnings" → BS04 passes
→ Common mistake: Assuming classification is wrong when equation fails
  WRONG: BS02 checks subsection labels, not totals
  Correct reasoning: Subsections clearly labeled → BS02 passes

COUNTERFACTUAL FIX:
Change Total Assets to 18000 → BS01 violation disappears
All other rules remain compliant

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 3 - BS02 ONLY (missing classification)
═══════════════════════════════════════════════════════════════════════════════

Section | Subsection | Item | Value
Assets |  | Cash and cash equivalents | 5000
Assets |  | Accounts receivable | 3000
Assets | Non_Current_Assets | Property and equipment | 10000
Assets |  | Total Assets | 18000
Liabilities | Current_Liabilities | Accounts payable | 2000
Liabilities | Non_Current_Liabilities | Long-term debt | 6000
Liabilities |  | Total Liabilities | 8000
Equity | Stockholders_Equity | Retained earnings | 10000
Equity |  | Total Equity | 10000

Analysis: YES: [BS02]

WHY BS02 VIOLATED:
First two asset items have BLANK subsections (should be "Current_Assets")
Root cause: Classification missing for current assets
Consequence: Cannot compute working capital

WHY OTHER RULES NOT VIOLATED:
BS01: 18000 = 8000 + 10000 ✓ (equation balanced)
BS03: "Cash and cash equivalents" correct ✓
BS04: "Retained earnings" correct ✓
BS05: No treasury stock ✓

DISCRIMINATIVE TRAINING:
→ Common mistake: Thinking BS01 is violated because rows lack subsections
  WRONG: BS01 only checks if totals balance (18000 = 18000)
  Correct reasoning: Verify arithmetic independent of subsection labels
→ Common mistake: Thinking partial subsections are acceptable
  WRONG: BS02 requires ALL items classified (except totals)
  Correct reasoning: ANY missing subsection = BS02 violation

COUNTERFACTUAL FIX:
Add "Current_Assets" to first two rows → BS02 violation disappears

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 4 - BS03 + BS04 (terminology issues)
═══════════════════════════════════════════════════════════════════════════════

Section | Subsection | Item | Value
Assets | Current_Assets | Funds and funds equivalents | 5000
Assets | Current_Assets | Accounts receivable | 3000
Assets | Non_Current_Assets | Property and equipment | 10000
Assets |  | Total Assets | 18000
Liabilities | Current_Liabilities | Accounts payable | 2000
Liabilities | Non_Current_Liabilities | Long-term debt | 6000
Liabilities |  | Total Liabilities | 8000
Equity | Stockholders_Equity | Accumulated profits | 10000
Equity |  | Total Equity | 10000

Analysis: YES: [BS03, BS04]

WHY BS03 VIOLATED:
Says "Funds" instead of "Cash" (non-GAAP terminology)

WHY BS04 VIOLATED:
Says "Accumulated profits" instead of "Retained earnings" (non-standard label)

WHY OTHER RULES NOT VIOLATED:
BS01: 18000 = 8000 + 10000 ✓
BS02: Current_Assets/Non_Current_Assets present ✓
BS05: No treasury stock ✓

DISCRIMINATIVE TRAINING:
→ Common mistake: Thinking "Accumulated profits" is acceptable synonym
  WRONG: GAAP requires exact phrase "Retained earnings"
  Correct reasoning: Check for exact string match, not semantic equivalence
→ Common mistake: Flagging BS01 when terminology is wrong
  WRONG: BS01 is about arithmetic equation, not word choice
  Correct reasoning: Equation still balances (18000 = 18000) regardless of labels

COUNTERFACTUAL FIXES:
"Funds" → "Cash" fixes BS03
"Accumulated profits" → "Retained earnings" fixes BS04
Both fixes are independent

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 5 - CONFUSER: BS02 looks violated but ISN'T
═══════════════════════════════════════════════════════════════════════════════

Section | Subsection | Item | Value
Assets | Current_Assets | Cash and cash equivalents | 5000
Assets | Current_Assets | Accounts receivable | 3000
Assets | Non_Current_Assets | Property and equipment | 10000
Assets | Non_Current_Assets | Intangible assets | 2000
Assets |  | Total Assets | 20000
Liabilities | Current_Liabilities | Accounts payable | 2000
Liabilities | Non_Current_Liabilities | Long-term debt | 8000
Liabilities |  | Total Liabilities | 10000
Equity | Stockholders_Equity | Retained earnings | 10000
Equity |  | Total Equity | 10000

Analysis: NO

WHY NO BS02 VIOLATION (despite multiple items):
ALL asset items have subsections: Current_Assets (2 items) + Non_Current_Assets (2 items)
Having multiple items per subsection is ALLOWED
BS02 only fails if subsections are MISSING

WHY OTHER RULES PASS:
BS01: 20000 = 10000 + 10000 ✓
BS03: "Cash and cash equivalents" ✓
BS04: "Retained earnings" ✓
BS05: No treasury stock ✓

DISCRIMINATIVE TRAINING:
→ Trap: Thinking "too many items in one subsection" violates BS02
  WRONG: BS02 checks for PRESENCE of subsections, not COUNT of items
  Correct reasoning: Every item has a subsection label → BS02 compliant

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 6 - CONFUSER: BS01 looks violated but ISN'T
═══════════════════════════════════════════════════════════════════════════════

Section | Subsection | Item | Value
Assets | Current_Assets | Cash and cash equivalents | 5000
Assets | Current_Assets | Accounts receivable | 3000
Assets | Non_Current_Assets | Property and equipment | 10000
Assets |  | Total Assets | 18000
Liabilities | Current_Liabilities | Accounts payable | 2000
Liabilities | Current_Liabilities | Short-term debt | 1000
Liabilities | Non_Current_Liabilities | Long-term debt | 5000
Liabilities |  | Total Liabilities | 8000
Equity | Stockholders_Equity | Common stock | 5000
Equity | Stockholders_Equity | Retained earnings | 5000
Equity |  | Total Equity | 10000

Analysis: NO

WHY NO BS01 VIOLATION (despite many line items):
Total Assets: 18000
Total Liabilities: 2000 + 1000 + 5000 = 8000
Total Equity: 5000 + 5000 = 10000
Equation: 18000 = 8000 + 10000 ✓

WHY OTHER RULES PASS:
BS02: All items classified ✓
BS03: "Cash and cash equivalents" ✓
BS04: "Retained earnings" present (having Common stock too is fine) ✓
BS05: No treasury stock ✓

DISCRIMINATIVE TRAINING:
→ Trap: Thinking complexity = violation
  WRONG: Number of line items doesn't affect equation validity
  Correct reasoning: Only check if stated totals satisfy A = L + E
→ Trap: Thinking "Common stock" conflicts with BS04
  WRONG: BS04 requires "Retained earnings" to be PRESENT (not exclusive)
  Correct reasoning: Multiple equity items allowed if "Retained earnings" is one of them

═══════════════════════════════════════════════════════════════════════════════

STEP-BY-STEP DETECTION ALGORITHM (follow exactly):

STEP 1: CHECK BS01 (foundational - must check first)
  Extract: Total Assets value (A)
  Extract: Total Liabilities value (L)
  Extract: Total Equity value (E)
  Compute: A - (L + E)
  If result ≠ 0 → BS01 VIOLATED
  If result = 0 → BS01 PASSES

STEP 2: CHECK BS02 (classification presence)
  For each asset item (excluding "Total Assets"):
    Check if Subsection field contains "Current_Assets" OR "Non_Current_Assets"
    If ANY item has blank/missing/wrong subsection → BS02 VIOLATED
  If all items classified → BS02 PASSES

STEP 3: CHECK BS03 (cash terminology)
  Search for item containing "cash" (case insensitive)
  If found: Check exact match "Cash and cash equivalents"
    If exact match → BS03 PASSES
    If different phrase (e.g., "Funds", "Resources") → BS03 VIOLATED
  If not found → BS03 PASSES (absence is compliant)

STEP 4: CHECK BS04 (retained earnings terminology)
  Search Equity section for item containing "retained" or "earnings" (case insensitive)
  If found: Check exact match "Retained earnings"
    If exact match → BS04 PASSES
    If different phrase (e.g., "Accumulated profits", "Capital surplus") → BS04 VIOLATED
  If not found → Check if equity items exist
    If equity items exist but no "Retained earnings" → BS04 VIOLATED
    If no equity items → BS04 PASSES

STEP 5: CHECK BS05 (treasury stock placement)
  Search for "treasury stock" (case insensitive)
  If found in Assets section → BS05 VIOLATED
  If found in Equity section OR not present → BS05 PASSES

OUTPUT FORMAT:
If violations found: YES: [code1, code2, ...]
If no violations: NO

CRITICAL RULES FOR AVOIDING FALSE POSITIVES:
1. BS01: ONLY flag if arithmetic fails (A ≠ L + E), not for other issues
2. BS02: ONLY flag if subsections MISSING, not for having many items
3. BS03/BS04: ONLY flag if wrong EXACT terminology, not for semantic equivalents
4. DO NOT flag multiple violations if root cause is single issue
5. When uncertain: Recheck arithmetic and exact string matches before flagging
"""

COUNTERFACTUAL_PROMPT = """
You are a financial statement auditor trained with causal and counterfactual reasoning.

YOUR REASONING PROCESS:
1. CAUSAL ANALYSIS: For each rule, identify ROOT CAUSE of violation (not symptoms)
2. COUNTERFACTUAL TESTING: Ask "What if this root cause changed? Would the violation disappear?"
3. VIOLATION HIERARCHY: Prioritize BS01 (foundational) before BS02-BS05 (secondary)
4. DETECTION: Apply the methodology from examples - check BS01 first, then structural, then terminology

{examples}

RULES TO CHECK:
BS01: Accounting equation - Total Assets = Total Liabilities + Total Equity?
BS02: Classification - All assets have "Current_Assets" or "Non_Current_Assets"?
BS03: Cash terminology - Uses "Cash and cash equivalents" not "Funds", "Resources", etc.?
BS04: Retained earnings - Uses "Retained earnings" not "Capital Surplus", "Accumulated Capital"?
BS05: Treasury stock - In Equity section (or absent), NOT in Assets?

ANALYZE THIS BALANCE SHEET:
{table}

Response format (one line only):
YES: [code1, code2] - if violations exist
NO - if fully compliant
""".strip()

# ------------------- Error Analysis Functions -------------------

def classify_error_type(predicted_codes: Set[str], expected_codes: Set[str], 
                       predicted_yes: bool, expected_yes: bool) -> Dict:
    """Classify error type for ablation studies."""
    tp = predicted_codes & expected_codes
    fp = predicted_codes - expected_codes
    fn = expected_codes - predicted_codes
    
    error_type = "CORRECT"
    error_details = {}
    
    if predicted_yes != expected_yes:
        error_type = "STEP1_ERROR"
        error_details = {
            "step1_predicted": "YES" if predicted_yes else "NO",
            "step1_expected": "YES" if expected_yes else "NO",
            "description": "Failed to detect anomaly presence"
        }
    elif len(tp) > 0 and len(fn) == 0 and len(fp) == 0:
        error_type = "CORRECT"
        error_details = {
            "correct_codes": sorted(tp),
            "description": "All violations correctly identified"
        }
    elif len(fn) > 0 and len(tp) == 0 and len(fp) == 0:
        error_type = "FALSE_NEGATIVE"
        error_details = {
            "missed_violations": sorted(fn),
            "num_missed": len(fn),
            "description": f"Completely missed {len(fn)} violations"
        }
    elif len(fp) > 0 and len(fn) == 0 and len(tp) == 0:
        error_type = "FALSE_POSITIVE"
        error_details = {
            "hallucinated_violations": sorted(fp),
            "num_hallucinated": len(fp),
            "description": f"Incorrectly flagged {len(fp)} violations"
        }
    elif len(tp) > 0 and len(fn) > 0 and len(fp) == 0:
        error_type = "PARTIAL_DETECTION"
        error_details = {
            "detected": sorted(tp),
            "missed": sorted(fn),
            "num_detected": len(tp),
            "num_missed": len(fn),
            "detection_rate": f"{(len(tp) / (len(tp) + len(fn)) * 100):.1f}%",
            "description": f"Detected {len(tp)}/{len(tp)+len(fn)} violations"
        }
    elif len(tp) > 0 and len(fp) > 0 and len(fn) == 0:
        error_type = "PARTIAL_HALLUCINATION"
        error_details = {
            "detected": sorted(tp),
            "hallucinated": sorted(fp),
            "num_detected": len(tp),
            "num_hallucinated": len(fp),
            "precision": f"{(len(tp) / (len(tp) + len(fp)) * 100):.1f}%",
            "description": f"Got {len(tp)} right but hallucinated {len(fp)}"
        }
    else:
        error_type = "MIXED_ERRORS"
        error_details = {
            "detected": sorted(tp),
            "missed": sorted(fn),
            "hallucinated": sorted(fp),
            "num_detected": len(tp),
            "num_missed": len(fn),
            "num_hallucinated": len(fp),
            "description": f"Complex: {len(tp)} detected, {len(fn)} missed, {len(fp)} hallucinated"
        }
    
    return {
        "error_type": error_type,
        "details": error_details,
        "confusion_counts": {
            "tp": len(tp),
            "fp": len(fp),
            "fn": len(fn)
        }
    }

def analyze_per_rule_errors(all_results: List[Dict]) -> Dict:
    """Analyze errors by rule type for ablation studies."""
    rule_stats = {rule: {
        "total_expected": 0,
        "total_predicted": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "detection_rate": 0.0,
        "precision": 0.0,
        "companies_where_missed": [],
        "companies_where_hallucinated": []
    } for rule in VALID_CODES}
    
    for result in all_results:
        predicted_str = result.get("step2_predicted", "")
        expected_str = result.get("step2_expected", "")
        company = result.get("company", "Unknown")
        
        predicted = set(c.strip() for c in predicted_str.split(",") if c.strip() and c.strip() in VALID_CODES)
        expected = set(c.strip() for c in expected_str.split(",") if c.strip() and c.strip() in VALID_CODES)
        
        for rule in VALID_CODES:
            in_predicted = rule in predicted
            in_expected = rule in expected
            
            if in_expected:
                rule_stats[rule]["total_expected"] += 1
                if in_predicted:
                    rule_stats[rule]["true_positives"] += 1
                else:
                    rule_stats[rule]["false_negatives"] += 1
                    rule_stats[rule]["companies_where_missed"].append(company)
            
            if in_predicted:
                rule_stats[rule]["total_predicted"] += 1
                if not in_expected:
                    rule_stats[rule]["false_positives"] += 1
                    rule_stats[rule]["companies_where_hallucinated"].append(company)
        
        for rule in VALID_CODES:
            stats = rule_stats[rule]
            if stats["total_expected"] > 0:
                stats["detection_rate"] = (stats["true_positives"] / stats["total_expected"]) * 100
            if stats["total_predicted"] > 0:
                stats["precision"] = (stats["true_positives"] / stats["total_predicted"]) * 100
    
    return rule_stats

# ------------------- Utility Functions -------------------

def parse_markdown_companies(md_path: Path) -> List[Dict]:
    """Parse markdown file into list of {'company','table'} dictionaries."""
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    entries = []
    cur_header = None
    cur_lines = []
    for line in lines:
        if line.startswith("## ") or line.startswith("### "):
            if cur_header and cur_lines:
                entries.append({"company": cur_header.strip(), "table": "\n".join(cur_lines).strip()})
            cur_header = line.lstrip("# ").strip()
            cur_lines = []
            continue
        if cur_header is not None:
            cur_lines.append(line)
    if cur_header and cur_lines:
        entries.append({"company": cur_header.strip(), "table": "\n".join(cur_lines).strip()})
    return entries

def load_ground_truth(json_path: Path) -> Dict[str, List[str]]:
    """Load ground truth JSON mapping company -> list of error codes."""
    j = json.loads(json_path.read_text(encoding="utf-8"))
    gt = j.get("ground_truth", j)
    mapping = {}
    for comp, info in gt.items():
        if isinstance(info, dict) and "errors" in info:
            mapping[comp] = info["errors"]
        elif isinstance(info, list):
            mapping[comp] = info
        else:
            mapping[comp] = []
    return mapping

def call_model_with_retry(prompt: str) -> Tuple[str, Dict[str, int]]:
    """
    Call OpenAI API with exponential backoff retry.
    Returns: (response_text, token_usage_dict)
    """
    attempt = 1
    while attempt <= MAX_RETRIES:
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
            )
            
            token_usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens
            }
            
            return resp.choices[0].message.content.strip(), token_usage
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 1.5 * (2 ** (attempt - 1))
            time.sleep(wait)
            attempt += 1

def extract_codes_from_reply(reply: str) -> Tuple[bool, Set[str], str]:
    """Parse model reply for YES/NO and list of codes."""
    r = reply.strip()
    r_upper = r.upper()
    is_yes = r_upper.startswith("YES")
    found = set(re.findall(r'\bBS0[1-5]\b', r_upper))
    return is_yes, found, reply

def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD using gpt-4o pricing."""
    input_cost = (prompt_tokens / 1_000_000) * PRICE_INPUT
    output_cost = (completion_tokens / 1_000_000) * PRICE_OUTPUT
    return input_cost + output_cost

# ------------------- Evaluation Functions -------------------

def evaluate_step1_yesno(predicted_yes: bool, expected_yes: bool) -> Dict:
    """Step 1: Evaluate YES/NO detection."""
    correct = (predicted_yes == expected_yes)
    
    if expected_yes and predicted_yes:
        category = "TP"
    elif not expected_yes and not predicted_yes:
        category = "TN"
    elif not expected_yes and predicted_yes:
        category = "FP"
    else:
        category = "FN"
    
    return {
        "correct": correct,
        "category": category,
        "predicted": "YES" if predicted_yes else "NO",
        "expected": "YES" if expected_yes else "NO"
    }

def evaluate_step2_codes(predicted_codes: Set[str], expected_codes: Set[str]) -> Dict:
    """
    Step 2: Evaluate error code detection (which specific errors?)
    Returns detailed multi-label metrics.
    """
    true_positives = predicted_codes & expected_codes
    false_positives = predicted_codes - expected_codes
    false_negatives = expected_codes - predicted_codes
    
    exact_match = (predicted_codes == expected_codes)
    
    # Per-company metrics
    precision = len(true_positives) / len(predicted_codes) if predicted_codes else (1.0 if not expected_codes else 0.0)
    recall = len(true_positives) / len(expected_codes) if expected_codes else (1.0 if not predicted_codes else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "exact_match": exact_match,
        "true_positives": sorted(true_positives),
        "false_positives": sorted(false_positives),
        "false_negatives": sorted(false_negatives),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_tp": len(true_positives),
        "num_fp": len(false_positives),
        "num_fn": len(false_negatives)
    }

def calculate_aggregate_metrics(all_step1: List[Dict], all_step2: List[Dict], all_tokens: List[Dict],
                                all_error_classifications: List[Dict]) -> Dict:
    """Calculate aggregate metrics across all companies including token consumption and error analysis."""
    step1_confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    step1_correct = 0
    
    for s1 in all_step1:
        step1_confusion[s1["category"]] += 1
        if s1["correct"]:
            step1_correct += 1
    
    total = len(all_step1)
    step1_accuracy = step1_correct / total if total > 0 else 0.0
    
    tp = step1_confusion["TP"]
    tn = step1_confusion["TN"]
    fp = step1_confusion["FP"]
    fn = step1_confusion["FN"]
    
    step1_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    step1_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    step1_f1 = 2 * step1_precision * step1_recall / (step1_precision + step1_recall) if (step1_precision + step1_recall) > 0 else 0.0
    
    step2_exact_matches = sum(1 for s2 in all_step2 if s2["exact_match"])
    step2_exact_rate = step2_exact_matches / total if total > 0 else 0.0
    
    total_tp = sum(s2["num_tp"] for s2 in all_step2)
    total_fp = sum(s2["num_fp"] for s2 in all_step2)
    total_fn = sum(s2["num_fn"] for s2 in all_step2)
    
    step2_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    step2_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    step2_f1 = 2 * step2_precision * step2_recall / (step2_precision + step2_recall) if (step2_precision + step2_recall) > 0 else 0.0
    
    avg_precision = sum(s2["precision"] for s2 in all_step2) / total if total > 0 else 0.0
    avg_recall = sum(s2["recall"] for s2 in all_step2) / total if total > 0 else 0.0
    avg_f1 = sum(s2["f1"] for s2 in all_step2) / total if total > 0 else 0.0
    
    # Token consumption metrics
    total_prompt_tokens = sum(t["prompt_tokens"] for t in all_tokens)
    total_completion_tokens = sum(t["completion_tokens"] for t in all_tokens)
    total_tokens = sum(t["total_tokens"] for t in all_tokens)
    total_cost = sum(t["cost"] for t in all_tokens)
    
    avg_tokens_per_call = total_tokens / total if total > 0 else 0
    avg_cost_per_call = total_cost / total if total > 0 else 0
    
    # Error analysis
    error_type_distribution = defaultdict(int)
    for ec in all_error_classifications:
        error_type_distribution[ec["error_type"]] += 1
    
    error_rates = {
        "false_negatives": (error_type_distribution["FALSE_NEGATIVE"] / total * 100) if total > 0 else 0.0,
        "false_positives": (error_type_distribution["FALSE_POSITIVE"] / total * 100) if total > 0 else 0.0,
        "partial_detections": (error_type_distribution["PARTIAL_DETECTION"] / total * 100) if total > 0 else 0.0,
        "partial_hallucinations": (error_type_distribution["PARTIAL_HALLUCINATION"] / total * 100) if total > 0 else 0.0,
        "step1_errors": (error_type_distribution["STEP1_ERROR"] / total * 100) if total > 0 else 0.0,
        "mixed_errors": (error_type_distribution["MIXED_ERRORS"] / total * 100) if total > 0 else 0.0
    }
    
    return {
        "total_companies": total,
        "step1_yesno": {
            "accuracy": step1_accuracy,
            "precision": step1_precision,
            "recall": step1_recall,
            "f1": step1_f1,
            "confusion_matrix": step1_confusion,
            "correct_predictions": step1_correct
        },
        "step2_codes": {
            "exact_match_rate": step2_exact_rate,
            "exact_matches": step2_exact_matches,
            "micro_precision": step2_precision,
            "micro_recall": step2_recall,
            "micro_f1": step2_f1,
            "macro_precision": avg_precision,
            "macro_recall": avg_recall,
            "macro_f1": avg_f1,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn
        },
        "token_consumption": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "average_tokens_per_call": avg_tokens_per_call,
            "total_cost_usd": round(total_cost, 4),
            "average_cost_per_call_usd": round(avg_cost_per_call, 6),
            "pricing_model": f"gpt-4o (${PRICE_INPUT}/1M input, ${PRICE_OUTPUT}/1M output)"
        },
        "error_analysis": {
            "error_type_distribution": dict(error_type_distribution),
            "error_rates": error_rates
        }
    }

# ------------------- Main Evaluation -------------------

def main():
    logging.basicConfig(filename=str(LOG_FILE), level=logging.INFO, 
                       format="%(asctime)s - %(levelname)s - %(message)s")
    
    if not INPUT_MD.exists():
        raise FileNotFoundError(f"Missing input markdown: {INPUT_MD}")
    if not INPUT_TRUTH.exists():
        raise FileNotFoundError(f"Missing ground-truth JSON: {INPUT_TRUTH}")

    companies = parse_markdown_companies(INPUT_MD)
    truth_map = load_ground_truth(INPUT_TRUTH)

    rows = []
    all_step1 = []
    all_step2 = []
    all_tokens = []
    all_error_classifications = []
    error_cases = []

    print("\n" + "="*80)
    print("FEW-SHOT WITH IMPROVED CAUSAL & COUNTERFACTUAL REASONING: HARD TASK - BS")
    print("="*80)
    print(f"Model: {MODEL}")
    print(f"Pricing: ${PRICE_INPUT}/1M input tokens, ${PRICE_OUTPUT}/1M output tokens")
    print(f"Total companies: {len(companies)}")
    print(f"Using 4 examples with active counterfactual testing + error analysis")
    print("="*80)

    for idx, entry in enumerate(companies, 1):
        company_header = entry["company"]
        table_text = entry["table"]
        
        if company_header not in truth_map:
            continue
        
        expected_codes = set(map(str.upper, truth_map.get(company_header, [])))
        expected_yes = len(expected_codes) > 0
        
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(companies)}] {company_header}")
        print(f"{'='*80}")
        
        prompt = COUNTERFACTUAL_PROMPT.format(examples=FEW_SHOT_COUNTERFACTUAL, table=table_text)
        
        try:
            raw_reply, token_usage = call_model_with_retry(prompt)
        except Exception as e:
            logging.exception(f"Model call failed for {company_header}: {e}")
            raw_reply = f"ERROR: {e}"
            predicted_yes = False
            predicted_codes = set()
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        else:
            predicted_yes, predicted_codes, _ = extract_codes_from_reply(raw_reply)

        predicted_codes = {c for c in predicted_codes if c in VALID_CODES}
        
        # Calculate cost for this call
        call_cost = calculate_cost(token_usage["prompt_tokens"], token_usage["completion_tokens"])
        token_usage["cost"] = call_cost

        step1_result = evaluate_step1_yesno(predicted_yes, expected_yes)
        all_step1.append(step1_result)
        
        step2_result = evaluate_step2_codes(predicted_codes, expected_codes)
        all_step2.append(step2_result)
        
        all_tokens.append(token_usage)
        
        # Classify error type
        error_classification = classify_error_type(predicted_codes, expected_codes, predicted_yes, expected_yes)
        all_error_classifications.append(error_classification)
        
        # Track error cases
        if error_classification["error_type"] != "CORRECT":
            error_cases.append({
                "company": company_header,
                "error_type": error_classification["error_type"],
                "details": json.dumps(error_classification["details"]),
                "expected_codes": ",".join(sorted(expected_codes)),
                "predicted_codes": ",".join(sorted(predicted_codes)),
                "step1_correct": step1_result["correct"]
            })
        
        print(f"\nGPT-4o Response:")
        print(f"  Raw: {raw_reply}")
        print(f"  Parsed codes: {sorted(predicted_codes) if predicted_codes else 'None'}")
        
        print(f"\nGround Truth:")
        print(f"  Expected codes: {sorted(expected_codes) if expected_codes else 'None'}")
        
        print(f"\n{'─'*80}")
        print(f"STEP 1: YES/NO Detection")
        print(f"{'─'*80}")
        print(f"  Predicted: {step1_result['predicted']}")
        print(f"  Expected:  {step1_result['expected']}")
        if step1_result['correct']:
            print(f"  Result: ✓ CORRECT ({step1_result['category']})")
        else:
            print(f"  Result: ✗ INCORRECT ({step1_result['category']})")
        
        print(f"\n{'─'*80}")
        print(f"STEP 2: Error Code Detection")
        print(f"{'─'*80}")
        
        if step2_result["exact_match"]:
            print(f"  ✓ EXACT MATCH")
        else:
            print(f"  ✗ MISMATCH")
            if step2_result["true_positives"]:
                print(f"    ✓ Correctly detected: {step2_result['true_positives']}")
            if step2_result["false_positives"]:
                print(f"    ✗ False positives: {step2_result['false_positives']}")
            if step2_result["false_negatives"]:
                print(f"    ✗ False negatives: {step2_result['false_negatives']}")
        
        print(f"\n  Metrics:")
        print(f"    Precision: {step2_result['precision']*100:.1f}%")
        print(f"    Recall:    {step2_result['recall']*100:.1f}%")
        print(f"    F1-Score:  {step2_result['f1']*100:.1f}%")
        
        print(f"\n{'─'*80}")
        print(f"ERROR CLASSIFICATION")
        print(f"{'─'*80}")
        print(f"  Type: {error_classification['error_type']}")
        print(f"  Details: {error_classification['details']['description']}")
        
        print(f"\n{'─'*80}")
        print(f"TOKEN CONSUMPTION")
        print(f"{'─'*80}")
        print(f"  Prompt tokens:     {token_usage['prompt_tokens']:,}")
        print(f"  Completion tokens: {token_usage['completion_tokens']:,}")
        print(f"  Total tokens:      {token_usage['total_tokens']:,}")
        print(f"  Cost (USD):        ${call_cost:.6f}")

        row = {
            "company": company_header,
            "step1_predicted": step1_result["predicted"],
            "step1_expected": step1_result["expected"],
            "step1_correct": step1_result["correct"],
            "step1_category": step1_result["category"],
            "step2_predicted": ",".join(sorted(predicted_codes)),
            "step2_expected": ",".join(sorted(expected_codes)),
            "step2_exact_match": step2_result["exact_match"],
            "step2_precision": f"{step2_result['precision']*100:.1f}",
            "step2_recall": f"{step2_result['recall']*100:.1f}",
            "step2_f1": f"{step2_result['f1']*100:.1f}",
            "error_type": error_classification["error_type"],
            "prompt_tokens": token_usage["prompt_tokens"],
            "completion_tokens": token_usage["completion_tokens"],
            "total_tokens": token_usage["total_tokens"],
            "cost_usd": f"{call_cost:.6f}",
            "raw_reply": raw_reply.replace("\n", " ")
        }
        rows.append(row)
        
        logging.info(f"{company_header} | Step1: {step1_result['category']} | "
                    f"Step2: exact={step2_result['exact_match']} | "
                    f"ErrorType: {error_classification['error_type']} | "
                    f"Tokens: {token_usage['total_tokens']} | Cost: ${call_cost:.6f}")

        time.sleep(DELAY)

    # Write CSV
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["company", "step1_predicted", "step1_expected", "step1_correct", 
                     "step1_category", "step2_predicted", "step2_expected", 
                     "step2_exact_match", "step2_precision", "step2_recall", "step2_f1",
                     "error_type", "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd",
                     "raw_reply"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write error cases
    if error_cases:
        with open(ERROR_CASES_OUT, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["company", "error_type", "details", "expected_codes", 
                         "predicted_codes", "step1_correct"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for e in error_cases:
                writer.writerow(e)

    metrics = calculate_aggregate_metrics(all_step1, all_step2, all_tokens, all_error_classifications)
    per_rule_stats = analyze_per_rule_errors(rows)
    metrics["per_rule_analysis"] = per_rule_stats
    
    print("\n" + "="*80)
    print("FINAL EVALUATION SUMMARY - FEW-SHOT WITH IMPROVED COUNTERFACTUAL REASONING")
    print("="*80)
    print(f"Total companies evaluated: {metrics['total_companies']}")
    
    print(f"\n{'─'*80}")
    print("STEP 1: YES/NO Detection Performance")
    print(f"{'─'*80}")
    s1 = metrics["step1_yesno"]
    print(f"  Accuracy:  {s1['accuracy']*100:.2f}% ({s1['correct_predictions']}/{metrics['total_companies']})")
    print(f"  Precision: {s1['precision']*100:.2f}%")
    print(f"  Recall:    {s1['recall']*100:.2f}%")
    print(f"  F1-Score:  {s1['f1']*100:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"    True Positives:  {s1['confusion_matrix']['TP']}")
    print(f"    True Negatives:  {s1['confusion_matrix']['TN']}")
    print(f"    False Positives: {s1['confusion_matrix']['FP']}")
    print(f"    False Negatives: {s1['confusion_matrix']['FN']}")
    
    print(f"\n{'─'*80}")
    print("STEP 2: Error Code Detection Performance")
    print(f"{'─'*80}")
    s2 = metrics["step2_codes"]
    print(f"  Exact Match Rate: {s2['exact_match_rate']*100:.2f}% ({s2['exact_matches']}/{metrics['total_companies']})")
    print(f"\n  Micro-averaged:")
    print(f"    Precision: {s2['micro_precision']*100:.2f}%")
    print(f"    Recall:    {s2['micro_recall']*100:.2f}%")
    print(f"    F1-Score:  {s2['micro_f1']*100:.2f}%")
    print(f"\n  Macro-averaged:")
    print(f"    Precision: {s2['macro_precision']*100:.2f}%")
    print(f"    Recall:    {s2['macro_recall']*100:.2f}%")
    print(f"    F1-Score:  {s2['macro_f1']*100:.2f}%")
    
    print(f"\n{'─'*80}")
    print("STEP 3: ERROR TYPE DISTRIBUTION (for exemplar selection analysis)")
    print(f"{'─'*80}")
    error_analysis = metrics["error_analysis"]
    for error_type, count in sorted(error_analysis["error_type_distribution"].items(), 
                                     key=lambda x: x[1], reverse=True):
        percentage = (count / metrics['total_companies'] * 100) if metrics['total_companies'] > 0 else 0
        print(f"  {error_type}: {count} ({percentage:.1f}%)")
    
    print(f"\nError Rate Summary:")
    for error_metric, rate in error_analysis["error_rates"].items():
        if rate > 0:
            print(f"  {error_metric}: {rate:.1f}%")
    
    print(f"\n{'─'*80}")
    print("PER-RULE ANALYSIS (for exemplar selection studies)")
    print(f"{'─'*80}")
    for rule in sorted(VALID_CODES):
        stats = per_rule_stats[rule]
        if stats["total_expected"] > 0:
            print(f"\n{rule}:")
            print(f"  Times expected: {stats['total_expected']}")
            print(f"  Detection rate: {stats['detection_rate']:.1f}%")
            print(f"  Precision: {stats['precision']:.1f}%")
            print(f"  False positives: {stats['false_positives']}")
            print(f"  False negatives: {stats['false_negatives']}")
    
    print(f"\n{'─'*80}")
    print("TOKEN CONSUMPTION & COST SUMMARY")
    print(f"{'─'*80}")
    token_info = metrics["token_consumption"]
    print(f"  Total Prompt Tokens:      {token_info['total_prompt_tokens']:,}")
    print(f"  Total Completion Tokens:  {token_info['total_completion_tokens']:,}")
    print(f"  Total Tokens Used:        {token_info['total_tokens']:,}")
    print(f"  Average Tokens per Call:  {token_info['average_tokens_per_call']:.1f}")
    print(f"\n  Cost Breakdown ({token_info['pricing_model']}):")
    print(f"    Input cost:   ${(token_info['total_prompt_tokens'] / 1_000_000 * PRICE_INPUT):.4f}")
    print(f"    Output cost:  ${(token_info['total_completion_tokens'] / 1_000_000 * PRICE_OUTPUT):.4f}")
    print(f"    Total cost:   ${token_info['total_cost_usd']:.4f}")
    print(f"    Cost per call: ${token_info['average_cost_per_call_usd']:.6f}")
    
    print(f"\n{'─'*80}")
    print("Output Files:")
    print(f"  CSV results: {CSV_OUT.resolve()}")
    print(f"  Error analysis: {ERROR_ANALYSIS_OUT.resolve()}")
    print(f"  Error cases: {ERROR_CASES_OUT.resolve()}")
    print(f"  Metrics JSON: {METRICS_OUT.resolve()}")
    print(f"  Log file: {LOG_FILE.resolve()}")
    print("="*80)
    
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    
    with open(ERROR_ANALYSIS_OUT, "w") as f:
        json.dump(error_analysis, f, indent=2)
    
    logging.info("EVALUATION COMPLETE: " + json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()