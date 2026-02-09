#!/usr/bin/env python3
"""
hard_few_shot_BS_gpt.py

Few-shot evaluation for hard multi-error detection task with error analysis.
Uses in-context examples (stratified by rule type) to improve model performance.

Inputs:
 - BS-MIXED-100.md            (mixed dataset)
 - BS-MIXED-100-truth.json    (ground-truth JSON)

Outputs:
 - few_shot_eval_results.csv              (detailed results with error types)
 - few_shot_eval_summary.log              (summary + logs)
 - few_shot_eval_metrics.json             (structured metrics)
 - few_shot_eval_error_analysis.json      (error type classification for ablation)
 - few_shot_eval_error_cases.csv          (detailed error cases)
 - few_shot_exemplar_info.json            (stratified exemplar selection metadata)

Features:
 - Two-step evaluation (YES/NO detection + code detection)
 - Error classification (False Negative, False Positive, Partial Detection, etc.)
 - Per-rule performance analysis
 - Token consumption tracking
 - Supports future exemplar selection analysis (random, stratified, hard negative mining)
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

CSV_OUT = Path("few_shot_eval_results.csv")
LOG_FILE = Path("few_shot_eval_summary.log")
METRICS_OUT = Path("few_shot_eval_metrics.json")
ERROR_ANALYSIS_OUT = Path("few_shot_eval_error_analysis.json")
ERROR_CASES_OUT = Path("few_shot_eval_error_cases.csv")
EXEMPLAR_INFO_OUT = Path("few_shot_exemplar_info.json")

MODEL = "gpt-4o"
TEMPERATURE = 0.0
DELAY = 1.0
MAX_RETRIES = 4

# Pricing for gpt-4o (in USD per million tokens)
PRICE_INPUT = 2.50    # $2.50 per 1M input tokens
PRICE_OUTPUT = 10.00  # $10.00 per 1M output tokens

VALID_CODES = {"BS01", "BS02", "BS03", "BS04", "BS05"}

client = OpenAI()

# ===============================================================================
# STRATIFIED FEW-SHOT EXAMPLES (for exemplar selection analysis)
# Each example labeled with: rule_focus, difficulty, violation_type
# ===============================================================================

EXEMPLAR_METADATA = {
    "strategy": "stratified_by_rule",
    "total_exemplars": 5,
    "exemplars": [
        {
            "id": "ex1",
            "label": "Compliant (Baseline)",
            "rule_focus": ["BS01", "BS02", "BS03", "BS04", "BS05"],
            "violation_type": "none",
            "difficulty": "easy"
        },
        {
            "id": "ex2",
            "label": "BS01 + BS02 (Classification + Equation)",
            "rule_focus": ["BS01", "BS02"],
            "violation_type": "multi",
            "difficulty": "hard"
        },
        {
            "id": "ex3",
            "label": "BS03 + BS04 (Terminology + Equity)",
            "rule_focus": ["BS03", "BS04"],
            "violation_type": "multi",
            "difficulty": "hard"
        },
        {
            "id": "ex4",
            "label": "BS02 Single (Classification only)",
            "rule_focus": ["BS02"],
            "violation_type": "single",
            "difficulty": "medium"
        },
        {
            "id": "ex5",
            "label": "BS05 Hard Negative (Treasury stock edge case)",
            "rule_focus": ["BS05"],
            "violation_type": "hard_negative",
            "difficulty": "hard"
        }
    ]
}

# ===============================================================================
# FEW-SHOT EXAMPLES (Stratified by Rule Type for Better Exemplar Selection)
# ===============================================================================

FEW_SHOT_EXAMPLES = """
STRATIFIED FEW-SHOT EXAMPLES:
(Selected to cover all rule types with varying difficulty levels)

──────────────────────────────────────────────────────────────────────────────
EXEMPLAR 1: COMPLIANT BASELINE (NO violations)
Strategy: Covers all rules (BS01-BS05)
──────────────────────────────────────────────────────────────────────────────
Assets | Current_Assets | Cash and cash equivalents | 5000
Assets | Current_Assets | Accounts receivable | 3000
Assets | Non_Current_Assets | Property and equipment | 10000
Assets |  | Total Assets | 18000
Liabilities | Current_Liabilities | Accounts payable | 2000
Liabilities | Non_Current_Liabilities | Long-term debt | 6000
Liabilities |  | Total Liabilities | 8000
Equity | Stockholders_Equity | Retained earnings | 10000
Equity |  | Total Equity | 10000

✓ All rules SATISFIED:
  - BS01: Assets (18000) = Liabilities (8000) + Equity (10000) ✓
  - BS02: Current/Non-Current classification labels present ✓
  - BS03: "Cash and cash equivalents" (correct GAAP term) ✓
  - BS04: "Retained earnings" (correct label) ✓
  - BS05: No treasury stock in assets ✓

VERDICT: NO

──────────────────────────────────────────────────────────────────────────────
EXEMPLAR 2: BS01 + BS02 VIOLATIONS (Equation + Classification)
Strategy: Tests math accuracy + structural understanding
──────────────────────────────────────────────────────────────────────────────
Assets | | Cash and cash equivalents | 5000
Assets | | Accounts receivable | 3000
Assets | Non_Current_Assets | Property and equipment | 10000
Assets | | Total Assets | 18500
Liabilities | Current_Liabilities | Accounts payable | 2000
Liabilities | Non_Current_Liabilities | Long-term debt | 6000
Liabilities | | Total Liabilities | 8000
Equity | Stockholders_Equity | Retained earnings | 10000
Equity | | Total Equity | 10000

✗ VIOLATIONS DETECTED:
  - BS01: Assets (18500) ≠ Liabilities (8000) + Equity (10000 = 18000) - OFF BY $500
    → Indicates calculation error or unrecorded liability
  - BS02: Missing "Current_Assets" classification for first two items
    → Items listed without clear current/non-current categorization
  - BS03: ✓ "Cash and cash equivalents" correct
  - BS04: ✓ "Retained earnings" correct
  - BS05: ✓ No treasury stock issue

VERDICT: YES: [BS01, BS02]

──────────────────────────────────────────────────────────────────────────────
EXEMPLAR 3: BS03 + BS04 VIOLATIONS (Terminology + Equity Presentation)
Strategy: Tests non-numeric rule understanding
──────────────────────────────────────────────────────────────────────────────
Assets | Current_Assets | Working Capital and resources | 5000
Assets | Current_Assets | Accounts receivable | 3000
Assets | Non_Current_Assets | Property and equipment | 10000
Assets | | Total Assets | 18000
Liabilities | Current_Liabilities | Accounts payable | 2000
Liabilities | Non_Current_Liabilities | Long-term debt | 6000
Liabilities | | Total Liabilities | 8000
Equity | Stockholders_Equity | Capital Surplus | 10000
Equity | | Total Equity | 10000

✗ VIOLATIONS DETECTED:
  - BS01: ✓ Assets (18000) = Liabilities (8000) + Equity (10000) - CORRECT
  - BS02: ✓ Current/Non-Current labels present
  - BS03: "Working Capital and resources" used INSTEAD of "Cash"
    → Non-GAAP terminology (working capital is not a cash account)
    → "Resources" is ambiguous (not standard)
  - BS04: "Capital Surplus" used INSTEAD of "Retained Earnings"
    → Capital Surplus is paid-in capital, not accumulated profits
    → Violates retained earnings presentation requirement
  - BS05: ✓ No treasury stock misclassification

VERDICT: YES: [BS03, BS04]

──────────────────────────────────────────────────────────────────────────────
EXEMPLAR 4: BS02 SINGLE VIOLATION (Classification only)
Strategy: Tests ability to isolate single rule without multi-violation confusion
──────────────────────────────────────────────────────────────────────────────
Assets | | Cash and cash equivalents | 5000
Assets | | Accounts receivable | 3000
Assets | | Property and equipment | 10000
Assets | | Total Assets | 18000
Liabilities | | Accounts payable | 2000
Liabilities | | Long-term debt | 6000
Liabilities | | Total Liabilities | 8000
Equity | | Retained earnings | 10000
Equity | | Total Equity | 10000

✗ VIOLATIONS DETECTED:
  - BS01: ✓ Assets (18000) = Liabilities (8000) + Equity (10000) - CORRECT
  - BS02: ALL items missing classification subsection labels
    → No "Current_Assets", "Non_Current_Assets", "Current_Liabilities", "Non_Current_Liabilities"
    → Items listed without structural categorization
  - BS03: ✓ "Cash and cash equivalents" correct
  - BS04: ✓ "Retained earnings" correct
  - BS05: ✓ No treasury stock issue

VERDICT: YES: [BS02]

──────────────────────────────────────────────────────────────────────────────
EXEMPLAR 5: BS05 HARD NEGATIVE (Treasury Stock - Edge Case)
Strategy: Tests edge case understanding - common hard negative for mining
──────────────────────────────────────────────────────────────────────────────
Assets | Current_Assets | Cash and cash equivalents | 5000
Assets | Current_Assets | Accounts receivable | 3000
Assets | Non_Current_Assets | Property and equipment | 10000
Assets | Non_Current_Assets | Treasury stock - restricted | (1000)
Assets | | Total Assets | 17000
Liabilities | Current_Liabilities | Accounts payable | 2000
Liabilities | Non_Current_Liabilities | Long-term debt | 6000
Liabilities | | Total Liabilities | 8000
Equity | Stockholders_Equity | Common stock | 5000
Equity | Stockholders_Equity | Retained earnings | 4000
Equity | Stockholders_Equity | Treasury stock (contra-account) | (1000)
Equity | | Total Equity | 9000
Equity | | Total Equity | 9000

✗ VIOLATION ANALYSIS:
  - BS01: ✓ Assets (17000) = Liabilities (8000) + Equity (9000) - CORRECT
  - BS02: ✓ Current/Non-Current labels present
  - BS03: ✓ "Cash and cash equivalents" correct
  - BS04: ✓ "Retained earnings" correct
  - BS05: Treasury stock appears in TWO PLACES
    → Line "Treasury stock - restricted" in Non-Current Assets (WRONG)
    → Line "Treasury stock (contra-account)" in Equity section (CORRECT placement)
    → BS05 VIOLATED because treasury stock is in Assets (even if also correctly in Equity)
    → Rule requires: Either in Equity only OR not present, NOT in both

VERDICT: YES: [BS05]
Rationale: This is a "hard negative" - superficially looks compliant (equation balances, correct categories)
but violates BS05 due to misclassification in Assets section. Good test for edge cases.

"""

# ------------------- Prompt Template -------------------
FEW_SHOT_PROMPT = """
You are a financial statement auditor. Your task is to identify if a balance sheet has violations and list ALL violation codes.

STRATIFIED EXAMPLES (covering all rule types):
{examples}

RULES:

BS01: Accounting Equation (Numeric)
  Rule: Total Assets = Total Liabilities + Total Equity (exactly)
  Violation: Any difference, even $1
  Tip: Look for mathematical balance issues

BS02: Classification Structure (Structural)
  Rule: Must have "Current_Assets", "Non_Current_Assets", "Current_Liabilities", "Non_Current_Liabilities"
  Violation: Missing or incorrect subsection labels
  Tip: Check for clear current/non-current separation

BS03: Cash Terminology (Terminology)
  Rule: Use "Cash and cash equivalents" or "Restricted cash"
  Violation: "Funds", "Resources", "Working Capital", "Capital" (instead of cash)
  Tip: Be strict about GAAP terminology for cash

BS04: Retained Earnings Presentation (Terminology)
  Rule: Use "Retained Earnings" or "Accumulated Deficit" or omit
  Violation: "Capital Surplus", "General Reserve", "Accumulated Capital"
  Tip: Don't confuse accumulated earnings with paid-in capital

BS05: Treasury Stock Classification (Structural)
  Rule: Treasury stock in Equity (as contra-account) OR absent entirely
  Violation: Treasury stock appears in Assets section
  Tip: Treasury stock is a deduction from equity, never an asset

DECISION PROCEDURE:
1. Check BS01 first (mathematical accuracy)
2. Check BS02 (structural labels)
3. Check BS03 (cash terminology)
4. Check BS04 (equity terminology)
5. Check BS05 (treasury stock placement)
6. Return ALL violations found, or NO if compliant

Response format (one line only):
YES: [code1, code2] - if violations exist
NO - if fully compliant

BALANCE SHEET TO ANALYZE:
{table}

Your response (ONE line only):
""".strip()

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
    """Call OpenAI API with exponential backoff retry."""
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

# ------------------- Error Classification -------------------

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
    """Step 2: Evaluate error code detection."""
    true_positives = predicted_codes & expected_codes
    false_positives = predicted_codes - expected_codes
    false_negatives = expected_codes - predicted_codes
    
    exact_match = (predicted_codes == expected_codes)
    
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
        "num_fn": len(false_negatives),
        "predicted": ",".join(sorted(predicted_codes)),
        "expected": ",".join(sorted(expected_codes))
    }

def calculate_aggregate_metrics(all_step1: List[Dict], all_step2: List[Dict], all_tokens: List[Dict],
                                all_error_classifications: List[Dict]) -> Dict:
    """Calculate aggregate metrics with error analysis."""
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
    
    total_prompt_tokens = sum(t["prompt_tokens"] for t in all_tokens)
    total_completion_tokens = sum(t["completion_tokens"] for t in all_tokens)
    total_tokens = sum(t["total_tokens"] for t in all_tokens)
    total_cost = sum(t["cost"] for t in all_tokens)
    
    avg_tokens_per_call = total_tokens / total if total > 0 else 0
    avg_cost_per_call = total_cost / total if total > 0 else 0
    
    error_distribution = defaultdict(int)
    for error_class in all_error_classifications:
        error_distribution[error_class["error_type"]] += 1
    
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
        "error_analysis": {
            "error_type_distribution": dict(error_distribution),
            "error_rates": {
                "false_negatives": (error_distribution["FALSE_NEGATIVE"] / total * 100) if total > 0 else 0.0,
                "false_positives": (error_distribution["FALSE_POSITIVE"] / total * 100) if total > 0 else 0.0,
                "partial_detection": (error_distribution["PARTIAL_DETECTION"] / total * 100) if total > 0 else 0.0,
                "partial_hallucination": (error_distribution["PARTIAL_HALLUCINATION"] / total * 100) if total > 0 else 0.0,
                "step1_errors": (error_distribution["STEP1_ERROR"] / total * 100) if total > 0 else 0.0,
                "mixed_errors": (error_distribution["MIXED_ERRORS"] / total * 100) if total > 0 else 0.0,
                "correct": (error_distribution["CORRECT"] / total * 100) if total > 0 else 0.0
            }
        },
        "token_consumption": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "average_tokens_per_call": avg_tokens_per_call,
            "total_cost_usd": round(total_cost, 4),
            "average_cost_per_call_usd": round(avg_cost_per_call, 6),
            "pricing_model": f"gpt-4o (${PRICE_INPUT}/1M input, ${PRICE_OUTPUT}/1M output)"
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
    print("FEW-SHOT EVALUATION: HARD TASK - BS (with Error Analysis)")
    print("="*80)
    print(f"Model: {MODEL}")
    print(f"Exemplar Selection Strategy: {EXEMPLAR_METADATA['strategy']}")
    print(f"Number of Exemplars: {EXEMPLAR_METADATA['total_exemplars']}")
    print(f"Pricing: ${PRICE_INPUT}/1M input tokens, ${PRICE_OUTPUT}/1M output tokens")
    print(f"Total companies: {len(companies)}")
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
        
        prompt = FEW_SHOT_PROMPT.format(examples=FEW_SHOT_EXAMPLES, table=table_text)
        
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
        
        call_cost = calculate_cost(token_usage["prompt_tokens"], token_usage["completion_tokens"])
        token_usage["cost"] = call_cost

        step1_result = evaluate_step1_yesno(predicted_yes, expected_yes)
        all_step1.append(step1_result)
        
        step2_result = evaluate_step2_codes(predicted_codes, expected_codes)
        all_step2.append(step2_result)
        
        all_tokens.append(token_usage)
        
        error_classification = classify_error_type(predicted_codes, expected_codes, predicted_yes, expected_yes)
        all_error_classifications.append(error_classification)
        
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
        print(f"STEP 3: Error Type Classification")
        print(f"{'─'*80}")
        print(f"  Error Type: {error_classification['error_type']}")
        print(f"  Details: {error_classification['details'].get('description', 'N/A')}")
        
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
        
        if error_classification["error_type"] != "CORRECT":
            error_cases.append({
                "company": company_header,
                "error_type": error_classification["error_type"],
                "details": json.dumps(error_classification["details"]),
                "expected_codes": ",".join(sorted(expected_codes)),
                "predicted_codes": ",".join(sorted(predicted_codes)),
                "step1_correct": step1_result["correct"]
            })
        
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
                     "error_type", "prompt_tokens", "completion_tokens", "total_tokens", 
                     "cost_usd", "raw_reply"]
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
    print("FINAL EVALUATION SUMMARY - FEW-SHOT WITH ERROR ANALYSIS")
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
    print(f"  Error cases CSV: {ERROR_CASES_OUT.resolve()}")
    print(f"  Error analysis JSON: {ERROR_ANALYSIS_OUT.resolve()}")
    print(f"  Metrics JSON: {METRICS_OUT.resolve()}")
    print(f"  Exemplar metadata: {EXEMPLAR_INFO_OUT.resolve()}")
    print(f"  Log file: {LOG_FILE.resolve()}")
    print("="*80)
    
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    
    error_analysis_data = {
        "error_type_distribution": error_analysis["error_type_distribution"],
        "error_rates": error_analysis["error_rates"],
        "per_rule_analysis": {k: {
            "detection_rate": v["detection_rate"],
            "precision": v["precision"],
            "total_expected": v["total_expected"],
            "false_positives": v["false_positives"],
            "false_negatives": v["false_negatives"]
        } for k, v in per_rule_stats.items()}
    }
    with open(ERROR_ANALYSIS_OUT, "w") as f:
        json.dump(error_analysis_data, f, indent=2)
    
    with open(EXEMPLAR_INFO_OUT, "w") as f:
        json.dump(EXEMPLAR_METADATA, f, indent=2)
    
    logging.info("EVALUATION COMPLETE WITH ERROR ANALYSIS")

if __name__ == "__main__":
    main()