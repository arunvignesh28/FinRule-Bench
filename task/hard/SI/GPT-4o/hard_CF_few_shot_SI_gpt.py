#!/usr/bin/env python3
"""
hard_CF_few_shot_SI_gpt.py

Few-shot with IMPROVED causal and counterfactual explanations for hard multi-error detection (Income Statements).
Includes error analysis for ablation studies (matching few-shot/counterfactual implementations for SE, BS, CF).

Inputs:
 - SI-MIXED-100.md            (mixed dataset: 20% clean, 80% anomaly)
 - SI-MIXED-100-truth.json    (ground-truth JSON)

Outputs:
 - few_shot_cf_eval_results_SI.csv              (with error_type column)
 - few_shot_cf_eval_summary_SI.log
 - few_shot_cf_eval_metrics_SI.json             (with error analysis)
 - few_shot_cf_eval_error_analysis_SI.json      (error type classification for ablation)
 - few_shot_cf_eval_error_cases_SI.csv          (detailed error cases)

Model: gpt-4o (cost-optimized for few-shot)
Token & Cost Tracking: Enabled
Error Analysis: Enabled for ablation studies (7 error categories)
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
INPUT_MD = Path("SI-MIXED-100.md")
INPUT_TRUTH = Path("SI-MIXED-100-truth.json")

CSV_OUT = Path("few_shot_cf_eval_results_SI.csv")
LOG_FILE = Path("few_shot_cf_eval_summary_SI.log")
METRICS_OUT = Path("few_shot_cf_eval_metrics_SI.json")
ERROR_ANALYSIS_OUT = Path("few_shot_cf_eval_error_analysis_SI.json")
ERROR_CASES_OUT = Path("few_shot_cf_eval_error_cases_SI.csv")

MODEL = "gpt-4o"
TEMPERATURE = 0.0
DELAY = 1.0
MAX_RETRIES = 4

# Pricing for gpt-4o (in USD per million tokens)
PRICE_INPUT = 2.50    # $2.50 per 1M input tokens
PRICE_OUTPUT = 10.00  # $10.00 per 1M output tokens

VALID_CODES = {"SI01", "SI02", "SI03", "SI04", "SI05"}

client = OpenAI()

# ------------------- Few-Shot Examples with Improved Causal & Counterfactual Reasoning -------------------

FEW_SHOT_COUNTERFACTUAL = """
EXAMPLES WITH DISCRIMINATIVE CAUSAL REASONING:

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 1 - CLEAN BASELINE (NO violations)
═══════════════════════════════════════════════════════════════════════════════

Revenue | 100,000
Cost of Goods Sold | (60,000)
Gross Profit | 40,000

Operating Expenses |
  Sales and Marketing | (8,000)
  Administrative | (5,000)
Operating Income | 27,000

Non-Operating Income (Expense) |
  Interest Expense | (2,000)
  Other Income | 500
Net Income | 25,500

Earnings Per Share |
  Basic | 4.25
  Diluted | 4.20

Analysis: NO

WHY NO VIOLATIONS:
SI01: Non-operating items in non-operating section ✓ (check classification first)
SI02: Depreciation in operating section ✓ (check location exists)
SI03: Both Basic and Diluted EPS present ✓ (check completeness)
SI04: No "Miscellaneous other income" ✓ (check for vague items)
SI05: Interest Expense present ✓ (check for critical financing cost)

CRITICAL COUNTERFACTUAL (train discrimination):
→ If "Other Income" were in Operating Expenses instead of Non-Operating?
  RESULT: SI01 violation (misclassified non-operating item)
  Current state: "Other Income" in Non-Operating, so NO violation
→ If Diluted EPS line were missing?
  RESULT: SI03 violation (incomplete EPS)
  Current state: "Diluted | 4.20" present, so NO violation

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 2 - SI01 ONLY (misclassification)
═══════════════════════════════════════════════════════════════════════════════

Revenue | 100,000
Cost of Goods Sold | (60,000)
Gross Profit | 40,000

Operating Expenses |
  Sales and Marketing | (8,000)
  Administrative | (5,000)
  Other Income | 500
Operating Income | 27,500

Non-Operating Income (Expense) |
  Interest Expense | (2,000)
Net Income | 25,500

Earnings Per Share |
  Basic | 4.25
  Diluted | 4.20

Analysis: YES: [SI01]

WHY SI01 VIOLATED:
Root cause: "Other Income" moved from Non-Operating into Operating section
Consequence: Operating income overstated (27,500 vs 27,000 correct)
User impact: Investors misinterpret operational profitability

WHY OTHER RULES NOT VIOLATED:
SI02: Depreciation not mentioned ✓ (absence is compliant)
SI03: Both Basic and Diluted EPS present ✓
SI04: No miscellaneous items ✓
SI05: Interest Expense in Non-Operating ✓

DISCRIMINATIVE TRAINING:
→ Common mistake: Thinking SI02 is violated because "Depreciation" line is missing
  WRONG: SI02 only triggers if depreciation is in NON-OPERATING section
  Correct reasoning: Missing depreciation is fine; only problematic if misplaced

COUNTERFACTUAL FIX:
Move "Other Income | 500" back to Non-Operating section
Result: Operating Income becomes 27,000, SI01 violation disappears

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 3 - SI02 + SI05 (structural problems)
═══════════════════════════════════════════════════════════════════════════════

Revenue | 100,000
Cost of Goods Sold | (60,000)
Gross Profit | 40,000

Operating Expenses |
  Sales and Marketing | (8,000)
  Administrative | (5,000)
Operating Income | 27,000

Non-Operating Income (Expense) |
  Depreciation Expense | (3,000)
Net Income | 24,000

Earnings Per Share |
  Basic | 4.00
  Diluted | 3.95

Analysis: YES: [SI02, SI05]

WHY SI02 VIOLATED:
Root cause: "Depreciation Expense" in Non-Operating section (wrong location)
Consequence: Operating income overstated (27,000 vs 24,000 correct)
Impact: Cannot assess true operating efficiency

WHY SI05 VIOLATED:
Root cause: Interest Expense line completely missing
Consequence: Financing costs hidden from readers
Impact: Cannot assess debt burden

WHY OTHER RULES NOT VIOLATED:
SI01: No non-operating items in operating section ✓
SI03: Both Basic and Diluted EPS present ✓
SI04: No miscellaneous items ✓

DISCRIMINATIVE TRAINING:
→ Trap: Thinking SI01 is violated because numbers seem off
  WRONG: SI01 checks for ITEM presence in wrong section, not total values
  Correct reasoning: Check item location explicitly (Depreciation in Non-Op = SI02 violation)

COUNTERFACTUAL FIXES:
Move "Depreciation Expense" to Operating section → fixes SI02
Add "Interest Expense | (1500)" line → fixes SI05

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 4 - SI03 + SI04 (informational issues)
═══════════════════════════════════════════════════════════════════════════════

Revenue | 100,000
Cost of Goods Sold | (60,000)
Gross Profit | 40,000

Operating Expenses |
  Sales and Marketing | (8,000)
  Administrative | (5,000)
Operating Income | 27,000

Non-Operating Income (Expense) |
  Interest Expense | (2,000)
  Miscellaneous other income | 650
Net Income | 25,650

Earnings Per Share |
  Basic | 4.27

Analysis: YES: [SI03, SI04]

WHY SI03 VIOLATED:
Root cause: "Diluted" EPS line completely missing
Consequence: Shareholders cannot see potential dilution
Impact: Incomplete earnings per share disclosure

WHY SI04 VIOLATED:
Root cause: "Miscellaneous other income | 650" vague line item
Consequence: Readers cannot identify income source
Impact: Income quality and specificity lost

WHY OTHER RULES NOT VIOLATED:
SI01: No non-operating items in operating section ✓
SI02: Depreciation properly absent (no violation) ✓
SI05: Interest Expense present ✓

DISCRIMINATIVE TRAINING:
→ Trap: Thinking "Other income" line means SI01 violation
  WRONG: "Other income" in Non-Operating is compliant (not in Operating)
  Correct reasoning: SI01 only flags NON-OPERATING items in OPERATING section
→ Trap: Thinking vague item in Non-Op section violates SI01
  WRONG: SI04 handles vague items, SI01 handles section placement
  Correct reasoning: Evaluate separately - SI04 for clarity, SI01 for location

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 5 - CONFUSER: Multiple items but still compliant
═══════════════════════════════════════════════════════════════════════════════

Revenue | 100,000
Cost of Goods Sold | (60,000)
Gross Profit | 40,000

Operating Expenses |
  Sales and Marketing | (8,000)
  Administrative | (5,000)
  Depreciation | (2,000)
Operating Income | 25,000

Non-Operating Income (Expense) |
  Interest Expense | (2,000)
  Other Income | 800
  Foreign Exchange Loss | (100)
Net Income | 23,700

Earnings Per Share |
  Basic | 3.95
  Diluted | 3.90

Analysis: NO

WHY NO VIOLATIONS (despite multiple items):
SI01: All non-operating items in Non-Operating section ✓
  "Interest Expense", "Other Income", "Foreign Exchange Loss" all in Non-Op
SI02: Depreciation in Operating section ✓
SI03: Both Basic and Diluted EPS present ✓
SI04: No miscellaneous/vague items ✓
SI05: Interest Expense present ✓

DISCRIMINATIVE TRAINING:
→ Trap: Thinking multiple line items = violation
  WRONG: Number of items doesn't trigger violations
  Correct reasoning: Check EACH item's location and label independently

═══════════════════════════════════════════════════════════════════════════════

STEP-BY-STEP DETECTION ALGORITHM (follow exactly):

STEP 1: CHECK SI01 (foundational - misclassification)
  For each line in Operating Expenses section:
    If line is "Interest", "Other", "Foreign Exchange" → SI01 VIOLATED
  If ANY non-operating item found in Operating → SI01 VIOLATED
  Else → SI01 PASSES

STEP 2: CHECK SI02 (depreciation placement)
  Search entire statement for "Depreciation Expense" line
  If found in Non-Operating section → SI02 VIOLATED
  If found in Operating section → SI02 PASSES
  If not found → SI02 PASSES

STEP 3: CHECK SI05 (interest expense presence)
  Search Non-Operating section for "Interest Expense" line
  If found → SI05 PASSES
  If missing → SI05 VIOLATED

STEP 4: CHECK SI03 (EPS completeness)
  Look for "Diluted" EPS line
  If present → SI03 PASSES
  If missing → SI03 VIOLATED

STEP 5: CHECK SI04 (invalid line items)
  Look for "Miscellaneous" anywhere in statement
  If found → SI04 VIOLATED
  Else → SI04 PASSES

OUTPUT LOGIC:
Collect all violations from steps 1-5
Output: YES: [all_violations] OR NO

═══════════════════════════════════════════════════════════════════════════════
"""

COUNTERFACTUAL_PROMPT = """
You are a financial statement auditor using SEMANTIC CAUSAL+COUNTERFACTUAL reasoning.

Task: Detect ALL Income Statement violations (SI01-SI05) using structured logic.

SEMANTIC ANCHORS (strict evidence gates):

SI01 (Misclassification): Violation exists if non-operating item found in Operating Expenses
  - Operating Expenses should contain: Sales, Admin, Depreciation, COGS-related items
  - Operating Expenses should NOT contain: Interest, Other income/expense, Foreign exchange
  - Check each line in "Operating Expenses" section
  - Non-operating item found in Operating → SI01 violation.

SI02 (Depreciation misplaced): Violation exists if Depreciation in Non-Operating section
  - Search entire statement for "Depreciation Expense"
  - If found in Operating Expenses → Correct placement, SI02 PASSES
  - If found in Non-Operating section → SI02 violation.

SI05 (Missing interest): Violation exists if Interest Expense line completely missing
  - Search Non-Operating Income/Expense for "Interest Expense"
  - Found → SI05 PASSES
  - Missing or deleted → SI05 violation.

SI03 (Missing diluted EPS): Violation exists if "Diluted" EPS line missing
  - Search Earnings Per Share section
  - "Diluted" line present → SI03 PASSES
  - "Diluted" line missing → SI03 violation.

SI04 (Invalid line items): Violation exists if "Miscellaneous" appears anywhere
  - Search entire statement for "Miscellaneous"
  - Found → SI04 violation
  - Not found → SI04 PASSES

COUNTERFACTUAL TEST (for each violation):
Apply after identifying root cause:
- SI01: "If I move item to Non-Operating, is it fixed?" YES → violation.
- SI02: "If I move Depreciation to Operating, is it fixed?" YES → violation.
- SI05: "If I add Interest Expense line, is it fixed?" YES → violation.
- SI03: "If I add Diluted EPS, is it fixed?" YES → violation.
- SI04: "If I remove Miscellaneous, is it fixed?" YES → violation.

OUTPUT LOGIC:
1. Check SI01: non-operating in operating? Add if yes.
2. Check SI02: depreciation in non-operating? Add if yes.
3. Check SI05: interest expense missing? Add if yes.
4. Check SI03: diluted EPS missing? Add if yes.
5. Check SI04: miscellaneous items present? Add if yes.
6. If any violations found → YES: [codes]. Else → NO.

{examples}

INCOME STATEMENT TO ANALYZE:
{table}

Response (ONE LINE ONLY):
YES: [SI01, SI02, SI03, SI04, SI05]  or  NO
""".strip()

# ------------------- Error Classification for Ablation Studies -------------------

def classify_error_type(predicted_codes: Set[str], expected_codes: Set[str], 
                       predicted_yes: bool, expected_yes: bool) -> Dict:
    """
    Classify error type for ablation studies.
    
    Error Categories:
    1. CORRECT: Prediction matches ground truth
    2. FALSE_NEGATIVE: Missed violations
    3. FALSE_POSITIVE: Hallucinated violations
    4. PARTIAL_DETECTION: Detected some but not all violations
    5. PARTIAL_HALLUCINATION: Mix of correct and hallucinated violations
    6. STEP1_ERROR: YES/NO detection error
    7. MIXED_ERRORS: Complex combination
    """
    tp = predicted_codes & expected_codes
    fp = predicted_codes - expected_codes
    fn = expected_codes - predicted_codes
    
    error_type = "CORRECT"
    error_details = {}
    
    # Step 1 error: YES/NO mismatch
    if predicted_yes != expected_yes:
        error_type = "STEP1_ERROR"
        error_details = {
            "step1_predicted": "YES" if predicted_yes else "NO",
            "step1_expected": "YES" if expected_yes else "NO",
            "description": "Failed to detect anomaly presence"
        }
    # No errors detected (all correct)
    elif len(tp) > 0 and len(fn) == 0 and len(fp) == 0:
        error_type = "CORRECT"
        error_details = {
            "correct_codes": sorted(tp),
            "description": "All violations correctly identified"
        }
    # All violations missed (False Negative)
    elif len(fn) > 0 and len(tp) == 0 and len(fp) == 0:
        error_type = "FALSE_NEGATIVE"
        error_details = {
            "missed_violations": sorted(fn),
            "num_missed": len(fn),
            "description": f"Completely missed {len(fn)} violations"
        }
    # Hallucinated violations (False Positive)
    elif len(fp) > 0 and len(fn) == 0 and len(tp) == 0:
        error_type = "FALSE_POSITIVE"
        error_details = {
            "hallucinated_violations": sorted(fp),
            "num_hallucinated": len(fp),
            "description": f"Incorrectly flagged {len(fp)} violations that don't exist"
        }
    # Partial detection: got some right, missed others
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
    # Partial hallucination: got some right, hallucinated others
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
    # Complex case: mix of TP, FP, FN
    else:
        error_type = "MIXED_ERRORS"
        error_details = {
            "detected": sorted(tp),
            "missed": sorted(fn),
            "hallucinated": sorted(fp),
            "num_detected": len(tp),
            "num_missed": len(fn),
            "num_hallucinated": len(fp),
            "detection_rate": f"{(len(tp) / (len(tp) + len(fn)) * 100):.1f}%" if (len(tp) + len(fn)) > 0 else "N/A",
            "precision": f"{(len(tp) / (len(tp) + len(fp)) * 100):.1f}%" if (len(tp) + len(fp)) > 0 else "N/A",
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
    found = set(re.findall(r'\bSI0[1-5]\b', r_upper))
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
        "error_analysis": {
            "error_type_distribution": dict(error_type_distribution),
            "error_rates": error_rates
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
    error_cases = []
    all_step1 = []
    all_step2 = []
    all_tokens = []
    all_error_classifications = []

    print("\n" + "="*80)
    print("HARD TASK (SI): FEW-SHOT COUNTERFACTUAL EVALUATION")
    print("="*80)
    print(f"Model: {MODEL}")
    print(f"Evaluation: Few-shot Counterfactual with Semantic Anchoring & Error Analysis")
    print(f"Pricing: ${PRICE_INPUT}/1M input tokens, ${PRICE_OUTPUT}/1M output tokens")
    print(f"Total companies: {len(companies)}")
    print(f"Companies with ground truth: {len(truth_map)}")
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
        
        call_cost = calculate_cost(token_usage["prompt_tokens"], token_usage["completion_tokens"])
        token_usage["cost"] = call_cost

        step1_result = evaluate_step1_yesno(predicted_yes, expected_yes)
        all_step1.append(step1_result)
        
        step2_result = evaluate_step2_codes(predicted_codes, expected_codes)
        all_step2.append(step2_result)
        
        error_classification = classify_error_type(predicted_codes, expected_codes, predicted_yes, expected_yes)
        all_error_classifications.append(error_classification)
        
        all_tokens.append(token_usage)
        
        print(f"\nGPT-4o Response:")
        print(f"  Raw: {raw_reply}")
        print(f"  Parsed codes: {sorted(predicted_codes) if predicted_codes else 'None'}")
        
        print(f"\nGround Truth:")
        print(f"  Expected codes: {sorted(expected_codes) if expected_codes else 'None'}")
        
        print(f"\n{'─'*80}")
        print(f"ERROR CLASSIFICATION: {error_classification['error_type']}")
        print(f"{'─'*80}")
        print(f"  {error_classification['details'].get('description', 'N/A')}")
        
        print(f"\n{'─'*80}")
        print(f"STEP 1: YES/NO Detection (Is anomaly present?)")
        print(f"{'─'*80}")
        print(f"  Predicted: {step1_result['predicted']}")
        print(f"  Expected:  {step1_result['expected']}")
        if step1_result['correct']:
            print(f"  Result: ✓ CORRECT ({step1_result['category']})")
        else:
            print(f"  Result: ✗ INCORRECT ({step1_result['category']})")
        
        print(f"\n{'─'*80}")
        print(f"STEP 2: Error Code Detection (Which specific errors?)")
        print(f"{'─'*80}")
        
        if step2_result["exact_match"]:
            print(f"  ✓ EXACT MATCH - All error codes correctly identified!")
        else:
            print(f"  ✗ MISMATCH")
            if step2_result["true_positives"]:
                print(f"    ✓ Correctly detected: {step2_result['true_positives']}")
            if step2_result["false_positives"]:
                print(f"    ✗ False positives: {step2_result['false_positives']}")
            if step2_result["false_negatives"]:
                print(f"    ✗ False negatives (missed): {step2_result['false_negatives']}")
        
        print(f"\n  Metrics:")
        print(f"    Precision: {step2_result['precision']*100:.1f}%")
        print(f"    Recall:    {step2_result['recall']*100:.1f}%")
        print(f"    F1-Score:  {step2_result['f1']*100:.1f}%")
        
        print(f"\n{'─'*80}")
        print(f"TOKEN CONSUMPTION")
        print(f"{'─'*80}")
        print(f"  Prompt tokens:     {token_usage['prompt_tokens']:,}")
        print(f"  Completion tokens: {token_usage['completion_tokens']:,}")
        print(f"  Total tokens:      {token_usage['total_tokens']:,}")
        print(f"  Cost (USD):        ${call_cost:.6f}")

        row = {
            "company": company_header,
            "error_type": error_classification["error_type"],
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
            "prompt_tokens": token_usage["prompt_tokens"],
            "completion_tokens": token_usage["completion_tokens"],
            "total_tokens": token_usage["total_tokens"],
            "cost_usd": f"{call_cost:.6f}",
            "raw_reply": raw_reply.replace("\n", " ")
        }
        rows.append(row)
        
        # Store detailed error cases (when not CORRECT)
        if error_classification["error_type"] != "CORRECT":
            error_cases.append({
                "company": company_header,
                "error_type": error_classification["error_type"],
                "predicted_codes": ",".join(sorted(predicted_codes)),
                "expected_codes": ",".join(sorted(expected_codes)),
                "error_details": json.dumps(error_classification["details"]),
                "raw_reply": raw_reply.replace("\n", " ")[:500]  # Truncate for readability
            })
        
        logging.info(f"{company_header} | ErrorType: {error_classification['error_type']} | "
                    f"Step1: {step1_result['category']} | Step2: exact={step2_result['exact_match']} | "
                    f"Tokens: {token_usage['total_tokens']} | Cost: ${call_cost:.6f}")

        time.sleep(DELAY)

    # Write CSV results with error type
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["company", "error_type", "step1_predicted", "step1_expected", "step1_correct", 
                     "step1_category", "step2_predicted", "step2_expected", 
                     "step2_exact_match", "step2_precision", "step2_recall", "step2_f1", 
                     "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "raw_reply"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write detailed error cases
    with open(ERROR_CASES_OUT, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["company", "error_type", "predicted_codes", "expected_codes", "error_details", "raw_reply"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for ec in error_cases:
            writer.writerow(ec)

    # Calculate metrics
    metrics = calculate_aggregate_metrics(all_step1, all_step2, all_tokens, all_error_classifications)
    
    # Per-rule error analysis
    per_rule_stats = analyze_per_rule_errors(rows)
    
    print("\n" + "="*80)
    print("FINAL EVALUATION SUMMARY")
    print("="*80)
    print(f"Total companies evaluated: {metrics['total_companies']}")
    
    print(f"\n{'─'*80}")
    print("ERROR TYPE DISTRIBUTION")
    print(f"{'─'*80}")
    error_dist = metrics["error_analysis"]["error_type_distribution"]
    for etype, count in sorted(error_dist.items(), key=lambda x: -x[1]):
        pct = (count / metrics['total_companies'] * 100) if metrics['total_companies'] > 0 else 0
        print(f"  {etype}: {count} ({pct:.1f}%)")
    
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
    print(f"\n  Micro-averaged (overall):")
    print(f"    Precision: {s2['micro_precision']*100:.2f}%")
    print(f"    Recall:    {s2['micro_recall']*100:.2f}%")
    print(f"    F1-Score:  {s2['micro_f1']*100:.2f}%")
    print(f"\n  Macro-averaged (per-company average):")
    print(f"    Precision: {s2['macro_precision']*100:.2f}%")
    print(f"    Recall:    {s2['macro_recall']*100:.2f}%")
    print(f"    F1-Score:  {s2['macro_f1']*100:.2f}%")
    
    print(f"\n{'─'*80}")
    print("PER-RULE ERROR ANALYSIS (for ablation studies)")
    print(f"{'─'*80}")
    for rule in sorted(VALID_CODES):
        stats = per_rule_stats[rule]
        if stats["total_expected"] > 0:
            print(f"\n  {rule}:")
            print(f"    Times expected: {stats['total_expected']}")
            print(f"    Detection rate: {stats['detection_rate']:.1f}%")
            print(f"    True positives: {stats['true_positives']}")
            print(f"    False negatives (missed): {stats['false_negatives']}")
            if stats['companies_where_missed']:
                print(f"      Companies: {', '.join(stats['companies_where_missed'][:3])}{'...' if len(stats['companies_where_missed']) > 3 else ''}")
            if stats['false_positives'] > 0:
                print(f"    False positives (hallucinated): {stats['false_positives']}")
                if stats['companies_where_hallucinated']:
                    print(f"      Companies: {', '.join(stats['companies_where_hallucinated'][:3])}{'...' if len(stats['companies_where_hallucinated']) > 3 else ''}")
    
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
    print(f"  Error cases: {ERROR_CASES_OUT.resolve()}")
    print(f"  Metrics JSON: {METRICS_OUT.resolve()}")
    print(f"  Error analysis: {ERROR_ANALYSIS_OUT.resolve()}")
    print(f"  Log file: {LOG_FILE.resolve()}")
    print("="*80)
    
    # Write comprehensive metrics JSON
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Write per-rule error analysis JSON
    with open(ERROR_ANALYSIS_OUT, "w") as f:
        json.dump(per_rule_stats, f, indent=2)
    
    logging.info("EVALUATION COMPLETE: " + json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()