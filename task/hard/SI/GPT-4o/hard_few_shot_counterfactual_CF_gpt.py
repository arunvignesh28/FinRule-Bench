#!/usr/bin/env python3
"""
hard_few_shot_counterfactual_CF_gpt.py

Few-shot with counterfactual explanations for hard multi-error detection task (Cash Flow Statements).
Uses contrastive examples showing causes, effects, and counterfactuals.

Inputs:
 - CF-MIXED-100.md            (mixed dataset: 20% clean, 80% anomaly)
 - CF-MIXED-100-truth.json    (ground-truth JSON)

Outputs:
 - few_shot_cf_eval_results.csv              (with error_type column)
 - few_shot_cf_eval_summary.log
 - few_shot_cf_eval_metrics.json             (with error analysis)
 - few_shot_cf_eval_error_analysis.json      (error type distribution for ablation)
 - few_shot_cf_eval_error_cases.csv          (detailed error cases)

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
INPUT_MD = Path("CF-MIXED-100.md")
INPUT_TRUTH = Path("CF-MIXED-100-truth.json")

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
PRICE_OUTPUT = 10.00   # $10.00 per 1M output tokens

VALID_CODES = {"CF01", "CF02", "CF03"}

client = OpenAI()

# ------------------- Few-Shot Examples with Counterfactuals -------------------
FEW_SHOT_COUNTERFACTUAL = """
STRUCTURED SEMANTIC REASONING: CAUSAL + COUNTERFACTUAL DETECTION

DECISION LOGIC:
1. Output YES only if at least one violation has explicit evidence (not suspicion)
2. MUST check all three rules (CF01, CF02, CF03) before outputting
3. MUST list ALL violations found (multi-label)

CAUSAL REASONING: Identify root cause of violation
- What element is MISSING, PROHIBITED, or WRONG?
- Why does this violate GAAP (chain: cause → effect → impact)?

COUNTERFACTUAL REASONING: Propose minimal fix
- What specific change removes this violation?
- Can I state exactly what to add/remove/replace?

VIOLATION DEFINITIONS:

CF01 Root Cause: Missing activity section header
Evidence: Absence of "Cash flows from operating/investing/financing activities"
Causal Chain: Missing section → Cannot see cash breakdown → Incomplete financial picture → Audit fails
Counterfactual: "If I add the missing section with line items, is it fixed?" YES=violation

CF02 Root Cause: Non-GAAP metric present
Evidence: Line contains "per share" + cash flow terminology
Causal Chain: Non-GAAP metric → Investor confusion → Regulatory violation
Counterfactual: "If I delete this line, is it fixed?" YES=violation

CF03 Root Cause: Non-standard cash terminology
Evidence: Cash balance line uses "Funds", "Resources", "Capital", "Working Capital", etc.
Causal Chain: Imprecise term → Reader ambiguity → Misinterpretation
Counterfactual: "If I change to 'Cash and cash equivalents', is it fixed?" YES=violation

DETECTION PROCEDURE:
1. CF01: Scan statement. Are all THREE activity section headers present with content?
   If ANY header absent or empty → flag CF01
2. CF02: Search for "per share" related to cash flow. If found → flag CF02
3. CF03: Find end-of-period cash line. Is it "Cash and cash equivalents" or "Restricted cash"?
   If NOT → flag CF03
4. Output all violations (CF01, CF02, CF03) found via evidence gates

EXAMPLES:

Example 1: No violations
Table has: Operating ✓, Investing ✓, Financing ✓ sections with items
           "Cash and cash equivalents" terminology ✓
           No "per share" metrics ✓
Analysis: NO

Example 2: CF01 + CF03
Table missing: "Cash flows from investing activities" (root cause → incomplete picture)
Table has: "Resources and equivalents" instead of "Cash and cash equivalents"
Analysis: YES: [CF01, CF03]
Fixes: (1) Add investing activities section, (2) Change terminology to "Cash and cash equivalents"

Example 3: CF02 + CF03
Table has: "Operating cash flow per share" line (non-GAAP → regulatory violation)
Table has: "Funds and equivalents" (imprecise → confusion)
Analysis: YES: [CF02, CF03]
Fixes: (1) Delete per-share line, (2) Change "Funds" to "Cash and cash equivalents"

Example 4: CF01 only
Table missing: "Cash flows from investing activities"
Table has: All other sections with content and correct terminology
Analysis: YES: [CF01]
Fix: Add investing activities section

MANDATORY RULES:
- Never output YES without explicit evidence (no suspicion, ambiguity, or formatting oddities)
- Always check all three rules (CF01 → CF02 → CF03) before deciding
- Always output ALL violations found in one line
- If you can propose a counterfactual fix, the violation is real
"""

COUNTERFACTUAL_PROMPT = """
You are a financial auditor using SEMANTIC CAUSAL+COUNTERFACTUAL reasoning.

Task: Detect ALL Cash Flow violations (CF01, CF02, CF03) using structured logic.

SEMANTIC ANCHORS (strict evidence gates):

CF01 (Missing section): Violation exists if any section header is absent or has zero line items
  - Operating activities header with line items? Check.
  - Investing activities header with line items? Check.
  - Financing activities header with line items? Check.
  - Missing any? → CF01 violation.

CF02 (Non-GAAP metric): Violation exists if "per share" metric found for cash flow
  - Search for "cash flow per share", "CFO per share", "operating cash flow per share", etc.
  - Found? → CF02 violation.

CF03 (Wrong terminology): Violation exists if cash balance uses non-GAAP terms
  - Find end-of-period cash balance line.
  - Is it "Cash and cash equivalents" or "Restricted cash"? Safe.
  - Uses "Funds", "Resources", "Capital", "Working Capital", "Financial position", etc.? → CF03 violation.

COUNTERFACTUAL TEST (for each violation):
Apply this test AFTER you identify the root cause:
- CF01: "If I add the missing section with proper line items, is it fixed?" YES → violation.
- CF02: "If I delete the per-share line, is it fixed?" YES → violation.
- CF03: "If I change to 'Cash and cash equivalents', is it fixed?" YES → violation.

OUTPUT LOGIC:
1. Check CF01: violation present? Add to output.
2. Check CF02: violation present? Add to output.
3. Check CF03: violation present? Add to output.
4. If any violation found → output YES: [codes]. Else → NO.

{examples}

CASH FLOW STATEMENT TO ANALYZE:
{table}

Response (ONE LINE ONLY):
YES: [CF01, CF02, CF03]  or  NO
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
    """
    Analyze errors by rule type for ablation studies.
    
    For each rule, track:
    - How many times it was missed (FN)
    - How many times it was hallucinated (FP)
    - Detection accuracy by rule
    """
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
        # Extract predicted and expected codes from CSV row format
        predicted_str = result.get("step2_predicted", "")
        expected_str = result.get("step2_expected", "")
        company = result.get("company", "Unknown")
        
        # Parse comma-separated codes
        predicted = set(c.strip() for c in predicted_str.split(",") if c.strip() and c.strip() in VALID_CODES)
        expected = set(c.strip() for c in expected_str.split(",") if c.strip() and c.strip() in VALID_CODES)
        
        # Track per-rule stats
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
        
        # Calculate metrics
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
    """
    Parse model reply for YES/NO and list of codes.
    Returns (is_yes, set_of_codes, raw_reply)
    """
    r = reply.strip()
    r_upper = r.upper()
    is_yes = r_upper.startswith("YES")
    found = set(re.findall(r'\bCF0[1-3]\b', r_upper))
    return is_yes, found, reply

def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD using gpt-4o pricing."""
    input_cost = (prompt_tokens / 1_000_000) * PRICE_INPUT
    output_cost = (completion_tokens / 1_000_000) * PRICE_OUTPUT
    return input_cost + output_cost

# ------------------- Two-Step Evaluation -------------------

def evaluate_step1_yesno(predicted_yes: bool, expected_yes: bool) -> Dict:
    """Step 1: Evaluate YES/NO detection (is anomaly present?)."""
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
    """Step 2: Evaluate error code detection (which specific errors?)."""
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
        "num_fn": len(false_negatives)
    }

def calculate_aggregate_metrics(all_step1: List[Dict], all_step2: List[Dict], all_tokens: List[Dict], 
                               all_error_classifications: List[Dict] = None) -> Dict:
    """Calculate aggregate metrics across all companies including error analysis."""
    
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
    
    metrics = {
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
        }
    }
    
    # Add error analysis if provided
    if all_error_classifications:
        error_type_dist = defaultdict(int)
        for ec in all_error_classifications:
            error_type_dist[ec["error_type"]] += 1
        
        error_analysis = {
            "error_type_distribution": dict(error_type_dist),
            "error_rates": {
                etype: f"{(count / total * 100):.1f}%" 
                for etype, count in error_type_dist.items()
            }
        }
        metrics["error_analysis"] = error_analysis
    
    return metrics

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
    print("HARD TASK (CF): FEW-SHOT COUNTERFACTUAL EVALUATION WITH ERROR ANALYSIS")
    print("="*80)
    print(f"Model: {MODEL}")
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
        print(f"ERROR CLASSIFICATION")
        print(f"{'─'*80}")
        print(f"  Type: {error_classification['error_type']}")
        print(f"  Details: {error_classification['details']}")
        
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
                    f"Error: {error_classification['error_type']} | "
                    f"Tokens: {token_usage['total_tokens']} | Cost: ${call_cost:.6f}")

        time.sleep(DELAY)

    # Write main CSV results
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

    # Write error cases CSV
    if error_cases:
        with open(ERROR_CASES_OUT, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["company", "error_type", "expected_codes", "predicted_codes", "step1_correct", "details"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for ec in error_cases:
                writer.writerow(ec)

    metrics = calculate_aggregate_metrics(all_step1, all_step2, all_tokens, all_error_classifications)
    per_rule_stats = analyze_per_rule_errors(rows)
    metrics["per_rule_analysis"] = per_rule_stats
    
    print("\n" + "="*80)
    print("FINAL EVALUATION SUMMARY")
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
    print(f"    True Positives:  {s1['confusion_matrix']['TP']} (correctly detected anomaly)")
    print(f"    True Negatives:  {s1['confusion_matrix']['TN']} (correctly identified no anomaly)")
    print(f"    False Positives: {s1['confusion_matrix']['FP']} (false alarm)")
    print(f"    False Negatives: {s1['confusion_matrix']['FN']} (missed anomaly)")
    
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
    print(f"\n  Error Detection Statistics:")
    print(f"    True Positives:  {s2['total_tp']} (correctly identified errors)")
    print(f"    False Positives: {s2['total_fp']} (incorrectly flagged)")
    print(f"    False Negatives: {s2['total_fn']} (missed errors)")
    
    print(f"\n{'─'*80}")
    print("ERROR TYPE DISTRIBUTION")
    print(f"{'─'*80}")
    error_analysis = metrics.get("error_analysis", {})
    if error_analysis:
        for error_type, count in sorted(error_analysis["error_type_distribution"].items()):
            rate = error_analysis["error_rates"].get(error_type, "N/A")
            print(f"  {error_type}: {count} ({rate})")
    
    print(f"\n{'─'*80}")
    print("PER-RULE ANALYSIS")
    print(f"{'─'*80}")
    for rule in sorted(VALID_CODES):
        stats = per_rule_stats[rule]
        det_rate = stats["detection_rate"]
        prec = stats["precision"]
        print(f"  {rule}:")
        print(f"    Detection rate: {det_rate:.1f}% ({stats['true_positives']}/{stats['total_expected']})")
        print(f"    Precision:      {prec:.1f}%")
        if stats["companies_where_missed"]:
            print(f"    Missed in:      {', '.join(stats['companies_where_missed'][:3])}")
        if stats["companies_where_hallucinated"]:
            print(f"    Hallucinated in: {', '.join(stats['companies_where_hallucinated'][:3])}")
    
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
    print(f"  CSV results:           {CSV_OUT.resolve()}")
    print(f"  Error cases CSV:       {ERROR_CASES_OUT.resolve()}")
    print(f"  Metrics JSON:          {METRICS_OUT.resolve()}")
    print(f"  Error analysis JSON:   {ERROR_ANALYSIS_OUT.resolve()}")
    print(f"  Log file:              {LOG_FILE.resolve()}")
    print("="*80)
    
    # Write metrics JSON
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Write error analysis JSON
    error_analysis_data = {
        "error_type_distribution": metrics.get("error_analysis", {}).get("error_type_distribution", {}),
        "error_rates": metrics.get("error_analysis", {}).get("error_rates", {}),
        "per_rule_analysis": per_rule_stats
    }
    with open(ERROR_ANALYSIS_OUT, "w") as f:
        json.dump(error_analysis_data, f, indent=2)
    
    logging.info("EVALUATION COMPLETE: " + json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()