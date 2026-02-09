#!/usr/bin/env python3
"""
zero_shot_easy_BS_gemini_evaluation.py

Comprehensive evaluation on balance sheet files using Gemini 2.5 Pro.
- BS-00.md: COMPLIANT (Ground truth: T)
- BS-01 to BS-05.md: ERROR-INJECTED (Ground truth: F each)

Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Token tracking
"""

import time
import logging
import re
import csv
import os
import sys
from pathlib import Path
from google import genai  # Correct import for 2026 SDK

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not set!")
    print("Set it using: export GEMINI_API_KEY='your_key_here'")
    sys.exit(1)

# Initialize Client (Replaces genai.configure)
client = genai.Client(api_key=GEMINI_API_KEY)

MODEL = "gemini-2.5-pro"
TEMPERATURE = 0.0

# Pricing for Gemini 2.5 Pro (in USD per million tokens)
PRICE_INPUT = 1.25     # $1.25 per 1M input tokens
PRICE_OUTPUT = 10.00     # $10.00 per 1M output tokens

RULES = {
    "BS00": {
        "filename": "new_BS/BS-00.md",
        "ground_truth": "T",  # Compliant file
        "prompt": """You are a strict financial auditor checking for balance sheet compliance.

RULE:
Total Assets MUST equal (Total Liabilities + Total Stockholders' Equity).
Also check for:
- Exact standard terminology: "Current Assets" and "Current Liabilities" (not variants)
- Standard GAAP cash terminology: "Cash and cash equivalents" (not "Resources"/"Funds")
- Stockholders' Equity MUST contain "Retained earnings"
- Treasury stock ONLY in Equity section, NOT in Assets

If ALL rules are satisfied, answer T. If ANY rule is violated, answer F.

BALANCE SHEET DATA:
{table_data}

Respond ONLY with 'T' or 'F':"""
    },
    "BS01": {
        "filename": "new_BS/BS-01.md",
        "ground_truth": "F",  # Error-injected
        "prompt": """You are a strict financial auditor checking for numeric balance sheet errors.

RULE:
Total Assets MUST equal (Total Liabilities + Total Stockholders' Equity).
Calculate: Assets - (Liabilities + Equity). If result = 0, answer T. If not, answer F.

Key points:
- Include mezzanine interests and non-controlling interests in equity total
- Check BOTH years if multiple years present
- Answer based on whether the fundamental equation holds

BALANCE SHEET DATA:
{table_data}

Respond ONLY with 'T' or 'F':"""
    },
    "BS02": {
        "filename": "new_BS/BS-02.md",
        "ground_truth": "F",
        "prompt": """You are a strict financial auditor checking for classification errors.

RULE:
Both sections MUST use exact standard terminology:
- Assets section: "Current Assets" (not "Current_Assets", "Operating_Assets", "Present_Assets", or variants)
- Liabilities section: "Current Liabilities" (not "Current_Liabilities", variations)

Common violations to detect:
- "Operating_Assets" / "Operating_Liabilities" instead of "Current"
- "Present_Assets" / "Present_Liabilities" instead of "Current"  
- "Primary_Liabilities" instead of "Current"
- Any non-standard adjective replacing "Current"

BALANCE SHEET DATA:
{table_data}

Respond ONLY with 'T' or 'F':"""
    },
    "BS03": {
        "filename": "new_BS/BS-03.md",
        "ground_truth": "F",
        "prompt": """You are a strict financial auditor checking for cash terminology errors.

RULE:
The balance sheet MUST use standard GAAP terminology for cash:
- Acceptable: "Cash and cash equivalents", "Restricted cash"
- NOT acceptable: "Resources", "Capital", "Funds", "Financial Resources"

Common violations to detect:
- "Resources and cash equivalents" (wrong word for "Cash")
- "Capital and cash equivalents" (wrong word for "Cash")
- "Funds held" (should be "Cash")
- "Working Capital" (not standard cash term)
- Any replacement of the word "Cash" with ambiguous synonyms

BALANCE SHEET DATA:
{table_data}

Respond ONLY with 'T' or 'F':"""
    },
    "BS04": {
        "filename": "new_BS/BS-04.md",
        "ground_truth": "F",
        "prompt": """You are a strict financial auditor checking for retained earnings labeling errors.

RULE:
Stockholders' Equity MUST contain a line labeled exactly "Retained earnings".
NOT acceptable: "General Reserve", "Accumulated Capital", "Capital Surplus"

Common violations to detect:
- "General Reserve" instead of "Retained earnings"
- "Accumulated Capital" instead of "Retained earnings"
- "Capital Surplus" instead of "Retained earnings"
- Complete absence of "Retained earnings" line item
- Any non-standard label for accumulated undistributed profits

BALANCE SHEET DATA:
{table_data}

Respond ONLY with 'T' or 'F':"""
    },
    "BS05": {
        "filename": "new_BS/BS-05.md",
        "ground_truth": "F",
        "prompt": """You are a strict financial auditor checking for treasury stock placement errors.

RULE:
"Treasury stock" MUST appear ONLY in the Stockholders' Equity section as a deduction (negative value).
Violation: Treasury stock appears in the Assets section.

Common violations to detect:
- "Treasury stock" line item in Assets section (with negative value)
- Missing "Treasury stock" from Equity when it should be there
- Treasury stock presented in wrong location (Assets vs Equity)

Check structure:
- Look at the "Section" column: Should say "Equity", not "Assets"
- If treasury stock appears under Assets section, it violates the rule

BALANCE SHEET DATA:
{table_data}

Respond ONLY with 'T' or 'F':"""
    },
}

def parse_markdown_file(filepath: Path) -> list:
    """Parse markdown file to extract balance sheet tables."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tables = []
    lines = content.split('\n')
    current_company = None
    current_date = None
    table_lines = []
    in_table = False
    
    for line in lines:
        if line.startswith('## ') or line.startswith('### '):
            if table_lines and current_company:
                tables.append({
                    'company_name': current_company,
                    'date': current_date,
                    'data': '\n'.join(table_lines)
                })
                table_lines = []
            
            header = line.lstrip('#').strip()
            if '—' in header:
                parts = header.split('—')
                current_company = parts[0].strip()
                current_date = parts[1].strip()
            else:
                current_company = header
                current_date = ""
            in_table = False
        
        elif line.strip().startswith('|'):
            in_table = True
            table_lines.append(line)
        elif in_table and line.strip() == '':
            in_table = False
        elif in_table:
            table_lines.append(line)
    
    if table_lines and current_company:
        tables.append({
            'company_name': current_company,
            'date': current_date,
            'data': '\n'.join(table_lines)
        })
    
    return tables

def extract_table_data(table_text: str) -> str:
    """Clean up markdown table."""
    lines = table_text.split('\n')
    clean_lines = []
    
    for line in lines:
        if not line.strip():
            continue
        line = line.strip()
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]
        if re.match(r'^-+\s*\|\s*-+', line):
            continue
        cells = [cell.strip() for cell in line.split('|')]
        clean_lines.append(' | '.join(cells))
    
    return '\n'.join(clean_lines)

def call_with_retry(prompt: str, max_attempts: int = 6, base_delay: float = 2.0) -> tuple:
    """Call Gemini API with exponential backoff retry."""
    for attempt in range(1, max_attempts + 1):
        try:
            # Using the client initialized at the top
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={'temperature': TEMPERATURE}
            )
            
            # Extract token usage (New GenAI SDK metadata structure)
            usage = response.usage_metadata
            token_usage = {
                "prompt_tokens": usage.prompt_token_count if usage else 0,
                "completion_tokens": usage.candidates_token_count if usage else 0,
                "total_tokens": usage.total_token_count if usage else 0
            }
            
            return response.text.strip(), token_usage
        
        except Exception as e:
            if attempt == max_attempts:
                raise
            wait = base_delay * (2 ** (attempt - 1))
            time.sleep(wait)

def check_table(table_info: dict, rule_code: str) -> tuple:
    """Send table to LLM and return (tf, token_usage)."""
    table_string = extract_table_data(table_info['data'])
    
    if not table_string.strip():
        raise ValueError("Table data is empty")
    
    prompt = RULES[rule_code]["prompt"].format(table_data=table_string)
    
    raw_reply, token_usage = call_with_retry(prompt)
    
    # Robustly find first T or F
    match = re.search(r'[TF]', raw_reply.upper())
    tf = match.group(0) if match else 'F'
    
    return tf, token_usage

def calculate_metrics(tp, tn, fp, fn):
    """Calculate accuracy, precision, recall, F1-score."""
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }

def main():
    log_file = Path("zero_shot_easy_bs_eval_gemini.log")
    logging.basicConfig(filename=str(log_file), level=logging.INFO, format='%(asctime)s - %(message)s')
    csv_file = Path("zero_shot_easy_bs_eval_results_gemini.csv")
    metrics_file = Path("zero_shot_easy_bs_eval_metrics_gemini.txt")
    
    print("\n" + "=" * 110)
    print("ZERO-SHOT ERROR DETECTION EVALUATION (BS00 COMPLIANT + BS01-BS05 ERROR-INJECTED)")
    print("=" * 110)
    print(f"Model: {MODEL}")
    print(f"Pricing: ${PRICE_INPUT}/1M input tokens, ${PRICE_OUTPUT}/1M output tokens")
    print("=" * 110)
    
    rule_results = {}
    all_rows = []
    
    for rule_code in ["BS00", "BS01", "BS02", "BS03", "BS04", "BS05"]:
        rule_info = RULES[rule_code]
        input_path = Path(rule_info["filename"])
        ground_truth = rule_info["ground_truth"]
        
        print(f"\n{'-' * 110}")
        print(f"Processing: {rule_code} (Ground Truth: {ground_truth})")
        print(f"{'-' * 110}")
        
        if not input_path.exists():
            print(f"ERROR: File not found: {input_path}")
            continue
        
        try:
            tables = parse_markdown_file(input_path)
            if not tables:
                print(f"WARNING: No tables found")
                continue
            gt_label = "COMPLIANT" if ground_truth == "T" else "ERROR-INJECTED"
            print(f"Found {len(tables)} balance sheets ({gt_label})")
        except Exception as e:
            print(f"ERROR: Failed to parse: {e}")
            logging.exception(f"Error parsing {input_path}")
            continue
        
        rule_results[rule_code] = {
            'processed': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
            'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0
        }
        
        for idx, tbl in enumerate(tables, start=1):
            comp = tbl.get("company_name", "Unknown")
            date = tbl.get("date", "")
            
            print(f"  [{idx:3d}/{len(tables)}] {comp:30s} ({date:20s}) GT: {ground_truth}...", end=" ", flush=True)
            rule_results[rule_code]['processed'] += 1
            
            try:
                pred, token_usage = check_table(tbl, rule_code)
                
                rule_results[rule_code]['total_tokens'] += token_usage['total_tokens']
                rule_results[rule_code]['prompt_tokens'] += token_usage['prompt_tokens']
                rule_results[rule_code]['completion_tokens'] += token_usage['completion_tokens']
                
                # Confusion matrix update
                if ground_truth == "T" and pred == "T":
                    rule_results[rule_code]['tn'] += 1
                    status = "✓TN"
                elif ground_truth == "T" and pred == "F":
                    rule_results[rule_code]['fp'] += 1
                    status = "✗FP"
                elif ground_truth == "F" and pred == "F":
                    rule_results[rule_code]['tp'] += 1
                    status = "✓TP"
                else:
                    rule_results[rule_code]['fn'] += 1
                    status = "✗FN"
                
                print(f"{status} Pred: {pred}")
                
                all_rows.append({
                    "rule": rule_code, "company": comp, "date": date, 
                    "ground_truth": ground_truth, "prediction": pred,
                    "tp": 1 if (ground_truth == "F" and pred == "F") else 0,
                    "tn": 1 if (ground_truth == "T" and pred == "T") else 0,
                    "fp": 1 if (ground_truth == "T" and pred == "F") else 0,
                    "fn": 1 if (ground_truth == "F" and pred == "T") else 0,
                    "prompt_tokens": token_usage['prompt_tokens'],
                    "completion_tokens": token_usage['completion_tokens'],
                    "total_tokens": token_usage['total_tokens']
                })
                
                logging.info(f"{rule_code} | {comp} | {date} | GT: {ground_truth} | Pred: {pred}")
                
            except Exception as e:
                print(f"ERROR: {e}")
                logging.error(f"{rule_code} | {comp} | {e}")
            
            time.sleep(1.5)
        
        stats = rule_results[rule_code]
        if stats['processed'] > 0:
            metrics = calculate_metrics(stats['tp'], stats['tn'], stats['fp'], stats['fn'])
            print(f"\n  {rule_code} Results:")
            print(f"    Processed: {stats['processed']} | TP: {stats['tp']} | TN: {stats['tn']} | FP: {stats['fp']} | FN: {stats['fn']}")
            print(f"    Accuracy:  {metrics['accuracy']:6.2f}% | Precision: {metrics['precision']:6.2f}% | Recall: {metrics['recall']:6.2f}% | F1: {metrics['f1']:6.2f}%")
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["rule", "company", "date", "ground_truth", "prediction", "tp", "tn", "fp", "fn", "prompt_tokens", "completion_tokens", "total_tokens"])
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"\n" + "=" * 110)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 110)
    
    total_processed = total_tp = total_tn = total_fp = total_fn = 0
    total_tokens = total_prompt = total_completion = 0
    
    for rule_code in ["BS00", "BS01", "BS02", "BS03", "BS04", "BS05"]:
        if rule_code in rule_results:
            stats = rule_results[rule_code]
            total_processed += stats['processed']
            total_tp += stats['tp']
            total_tn += stats['tn']
            total_fp += stats['fp']
            total_fn += stats['fn']
            total_tokens += stats['total_tokens']
            total_prompt += stats['prompt_tokens']
            total_completion += stats['completion_tokens']
            
            if stats['processed'] > 0:
                metrics = calculate_metrics(stats['tp'], stats['tn'], stats['fp'], stats['fn'])
                print(f"{rule_code}: Acc={metrics['accuracy']:5.1f}% | Prec={metrics['precision']:5.1f}% | Rec={metrics['recall']:5.1f}% | F1={metrics['f1']:5.1f}%")
    
    print(f"\n{'-' * 110}")
    print("OVERALL METRICS (All Rules Combined)")
    print(f"{'-' * 110}")
    
    if total_processed > 0:
        overall_metrics = calculate_metrics(total_tp, total_tn, total_fp, total_fn)
        print(f"Total Samples: {total_processed}")
        print(f"Confusion Matrix:")
        print(f"  True Positives  (TP): {total_tp:3d}  [Correctly detected errors]")
        print(f"  True Negatives  (TN): {total_tn:3d}  [Correctly approved valid statements]")
        print(f"  False Positives (FP): {total_fp:3d}  [Wrongly flagged valid statements]")
        print(f"  False Negatives (FN): {total_fn:3d}  [Missed errors]")
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {overall_metrics['accuracy']:6.2f}%  [Overall correctness]")
        print(f"  Precision: {overall_metrics['precision']:6.2f}%  [Of flagged errors, how many are real]")
        print(f"  Recall:    {overall_metrics['recall']:6.2f}%  [Of actual errors, how many caught]")
        print(f"  F1-Score:  {overall_metrics['f1']:6.2f}%  [Balance between precision and recall]")
    
    cost = (total_prompt / 1_000_000 * PRICE_INPUT + total_completion / 1_000_000 * PRICE_OUTPUT)
    print(f"\nToken Usage & Cost:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"    Prompt tokens:     {total_prompt:,}")
    print(f"    Completion tokens: {total_completion:,}")
    print(f"  Cost (Gemini 2.5 Pro): ${cost:.4f}")
    print("=" * 110)
    
    # Write summary to files
    summary_lines = [
        "=" * 110,
        "COMPREHENSIVE EVALUATION SUMMARY - ZERO-SHOT EASY BS",
        "=" * 110,
        f"Evaluation Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "OVERALL METRICS:",
        f"Total Samples: {total_processed}",
        f"Accuracy: {overall_metrics['accuracy']:6.2f}%",
        f"F1-Score: {overall_metrics['f1']:6.2f}%",
        f"Total Cost: ${cost:.4f}",
        "=" * 110,
    ]
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\nResults written to:")
    print(f"  CSV: {csv_file.absolute()}")
    print(f"  Metrics: {metrics_file.absolute()}")

if __name__ == "__main__":
    main()