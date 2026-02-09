#!/usr/bin/env python3
"""
fewshot_easy_SE_gpt_evaluation.py

Comprehensive few-shot evaluation including both compliant and error-injected violations.
- SE-00.md: COMPLIANT (Ground truth: T)
- SE-01 to SE-03.md: ERROR-INJECTED (Ground truth: F each)

Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Token tracking
"""

import time
import logging
import re
import csv
from pathlib import Path
from openai import OpenAI
import tiktoken

client = OpenAI()

try:
    encoding = tiktoken.encoding_for_model("gpt-4o")
except:
    encoding = tiktoken.get_encoding("cl100k_base")

RULES = {
    "SE00": {
        "filename": "new_SE/SE-00.md",
        "ground_truth": "T",
        "prompt_prefix": """You are a strict financial auditor checking Stockholders' Equity statements using few-shot examples.

RULE: All equity statement rules must be satisfied.

EXAMPLES (Compliant):
- Year 2023: Components sum to Total Equity correctly | Net income shown in reconciliation | OCI shown in reconciliation → T
- All sections reconcile properly for all years → T

Now evaluate:

STATEMENT OF EQUITY DATA:
""",
    },
    "SE01": {
        "filename": "new_SE/SE-01.md",
        "ground_truth": "F",
        "prompt_prefix": """You are a strict financial auditor checking Stockholders' Equity statements.

RULE: Total Equity MUST equal the sum of all components for EVERY year.

EXAMPLES (Compliant):
- Year 2023: Components (5,893 + 62,564 - 1,099 + 46 - 60,429) = 6,975 ✓ → T
- Year 2022: Components (5,298 + 53,607 - 1,253 + 58 - 51,354) = 6,356 ✓ → T

EXAMPLES (Non-Compliant - ERRORS):
- Year 2023: Components sum to 6,975 but reported Total Equity = 7,000 (WRONG by 25) → F
- Year 2023: Retained Earnings changed to 62,600 (INJECTED), breaks equation → F

Now evaluate:

STATEMENT OF EQUITY DATA:
""",
    },
    "SE02": {
        "filename": "new_SE/SE-02.md",
        "ground_truth": "F",
        "prompt_prefix": """You are a strict financial auditor checking Stockholders' Equity statements.

RULE: Retained earnings reconciliation MUST show "Net income (loss)" line for each period.

EXAMPLES (Compliant):
- Shows "Net income | 11,195" in 2023 retained earnings reconciliation → T
- Shows "Net income | 9,930" in 2022 retained earnings reconciliation → T

EXAMPLES (Non-Compliant - ERRORS):
- 2023 reconciliation missing "Net income" line entirely (DELETED) → F
- 2022 reconciliation shows other items but no "Net income" line → F

Now evaluate:

STATEMENT OF EQUITY DATA:
""",
    },
    "SE03": {
        "filename": "new_SE/SE-03.md",
        "ground_truth": "F",
        "prompt_prefix": """You are a strict financial auditor checking Stockholders' Equity statements.

RULE: AOCI reconciliation MUST show "Other comprehensive income" or "Total OCI" line for each period.

EXAMPLES (Compliant):
- Shows "Other comprehensive income (loss) | 154" in 2023 AOCI reconciliation → T
- Shows "Other comprehensive income (loss) | (444)" in 2022 AOCI reconciliation → T

EXAMPLES (Non-Compliant - ERRORS):
- 2023 reconciliation missing "Other comprehensive income" line (DELETED) → F
- 2022 reconciliation shows only beginning/ending AOCI but no OCI flow → F

Now evaluate:

STATEMENT OF EQUITY DATA:
""",
    },
}

PROMPT_SUFFIX = "\nBased on these examples, evaluate the equity statement above.\nRespond ONLY with 'T' or 'F':"

def parse_markdown_file(filepath: Path) -> list:
    """Parse markdown file to extract equity tables."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tables = []
    lines = content.split('\n')
    current_company = None
    current_date = None
    table_lines = []
    in_table = False
    
    for line in lines:
        if line.startswith('### ') or line.startswith('## '):
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

def call_with_retry(messages, model="gpt-4o", max_attempts=6, base_delay=2):
    """Call OpenAI API with exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
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
    
    prompt = RULES[rule_code]["prompt_prefix"] + table_string + PROMPT_SUFFIX
    
    completion = call_with_retry([{"role": "user", "content": prompt}], model="gpt-4o")
    raw_reply = completion.choices[0].message.content.strip()
    
    tf = raw_reply[0].upper() if raw_reply and raw_reply[0].upper() in ('T', 'F') else 'F'
    
    return tf, {
        "prompt_tokens": completion.usage.prompt_tokens,
        "completion_tokens": completion.usage.completion_tokens,
        "total_tokens": completion.usage.total_tokens
    }

def calculate_metrics(tp, tn, fp, fn):
    """Calculate accuracy, precision, recall, F1-score."""
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
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
    log_file = Path("fewshot_easy_se_eval.log")
    logging.basicConfig(filename=str(log_file), level=logging.INFO, format='%(asctime)s - %(message)s')
    csv_file = Path("fewshot_easy_se_eval_results.csv")
    metrics_file = Path("fewshot_easy_se_eval_metrics.txt")
    
    print("\n" + "=" * 110)
    print("FEW-SHOT ERROR DETECTION EVALUATION (SE00 COMPLIANT + SE01-SE03 ERROR-INJECTED)")
    print("=" * 110)
    
    rule_results = {}
    all_rows = []
    
    for rule_code in ["SE00", "SE01", "SE02", "SE03"]:
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
            print(f"Found {len(tables)} equity statements ({gt_label})")
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
    
    for rule_code in ["SE00", "SE01", "SE02", "SE03"]:
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
    
    cost = (total_prompt / 1_000_000 * 2.50 + total_completion / 1_000_000 * 10.00)
    print(f"\nToken Usage & Cost:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"    Prompt tokens:     {total_prompt:,}")
    print(f"    Completion tokens: {total_completion:,}")
    print(f"  Cost (gpt-4o): ${cost:.4f}")
    print("=" * 110)
    
    # ========================================================================
    # Write comprehensive summary to metrics file and log file
    # ========================================================================
    
    summary_lines = [
        "=" * 110,
        "COMPREHENSIVE EVALUATION SUMMARY - FEW-SHOT EASY SE",
        "=" * 110,
        "",
        f"Evaluation Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "PER-RULE METRICS:",
        "-" * 110,
    ]
    
    for rule_code in ["SE00", "SE01", "SE02", "SE03"]:
        if rule_code in rule_results:
            stats = rule_results[rule_code]
            if stats['processed'] > 0:
                metrics = calculate_metrics(stats['tp'], stats['tn'], stats['fp'], stats['fn'])
                gt_label = "COMPLIANT" if rule_code == "SE00" else f"ERROR-INJECTED ({rule_code})"
                summary_lines.extend([
                    f"\n{rule_code} ({gt_label}):",
                    f"  Processed: {stats['processed']} | TP: {stats['tp']:3d} | TN: {stats['tn']:3d} | FP: {stats['fp']:3d} | FN: {stats['fn']:3d}",
                    f"  Accuracy:  {metrics['accuracy']:6.2f}% | Precision: {metrics['precision']:6.2f}% | Recall: {metrics['recall']:6.2f}% | F1: {metrics['f1']:6.2f}%",
                    f"  Tokens: {stats['total_tokens']:,} (Input: {stats['prompt_tokens']:,}, Output: {stats['completion_tokens']:,})",
                    f"  Cost: ${(stats['prompt_tokens'] / 1_000_000 * 2.50 + stats['completion_tokens'] / 1_000_000 * 10.00):.4f}",
                ])
    
    summary_lines.extend([
        "",
        "-" * 110,
        "OVERALL METRICS (All Rules Combined):",
        "-" * 110,
        f"Total Samples: {total_processed}",
        "",
        "Confusion Matrix:",
        f"  True Positives  (TP): {total_tp:3d}  [Correctly detected errors]",
        f"  True Negatives  (TN): {total_tn:3d}  [Correctly approved valid statements]",
        f"  False Positives (FP): {total_fp:3d}  [Wrongly flagged valid statements]",
        f"  False Negatives (FN): {total_fn:3d}  [Missed errors]",
        "",
        "Performance Metrics:",
    ])
    
    if total_processed > 0:
        overall_metrics = calculate_metrics(total_tp, total_tn, total_fp, total_fn)
        summary_lines.extend([
            f"  Accuracy:  {overall_metrics['accuracy']:6.2f}%  [Overall correctness = (TP+TN) / Total]",
            f"  Precision: {overall_metrics['precision']:6.2f}%  [Of flagged errors, how many are real = TP / (TP+FP)]",
            f"  Recall:    {overall_metrics['recall']:6.2f}%  [Of actual errors, how many caught = TP / (TP+FN)]",
            f"  F1-Score:  {overall_metrics['f1']:6.2f}%  [Balance between precision and recall]",
        ])
    
    summary_lines.extend([
        "",
        "Token Usage & Cost:",
        f"  Total tokens: {total_tokens:,}",
        f"    Prompt tokens:     {total_prompt:,}",
        f"    Completion tokens: {total_completion:,}",
        f"  Cost (gpt-4o @ $2.50/1M input, $10.00/1M output): ${cost:.4f}",
        "",
        "=" * 110,
        "Output Files Generated:",
        f"  - Results CSV: {csv_file.absolute()}",
        f"  - Metrics Summary: {metrics_file.absolute()}",
        f"  - Detailed Log: {log_file.absolute()}",
        "=" * 110,
    ])
    
    # Write to metrics file
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    # Write to log file as well
    logging.info("\n" + '\n'.join(summary_lines))
    
    # Print to console
    print(f"\nResults written to:")
    print(f"  CSV: {csv_file.absolute()}")
    print(f"  Metrics: {metrics_file.absolute()}")
    print(f"  Log: {log_file.absolute()}")

if __name__ == "__main__":
    main()