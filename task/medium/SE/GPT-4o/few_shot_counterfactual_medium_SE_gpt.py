#!/usr/bin/env python3
"""
few_shot_counterfactual_medium_SE_gpt.py

Few-shot with causal and counterfactual reasoning for medium task.
- Includes CONCRETE TABLE EXAMPLES showing compliant and violation patterns
- Causal: What causes a violation?
- Counterfactual: If X changed to Y, would it violate the rule?

Includes SE-00 (compliant) baseline and SE-01 through SE-03 (error-injected).

Outputs:
 - few_shot_counterfactual_medium_se_eval_results.csv with confusion matrix columns
 - few_shot_counterfactual_medium_se_eval_metrics.txt with comprehensive summary
 - few_shot_counterfactual_medium_se_eval.log
"""

import time
import logging
import re
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
from openai import OpenAI

# --- Configuration ----------------------------------------------------------
INPUT_FILES = [
    "new_SE/SE-00.md",  # Compliant (ground truth: NO)
    "new_SE/SE-01.md",
    "new_SE/SE-02.md",
    "new_SE/SE-03.md",
]

GROUND_TRUTH = {
    "SE-00.md": "NO",   # Compliant - no violations
    "SE-01.md": "SE01",
    "SE-02.md": "SE02",
    "SE-03.md": "SE03",
}

client = OpenAI()

MODEL_NAME = "gpt-4o"
TEMPERATURE = 0
DELAY_BETWEEN_CALLS = 1.5

# Pricing for gpt-4o (per million tokens)
PRICE_INPUT = 2.50
PRICE_OUTPUT = 10.00

LOG_FILE = Path("few_shot_counterfactual_medium_se_eval.log")
CSV_OUT = Path("few_shot_counterfactual_medium_se_eval_results.csv")
METRICS_FILE = Path("few_shot_counterfactual_medium_se_eval_metrics.txt")

# ---------------------------------------------------------------------------
# Few-Shot Examples with Concrete Tables and Causal/Counterfactual Reasoning
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = """
REFERENCE EXAMPLES (check rules in priority order: SE01→SE02→SE03):

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 1: NO VIOLATION
Total Equity = sum of components for all years ✓
Retained Earnings shows "Net income" line ✓
AOCI shows "Other comprehensive income" line ✓
→ OUTPUT: NO

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 2: SE01 VIOLATION
Year 2023: 5,893 + 62,564 - 1,099 + 46 - 60,429 = 6,975
But reported Total Equity = 6,850 ✗
(Off by -125)
→ OUTPUT: SE01
(Counterfactual: If Total Equity→6,975, would be NO)

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 3: SE02 VIOLATION
Equation ✓
Retained Earnings reconciliation MISSING "Net income" line for 2023 ✗
→ OUTPUT: SE02
(Counterfactual: If "Net income | 11,195" added, would be NO)

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 4: SE03 VIOLATION
Equation ✓ | Net income line ✓
AOCI reconciliation MISSING "Other comprehensive income" line for 2023 ✗
→ OUTPUT: SE03
(Counterfactual: If "Other comprehensive income | 154" added, would be NO)
"""

PROMPT_TEMPLATE = """
Financial auditor: Identify THE SINGLE violated rule (or NO if compliant).

RULES (check sequentially, report FIRST violation):

SE01: Total Equity MUST equal sum of components for EVERY year
      (Common Stock + Retained Earnings + AOCI + Non-controlling - Treasury = Total Equity)

SE02: Retained Earnings reconciliation MUST show "Net income" or "Net loss" line
      for each period

SE03: AOCI reconciliation MUST show "Other comprehensive income" line
      for each period

{few_shot_examples}

INSTRUCTIONS:
- Check rules in order: SE01 → SE02 → SE03
- Each statement has AT MOST ONE violation
- Report FIRST violation found
- Respond with CODE ONLY (NO explanation)

Valid codes: SE01, SE02, SE03, NO

STATEMENT OF EQUITY DATA:
{table_data_string}

Your response (ONE WORD ONLY):
""".strip()

# ---------------------------------------------------------------------------
# Utility Functions

def parse_markdown_file(filepath: Path) -> List[Dict[str, Any]]:
    """Parse markdown file into list of tables."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    tables = []
    lines = content.split('\n')
    current_company = None
    current_date = None
    table_lines = []
    in_table = False
    i = 0

    while i < len(lines):
        line = lines[i]

        if line.startswith('## ') or line.startswith('### '):
            if table_lines and current_company:
                tables.append({
                    'company_name': current_company,
                    'date': current_date,
                    'table_name': 'se',
                    'data': '\n'.join(table_lines)
                })
                table_lines = []

            header = line.replace('## ', '').replace('### ', '').strip()
            if '—' in header:
                current_company, current_date = header.split('—', 1)
                current_company = current_company.strip()
                current_date = current_date.strip()
            else:
                current_company = header
                current_date = ""

            in_table = False
            i += 1
            continue

        if line.strip().startswith('|'):
            if not in_table:
                in_table = True
            table_lines.append(line)

        elif in_table and line.strip() == '':
            in_table = False

        elif in_table and line.strip():
            table_lines.append(line)

        i += 1

    if table_lines and current_company:
        tables.append({
            'company_name': current_company,
            'date': current_date,
            'table_name': 'se',
            'data': '\n'.join(table_lines)
        })

    return tables


def extract_table_data(table_text: str) -> str:
    """Turn markdown table into clean pipe-less text for the prompt."""
    lines = table_text.split('\n')
    clean_lines = []

    for line in lines:
        if not line.strip():
            continue

        l = line.strip()
        if l.startswith('|'):
            l = l[1:]
        if l.endswith('|'):
            l = l[:-1]

        if re.match(r'^\s*-+\s*\|\s*-+\s*$', l):
            continue

        cells = [cell.strip() for cell in l.split('|')]
        clean_lines.append(' | '.join(cells))

    return '\n'.join(clean_lines)


def build_prompt(table_data_string: str) -> str:
    """Fill the template with the cleaned table data and few-shot examples."""
    return PROMPT_TEMPLATE.format(few_shot_examples=FEW_SHOT_EXAMPLES, table_data_string=table_data_string)


def call_with_retry(*, messages, model=MODEL_NAME, max_attempts=6, base_delay=2):
    """Call the OpenAI client with exponential backoff and return the response."""
    attempt = 1
    while attempt <= max_attempts:
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
            )
        except Exception as e:
            print(f"[ERROR] Attempt {attempt}: {e}")
            if attempt < max_attempts:
                wait = base_delay * (2 ** (attempt - 1))
                print(f"Retrying in {wait:.1f}s...")
                time.sleep(wait)
                attempt += 1
            else:
                raise


def check_table(table_info: Dict[str, Any]) -> Tuple[str, str, str, Dict[str, int]]:
    """
    Send one table to the LLM and return (raw_reply, code, explanation, token_usage).
    Accepts codes: SE00, SE01, SE02, SE03, NO
    Token usage: {prompt_tokens, completion_tokens, total_tokens}
    """
    table_text = table_info.get("data", "")
    if isinstance(table_text, list):
        table_text = '\n'.join(table_text)

    cleaned_table = extract_table_data(table_text)
    if not cleaned_table.strip():
        raise ValueError("Table data is empty after processing")

    prompt = build_prompt(cleaned_table)
    response = call_with_retry(messages=[{"role": "user", "content": prompt}], model=MODEL_NAME)
    raw_reply = response.choices[0].message.content.strip()

    # Capture token usage
    token_usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }

    reply = raw_reply.strip()
    valid_responses = {"SE00", "SE01", "SE02", "SE03", "NO"}

    if ':' in reply:
        parts = reply.split(':', 1)
        code = parts[0].strip().upper()
        explanation = parts[1].strip()
        if code in valid_responses:
            return raw_reply, code, explanation, token_usage
        for valid in valid_responses:
            if valid in code:
                return raw_reply, valid, explanation, token_usage

    for valid in valid_responses:
        if valid in reply:
            after = reply.split(valid, 1)[1].strip()
            if after.startswith(':'):
                after = after[1:].strip()
            return raw_reply, valid, after, token_usage

    raise ValueError(f"Unrecognized LLM output: {raw_reply!r}")


def calculate_metrics(tp, tn, fp, fn):
    """Calculate Accuracy, Precision, Recall, F1-Score."""
    total = tp + tn + fp + fn
    
    accuracy = ((tp + tn) / total * 100) if total > 0 else 0.0
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(filename=str(LOG_FILE), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    csv_fields = ["file", "company_name", "date", "expected", "raw_reply", "predicted", "explanation", 
                  "tp", "tn", "fp", "fn", "error_flag", "prompt_tokens", "completion_tokens", "total_tokens"]
    csv_rows = []

    overall = {
        'processed': 0,
        'correct': 0,
        'errors': 0,
        'tp': 0,
        'tn': 0,
        'fp': 0,
        'fn': 0,
        'total_prompt_tokens': 0,
        'total_completion_tokens': 0,
        'total_tokens': 0
    }

    file_rule_stats = {}  # Track stats per rule for detailed summary

    for input_file in INPUT_FILES:
        p = Path(input_file)
        print(f"\nProcessing file: {input_file} -- exists: {p.exists()}")
        if not p.exists():
            print(f"ERROR: File not found: {input_file}")
            logging.error(f"File not found: {input_file}")
            continue

        expected_label = GROUND_TRUTH.get(p.name)
        if not expected_label:
            print(f"WARNING: No ground truth for {p.name}, skipping.")
            logging.warning(f"No ground truth for {p.name}")
            continue

        try:
            tables = parse_markdown_file(p)
            if not tables:
                print(f"WARNING: No tables found in {input_file}")
                logging.warning(f"No tables found in {input_file}")
                continue
            gt_label = "COMPLIANT" if expected_label == "NO" else f"ERROR-INJECTED ({expected_label})"
            print(f"Found {len(tables)} tables ({gt_label})")
        except Exception as e:
            print(f"ERROR parsing {input_file}: {e}")
            logging.exception(f"Error parsing {input_file}: {e}")
            continue

        file_stats = {
            'processed': 0,
            'correct': 0,
            'errors': 0,
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }

        rule_key = p.stem  # e.g., "SE-00", "SE-01"
        file_rule_stats[rule_key] = file_stats.copy()

        for idx, tbl in enumerate(tables, start=1):
            comp = tbl.get("company_name", "Unknown")
            date = tbl.get("date", "")
            file_stats['processed'] += 1
            overall['processed'] += 1

            print(f"  [{idx:3d}/{len(tables)}] {comp:30s} ({date:20s}) GT: {expected_label}...", end=" ", flush=True)
            try:
                raw_reply, predicted, explanation, token_usage = check_table(tbl)
                
                file_stats['prompt_tokens'] += token_usage['prompt_tokens']
                file_stats['completion_tokens'] += token_usage['completion_tokens']
                file_stats['total_tokens'] += token_usage['total_tokens']
                
                overall['total_prompt_tokens'] += token_usage['prompt_tokens']
                overall['total_completion_tokens'] += token_usage['completion_tokens']
                overall['total_tokens'] += token_usage['total_tokens']

                # Confusion matrix logic
                if expected_label == "NO":  # Compliant file
                    if predicted == "NO":
                        file_stats['tn'] += 1
                        overall['tn'] += 1
                        status = "✓TN"
                    else:
                        file_stats['fp'] += 1
                        overall['fp'] += 1
                        status = "✗FP"
                else:  # Error-injected files
                    if predicted == expected_label:
                        file_stats['tp'] += 1
                        overall['tp'] += 1
                        status = "✓TP"
                    else:
                        file_stats['fn'] += 1
                        overall['fn'] += 1
                        status = "✗FN"

                if predicted == expected_label:
                    file_stats['correct'] += 1
                    overall['correct'] += 1

                csv_rows.append({
                    "file": input_file,
                    "company_name": comp,
                    "date": date,
                    "expected": expected_label,
                    "raw_reply": raw_reply,
                    "predicted": predicted,
                    "explanation": explanation,
                    "tp": 1 if status == "✓TP" else 0,
                    "tn": 1 if status == "✓TN" else 0,
                    "fp": 1 if status == "✗FP" else 0,
                    "fn": 1 if status == "✗FN" else 0,
                    "error_flag": False,
                    "prompt_tokens": token_usage['prompt_tokens'],
                    "completion_tokens": token_usage['completion_tokens'],
                    "total_tokens": token_usage['total_tokens']
                })

                print(f"{predicted} (expected={expected_label}) {status}")
                logging.info(f"File: {input_file}, Company: {comp}, Date: {date}, Predicted: {predicted}, Expected: {expected_label}, Status: {status}, Tokens: {token_usage['total_tokens']}")

            except Exception as e:
                print("ERROR")
                logging.exception(f"Error processing {input_file} / {comp}: {e}")
                csv_rows.append({
                    "file": input_file,
                    "company_name": comp,
                    "date": date,
                    "expected": expected_label,
                    "raw_reply": "",
                    "predicted": "",
                    "explanation": str(e),
                    "tp": 0,
                    "tn": 0,
                    "fp": 0,
                    "fn": 0,
                    "error_flag": True,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                })
                file_stats['errors'] += 1
                overall['errors'] += 1

            time.sleep(DELAY_BETWEEN_CALLS)

        # Update per-rule stats tracking
        file_rule_stats[rule_key] = file_stats.copy()

        # per-file summary
        proc = file_stats['processed']
        corr = file_stats['correct']
        tp = file_stats['tp']
        tn = file_stats['tn']
        fp = file_stats['fp']
        fn = file_stats['fn']
        errs = file_stats['errors']
        
        metrics = calculate_metrics(tp, tn, fp, fn)
        
        cost = (file_stats['prompt_tokens'] / 1_000_000 * PRICE_INPUT + 
                file_stats['completion_tokens'] / 1_000_000 * PRICE_OUTPUT)
        print(f"\n  {rule_key} Results:")
        print(f"    Processed: {proc}, Correct: {corr}, Errors: {errs}")
        print(f"    Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f"    Accuracy: {metrics['accuracy']:.2f}%, Precision: {metrics['precision']:.2f}%, Recall: {metrics['recall']:.2f}%, F1: {metrics['f1']:.2f}%")
        print(f"    Tokens: {file_stats['total_tokens']:,} (Input: {file_stats['prompt_tokens']:,}, Output: {file_stats['completion_tokens']:,})")
        print(f"    Cost: ${cost:.4f}")
        logging.info(f"File summary for {input_file}: processed={proc}, correct={corr}, errors={errs}, TP={tp}, TN={tn}, FP={fp}, FN={fn}, accuracy={metrics['accuracy']:.2f}%, tokens={file_stats['total_tokens']}, cost=${cost:.4f}")

    # write CSV
    try:
        with open(CSV_OUT, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)
        print(f"\nResults written to {CSV_OUT}")
        logging.info(f"Results written to {CSV_OUT}")
    except Exception as e:
        print(f"ERROR writing CSV: {e}")
        logging.exception(f"Error writing CSV: {e}")

    # overall summary to console
    print("\n" + "=" * 110)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 110)

    for rule_key in ["SE-00", "SE-01", "SE-02", "SE-03"]:
        if rule_key in file_rule_stats:
            stats = file_rule_stats[rule_key]
            if stats['processed'] > 0:
                metrics = calculate_metrics(stats['tp'], stats['tn'], stats['fp'], stats['fn'])
                gt_label = "COMPLIANT" if rule_key == "SE-00" else f"ERROR-INJECTED ({rule_key})"
                print(f"{rule_key} ({gt_label}): Acc={metrics['accuracy']:5.1f}% | Prec={metrics['precision']:5.1f}% | Rec={metrics['recall']:5.1f}% | F1={metrics['f1']:5.1f}%")

    print(f"\n{'-' * 110}")
    print("OVERALL METRICS (All Rules Combined)")
    print(f"{'-' * 110}")
    print(f"Total Samples: {overall['processed']}")
    print(f"Confusion Matrix:")
    print(f"  True Positives  (TP): {overall['tp']:3d}  [Correctly detected violations]")
    print(f"  True Negatives  (TN): {overall['tn']:3d}  [Correctly approved compliant statements]")
    print(f"  False Positives (FP): {overall['fp']:3d}  [Wrongly flagged compliant statements]")
    print(f"  False Negatives (FN): {overall['fn']:3d}  [Missed violations]")
    
    if overall['processed'] > 0:
        overall_metrics = calculate_metrics(overall['tp'], overall['tn'], overall['fp'], overall['fn'])
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {overall_metrics['accuracy']:6.2f}%  [Overall correctness]")
        print(f"  Precision: {overall_metrics['precision']:6.2f}%  [Of flagged violations, how many are real]")
        print(f"  Recall:    {overall_metrics['recall']:6.2f}%  [Of actual violations, how many caught]")
        print(f"  F1-Score:  {overall_metrics['f1']:6.2f}%  [Balance between precision and recall]")
    
    print(f"\nToken Usage & Cost:")
    print(f"  Total tokens: {overall['total_tokens']:,}")
    print(f"    Prompt tokens:     {overall['total_prompt_tokens']:,}")
    print(f"    Completion tokens: {overall['total_completion_tokens']:,}")
    overall_cost = (overall['total_prompt_tokens'] / 1_000_000 * PRICE_INPUT + 
                    overall['total_completion_tokens'] / 1_000_000 * PRICE_OUTPUT)
    print(f"  Cost (gpt-4o): ${overall_cost:.4f}")
    print("=" * 110)

    # ========================================================================
    # Write comprehensive summary to metrics file
    # ========================================================================
    
    summary_lines = [
        "=" * 110,
        "COMPREHENSIVE EVALUATION SUMMARY - FEW-SHOT COUNTERFACTUAL MEDIUM SE",
        "=" * 110,
        "",
        f"Evaluation Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "PER-RULE METRICS:",
        "-" * 110,
    ]

    for rule_key in ["SE-00", "SE-01", "SE-02", "SE-03"]:
        if rule_key in file_rule_stats:
            stats = file_rule_stats[rule_key]
            if stats['processed'] > 0:
                metrics = calculate_metrics(stats['tp'], stats['tn'], stats['fp'], stats['fn'])
                gt_label = "COMPLIANT" if rule_key == "SE-00" else f"ERROR-INJECTED ({rule_key})"
                summary_lines.extend([
                    f"\n{rule_key} ({gt_label}):",
                    f"  Processed: {stats['processed']} | TP: {stats['tp']:3d} | TN: {stats['tn']:3d} | FP: {stats['fp']:3d} | FN: {stats['fn']:3d}",
                    f"  Accuracy:  {metrics['accuracy']:6.2f}% | Precision: {metrics['precision']:6.2f}% | Recall: {metrics['recall']:6.2f}% | F1: {metrics['f1']:6.2f}%",
                    f"  Tokens: {stats['total_tokens']:,} (Input: {stats['prompt_tokens']:,}, Output: {stats['completion_tokens']:,})",
                    f"  Cost: ${(stats['prompt_tokens'] / 1_000_000 * PRICE_INPUT + stats['completion_tokens'] / 1_000_000 * PRICE_OUTPUT):.4f}",
                ])

    summary_lines.extend([
        "",
        "-" * 110,
        "OVERALL METRICS (All Rules Combined):",
        "-" * 110,
        f"Total Samples: {overall['processed']}",
        "",
        "Confusion Matrix:",
        f"  True Positives  (TP): {overall['tp']:3d}  [Correctly detected violations]",
        f"  True Negatives  (TN): {overall['tn']:3d}  [Correctly approved compliant statements]",
        f"  False Positives (FP): {overall['fp']:3d}  [Wrongly flagged compliant statements]",
        f"  False Negatives (FN): {overall['fn']:3d}  [Missed violations]",
        "",
        "Performance Metrics:",
    ])

    if overall['processed'] > 0:
        overall_metrics = calculate_metrics(overall['tp'], overall['tn'], overall['fp'], overall['fn'])
        summary_lines.extend([
            f"  Accuracy:  {overall_metrics['accuracy']:6.2f}%  [Overall correctness = (TP+TN) / Total]",
            f"  Precision: {overall_metrics['precision']:6.2f}%  [Of flagged violations, how many are real = TP / (TP+FP)]",
            f"  Recall:    {overall_metrics['recall']:6.2f}%  [Of actual violations, how many caught = TP / (TP+FN)]",
            f"  F1-Score:  {overall_metrics['f1']:6.2f}%  [Balance between precision and recall]",
        ])

    summary_lines.extend([
        "",
        "Token Usage & Cost:",
        f"  Total tokens: {overall['total_tokens']:,}",
        f"    Prompt tokens:     {overall['total_prompt_tokens']:,}",
        f"    Completion tokens: {overall['total_completion_tokens']:,}",
        f"  Cost (gpt-4o @ $2.50/1M input, $10.00/1M output): ${overall_cost:.4f}",
        "",
        "=" * 110,
        "Output Files Generated:",
        f"  - Results CSV: {CSV_OUT.absolute()}",
        f"  - Metrics Summary: {METRICS_FILE.absolute()}",
        f"  - Detailed Log: {LOG_FILE.absolute()}",
        "=" * 110,
    ])

    # Write to metrics file
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    # Write to log file as well
    logging.info("\n" + '\n'.join(summary_lines))

    # Print to console
    print(f"\nResults written to:")
    print(f"  CSV: {CSV_OUT.absolute()}")
    print(f"  Metrics: {METRICS_FILE.absolute()}")
    print(f"  Log: {LOG_FILE.absolute()}")


if __name__ == "__main__":
    main()