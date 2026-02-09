#!/usr/bin/env python3
"""
few_shot_counterfactual_medium_SI_gpt.py

Runs the medium multi-class classification prompt across SI-00.md (baseline compliant),
SI-01.md ... SI-05.md and evaluates the model predictions against ground truth.

Counterfactual few-shot variant: Includes examples with "if X changed, result would flip" reasoning.

Outputs:
- Console per-file and overall summary with token metrics and confusion matrix
- CSV file "few_shot_counterfactual_medium_si_eval_results.csv" with one row per company-table
- Log file "few_shot_counterfactual_medium_si_eval.log"
- Metrics file "few_shot_counterfactual_medium_si_eval_metrics.txt" with per-rule and overall metrics
"""

import time
import sys
import logging
import re
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
from openai import OpenAI

# --- Configuration ----------------------------------------------------------
INPUT_FILES = [
    "new_SI/SI-00.md",  # Baseline compliant (no violations)
    "new_SI/SI-01.md",
    "new_SI/SI-02.md",
    "new_SI/SI-03.md",
    "new_SI/SI-04.md",
    "new_SI/SI-05.md",
]

# Ground-truth mapping: each file's expected outcome
GROUND_TRUTH = {
    "SI-00.md": "NO",    # Compliant baseline
    "SI-01.md": "SI01",  # Violation: Operating vs Non-Operating classification
    "SI-02.md": "SI02",  # Violation: Depreciation placement
    "SI-03.md": "SI03",  # Violation: EPS presentation
    "SI-04.md": "SI04",  # Violation: Miscellaneous income clarity
    "SI-05.md": "SI05",  # Violation: Interest expense presence
}

# OpenAI client (assumes OPENAI_API_KEY env var set)
client = OpenAI()

# Model settings
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0  # Deterministic responses for consistent evaluation
DELAY_BETWEEN_CALLS = 1.5  # seconds

# Pricing for gpt-4o
PRICE_INPUT = 2.50  # per million tokens
PRICE_OUTPUT = 10.00  # per million tokens

# Files for output
LOG_FILE = Path("few_shot_counterfactual_medium_si_eval.log")
CSV_OUT = Path("few_shot_counterfactual_medium_si_eval_results.csv")
METRICS_FILE = Path("few_shot_counterfactual_medium_si_eval_metrics.txt")

# ---------------------------------------------------------------------------
# Counterfactual few-shot examples (simplified causal/counterfactual reasoning)
FEW_SHOT_EXAMPLES = """
REFERENCE EXAMPLES (check rules in priority order: SI01→SI02→SI03→SI04→SI05):

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 1: NO VIOLATION
Non-operating items AFTER operating expenses ✓
Depreciation within operating expenses ✓
Both Basic and Diluted EPS present ✓
All categories well-defined (no "Miscellaneous") ✓
Interest expense separately presented ✓
→ OUTPUT: NO

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 2: SI01 VIOLATION
Operating Expenses includes "Interest expense" ✗
(Should be in Non-Operating section AFTER operating)
→ OUTPUT: SI01
(Counterfactual: If Interest moved to Non-Operating, would be NO)

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 3: SI02 VIOLATION
Section separation ✓
Standalone "Depreciation Expense" line outside Operating ✗
(Should be within Operating Expenses)
→ OUTPUT: SI02
(Counterfactual: If Depreciation moved into Operating, would be NO)

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 4: SI03 VIOLATION
Sections ✓ | Depreciation ✓
Shows only "Basic EPS: 10.05", missing Diluted EPS ✗
→ OUTPUT: SI03
(Counterfactual: If "Diluted EPS: 9.95" added, would be NO)

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 5: SI04 VIOLATION
All above ✓
Line item: "Miscellaneous other income: 268" ✗
(Vague category, not well-defined)
→ OUTPUT: SI04
(Counterfactual: If changed to "Investment income", would be NO)

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 6: SI05 VIOLATION
All above ✓
Interest expense line completely missing ✗
→ OUTPUT: SI05
(Counterfactual: If "Interest expense: -575" added, would be NO)
"""

PROMPT_TEMPLATE = """
Financial auditor: Identify THE SINGLE violated rule (or NO if compliant).

RULES (check sequentially, report FIRST violation):

SI01: Non-operating items MUST appear AFTER operating expenses (in separate section)

SI02: Depreciation MUST be WITHIN operating expenses (not listed separately)

SI03: BOTH Basic AND Diluted EPS MUST be presented on face of statement

SI04: All line items MUST be well-defined categories
      (NO vague terms like "Miscellaneous", "Other income", "Sundry")

SI05: Interest expense MUST be separately presented as non-operating expense

{few_shot_examples}

INSTRUCTIONS:
- Check rules in order: SI01 → SI02 → SI03 → SI04 → SI05
- Each statement has AT MOST ONE violation
- Report FIRST violation found
- Respond with CODE ONLY (NO explanation)

Valid codes: SI01, SI02, SI03, SI04, SI05, NO

INCOME STATEMENT DATA:
{table_data_string}

Your response (ONE WORD ONLY):
""".strip()

# ---------------------------------------------------------------------------
# Utilities (parsing, cleaning, retrying)

def parse_markdown_file(filepath: Path) -> List[Dict[str, Any]]:
    """Parse markdown file into list of tables (company_name, date, data)."""
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

        if line.startswith('### ') or line.startswith('## '):
            if table_lines and current_company:
                tables.append({
                    'company_name': current_company,
                    'date': current_date,
                    'table_name': 'si',
                    'data': '\n'.join(table_lines)
                })
                table_lines = []

            header = line.replace('### ', '').replace('## ', '').strip()
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
            'table_name': 'si',
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

        # skip separator rows like |-|-|
        if re.match(r'^\s*-+\s*\|\s*-+\s*$', l):
            continue

        cells = [cell.strip() for cell in l.split('|')]
        clean_lines.append(' | '.join(cells))

    return '\n'.join(clean_lines)


def build_compliance_prompt(table_data_string: str) -> str:
    """Fill the template with the cleaned table data and counterfactual examples."""
    return PROMPT_TEMPLATE.format(table_data_string=table_data_string, few_shot_examples=FEW_SHOT_EXAMPLES)


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
    Accepts codes: SI01, SI02, SI03, SI04, SI05, NO
    
    Returns:
        Tuple of (raw_reply, code, explanation, token_usage_dict)
        token_usage_dict contains: prompt_tokens, completion_tokens, total_tokens
    """
    table_text = table_info.get("data", "")
    if isinstance(table_text, list):
        table_text = '\n'.join(table_text)

    cleaned_table = extract_table_data(table_text)
    if not cleaned_table.strip():
        raise ValueError("Table data is empty after processing")

    prompt = build_compliance_prompt(cleaned_table)
    response = call_with_retry(messages=[{"role": "user", "content": prompt}], model=MODEL_NAME)
    raw_reply = response.choices[0].message.content.strip()
    
    # Extract token usage
    token_usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }

    # parse reply
    reply = raw_reply.strip()
    valid_responses = {"SI01", "SI02", "SI03", "SI04", "SI05", "NO"}

    if ':' in reply:
        parts = reply.split(':', 1)
        code = parts[0].strip().upper()
        explanation = parts[1].strip()
        # If code valid, return
        if code in valid_responses:
            return raw_reply, code, explanation, token_usage
        # else try to find any valid token in the code chunk
        for valid in valid_responses:
            if valid in code:
                return raw_reply, valid, explanation, token_usage

    # fallback: try to find any valid token anywhere in reply
    for valid in valid_responses:
        if valid in reply:
            # Explanation may not be present
            after = reply.split(valid, 1)[1].strip()
            # remove leading punctuation like ":" if present
            if after.startswith(':'):
                after = after[1:].strip()
            return raw_reply, valid, after, token_usage

    raise ValueError(f"Unrecognized LLM output: {raw_reply!r}")


def calculate_metrics(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    """Calculate precision, recall, F1-score, and accuracy from confusion matrix."""
    metrics = {}
    
    # Accuracy: (TP + TN) / (TP + TN + FP + FN)
    total = tp + tn + fp + fn
    if total > 0:
        metrics['accuracy'] = ((tp + tn) / total) * 100
    else:
        metrics['accuracy'] = 0.0
    
    # Precision: TP / (TP + FP)
    if (tp + fp) > 0:
        metrics['precision'] = (tp / (tp + fp)) * 100
    else:
        metrics['precision'] = 0.0
    
    # Recall: TP / (TP + FN)
    if (tp + fn) > 0:
        metrics['recall'] = (tp / (tp + fn)) * 100
    else:
        metrics['recall'] = 0.0
    
    # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    if (metrics['precision'] + metrics['recall']) > 0:
        metrics['f1'] = (2 * metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1'] = 0.0
    
    return metrics


# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(filename=str(LOG_FILE), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    csv_fields = ["file", "company_name", "date", "expected", "raw_reply", "predicted", "explanation", "tp", "tn", "fp", "fn", "error_flag", "prompt_tokens", "completion_tokens", "total_tokens"]
    csv_rows = []

    overall = {
        'processed': 0,
        'tp': 0,
        'tn': 0,
        'fp': 0,
        'fn': 0,
        'errors': 0,
        'total_prompt_tokens': 0,
        'total_completion_tokens': 0,
        'total_tokens': 0
    }

    # Track stats per rule (SI-00 through SI-05)
    file_rule_stats = {
        "SI-00": {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'total': 0},
        "SI-01": {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'total': 0},
        "SI-02": {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'total': 0},
        "SI-03": {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'total': 0},
        "SI-04": {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'total': 0},
        "SI-05": {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'total': 0},
    }

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
            print(f"Found {len(tables)} tables in {input_file}")
        except Exception as e:
            print(f"ERROR parsing {input_file}: {e}")
            logging.exception(f"Error parsing {input_file}: {e}")
            continue

        file_stats = {
            'processed': 0,
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0,
            'errors': 0,
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0
        }

        for idx, tbl in enumerate(tables, start=1):
            comp = tbl.get("company_name", "Unknown")
            date = tbl.get("date", "")
            file_stats['processed'] += 1
            overall['processed'] += 1

            print(f"[{idx}/{len(tables)}] {comp} ({date}) ... ", end="", flush=True)
            try:
                raw_reply, predicted, explanation, token_usage = check_table(tbl)
                
                # Confusion matrix logic
                tp = tn = fp = fn = 0
                status = ""
                
                if expected_label == "NO":  # Compliant baseline
                    if predicted == "NO":
                        tn = 1
                        status = "✓TN"
                    else:
                        fp = 1
                        status = "✗FP"
                else:  # Error-injected files
                    if predicted == expected_label:
                        tp = 1
                        status = "✓TP"
                    else:
                        fn = 1
                        status = "✗FN"
                
                # Update file and overall stats
                file_stats['tp'] += tp
                file_stats['tn'] += tn
                file_stats['fp'] += fp
                file_stats['fn'] += fn
                overall['tp'] += tp
                overall['tn'] += tn
                overall['fp'] += fp
                overall['fn'] += fn

                # Accumulate token counts
                file_stats['total_prompt_tokens'] += token_usage['prompt_tokens']
                file_stats['total_completion_tokens'] += token_usage['completion_tokens']
                file_stats['total_tokens'] += token_usage['total_tokens']
                overall['total_prompt_tokens'] += token_usage['prompt_tokens']
                overall['total_completion_tokens'] += token_usage['completion_tokens']
                overall['total_tokens'] += token_usage['total_tokens']

                # Track per-rule stats
                rule_key = expected_label
                if rule_key in file_rule_stats:
                    file_rule_stats[rule_key]['tp'] += tp
                    file_rule_stats[rule_key]['tn'] += tn
                    file_rule_stats[rule_key]['fp'] += fp
                    file_rule_stats[rule_key]['fn'] += fn
                    file_rule_stats[rule_key]['total'] += 1

                csv_rows.append({
                    "file": input_file,
                    "company_name": comp,
                    "date": date,
                    "expected": expected_label,
                    "raw_reply": raw_reply,
                    "predicted": predicted,
                    "explanation": explanation,
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "error_flag": False,
                    "prompt_tokens": token_usage['prompt_tokens'],
                    "completion_tokens": token_usage['completion_tokens'],
                    "total_tokens": token_usage['total_tokens']
                })

                print(f"{predicted} (expected={expected_label}) -> {status}")
                logging.info(f"File: {input_file}, Company: {comp}, Date: {date}, Predicted: {predicted}, Expected: {expected_label}, Tokens: {token_usage['total_tokens']}")

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

        # per-file summary
        proc = file_stats['processed']
        tp_f = file_stats['tp']
        tn_f = file_stats['tn']
        fp_f = file_stats['fp']
        fn_f = file_stats['fn']
        errs = file_stats['errors']
        file_metrics = calculate_metrics(tp_f, tn_f, fp_f, fn_f)
        
        file_cost = (file_stats['total_prompt_tokens'] / 1_000_000 * PRICE_INPUT + 
                     file_stats['total_completion_tokens'] / 1_000_000 * PRICE_OUTPUT)
        print(f"--> File summary for {input_file}:")
        print(f"    Processed: {proc}, Errors: {errs}")
        print(f"    Confusion Matrix - TP: {tp_f}, TN: {tn_f}, FP: {fp_f}, FN: {fn_f}")
        print(f"    Accuracy: {file_metrics['accuracy']:.2f}%, Precision: {file_metrics['precision']:.2f}%, Recall: {file_metrics['recall']:.2f}%, F1: {file_metrics['f1']:.2f}%")
        print(f"    Tokens - Prompt: {file_stats['total_prompt_tokens']:,}, Completion: {file_stats['total_completion_tokens']:,}, Total: {file_stats['total_tokens']:,}")
        print(f"    Cost (gpt-4o): ${file_cost:.4f}")
        logging.info(f"File summary for {input_file}: processed={proc}, tp={tp_f}, tn={tn_f}, fp={fp_f}, fn={fn_f}, accuracy={file_metrics['accuracy']:.2f}%, cost=${file_cost:.4f}")

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

    # Write metrics file
    try:
        with open(METRICS_FILE, 'w', encoding='utf-8') as mf:
            mf.write("=" * 80 + "\n")
            mf.write("FEW-SHOT COUNTERFACTUAL INCOME STATEMENT (SI) EVALUATION METRICS\n")
            mf.write("=" * 80 + "\n\n")
            
            mf.write("PER-RULE BREAKDOWN:\n")
            mf.write("-" * 80 + "\n")
            for rule in ["SI-00", "SI-01", "SI-02", "SI-03", "SI-04", "SI-05"]:
                stats = file_rule_stats[rule]
                if stats['total'] > 0:
                    metrics = calculate_metrics(stats['tp'], stats['tn'], stats['fp'], stats['fn'])
                    mf.write(f"\n{rule} (Samples: {stats['total']}):\n")
                    mf.write(f"  Confusion Matrix - TP: {stats['tp']}, TN: {stats['tn']}, FP: {stats['fp']}, FN: {stats['fn']}\n")
                    mf.write(f"  Accuracy:  {metrics['accuracy']:6.2f}%\n")
                    mf.write(f"  Precision: {metrics['precision']:6.2f}%\n")
                    mf.write(f"  Recall:    {metrics['recall']:6.2f}%\n")
                    mf.write(f"  F1-Score:  {metrics['f1']:6.2f}%\n")
            
            mf.write("\n" + "=" * 80 + "\n")
            mf.write("OVERALL METRICS:\n")
            mf.write("=" * 80 + "\n")
            overall_metrics = calculate_metrics(overall['tp'], overall['tn'], overall['fp'], overall['fn'])
            mf.write(f"Total samples processed: {overall['processed']}\n")
            mf.write(f"Total errors (exceptions): {overall['errors']}\n")
            mf.write(f"Confusion Matrix - TP: {overall['tp']}, TN: {overall['tn']}, FP: {overall['fp']}, FN: {overall['fn']}\n\n")
            mf.write(f"Accuracy:  {overall_metrics['accuracy']:6.2f}%\n")
            mf.write(f"Precision: {overall_metrics['precision']:6.2f}%\n")
            mf.write(f"Recall:    {overall_metrics['recall']:6.2f}%\n")
            mf.write(f"F1-Score:  {overall_metrics['f1']:6.2f}%\n\n")
            
            mf.write(f"Token Usage:\n")
            mf.write(f"  Total prompt tokens:     {overall['total_prompt_tokens']:,}\n")
            mf.write(f"  Total completion tokens: {overall['total_completion_tokens']:,}\n")
            mf.write(f"  Total tokens:            {overall['total_tokens']:,}\n\n")
            
            overall_cost = (overall['total_prompt_tokens'] / 1_000_000 * PRICE_INPUT + 
                           overall['total_completion_tokens'] / 1_000_000 * PRICE_OUTPUT)
            mf.write(f"Cost (gpt-4o @ ${PRICE_INPUT:.2f}/1M input, ${PRICE_OUTPUT:.2f}/1M output):\n")
            mf.write(f"  Input cost:  ${overall['total_prompt_tokens'] / 1_000_000 * PRICE_INPUT:.4f}\n")
            mf.write(f"  Output cost: ${overall['total_completion_tokens'] / 1_000_000 * PRICE_OUTPUT:.4f}\n")
            mf.write(f"  Total cost:  ${overall_cost:.4f}\n")
            mf.write("=" * 80 + "\n")
        
        print(f"Metrics written to {METRICS_FILE}")
        logging.info(f"Metrics written to {METRICS_FILE}")
    except Exception as e:
        print(f"ERROR writing metrics file: {e}")
        logging.exception(f"Error writing metrics file: {e}")

    # overall summary
    overall_metrics = calculate_metrics(overall['tp'], overall['tn'], overall['fp'], overall['fn'])
    overall_cost = (overall['total_prompt_tokens'] / 1_000_000 * PRICE_INPUT + 
                    overall['total_completion_tokens'] / 1_000_000 * PRICE_OUTPUT)
    print("\n" + "=" * 80)
    print("OVERALL EVALUATION SUMMARY (with confusion matrix)")
    print("=" * 80)
    print(f"Total samples processed : {overall['processed']}")
    print(f"Total errors (exceptions): {overall['errors']}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives  (TP): {overall['tp']}")
    print(f"  True Negatives  (TN): {overall['tn']}")
    print(f"  False Positives (FP): {overall['fp']}")
    print(f"  False Negatives (FN): {overall['fn']}")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {overall_metrics['accuracy']:6.2f}%")
    print(f"  Precision: {overall_metrics['precision']:6.2f}%")
    print(f"  Recall:    {overall_metrics['recall']:6.2f}%")
    print(f"  F1-Score:  {overall_metrics['f1']:6.2f}%")
    print(f"\nToken Usage Summary:")
    print(f"  Total prompt tokens      : {overall['total_prompt_tokens']:,}")
    print(f"  Total completion tokens  : {overall['total_completion_tokens']:,}")
    print(f"  Total tokens             : {overall['total_tokens']:,}")
    print(f"\nCost Breakdown (gpt-4o @ ${PRICE_INPUT:.2f}/1M input, ${PRICE_OUTPUT:.2f}/1M output):")
    print(f"  Input cost  : ${overall['total_prompt_tokens'] / 1_000_000 * PRICE_INPUT:.4f}")
    print(f"  Output cost : ${overall['total_completion_tokens'] / 1_000_000 * PRICE_OUTPUT:.4f}")
    print(f"  Total cost  : ${overall_cost:.4f}")
    print("=" * 80)
    print(f"Detailed results: {CSV_OUT.absolute()}")
    print(f"Metrics summary: {METRICS_FILE.absolute()}")
    print(f"Detailed log:    {LOG_FILE.absolute()}")

if __name__ == "__main__":
    main()