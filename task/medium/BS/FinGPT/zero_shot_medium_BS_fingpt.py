#!/usr/bin/env python3
"""
zero_shot_medium_BS_fingpt.py

FinGPT zero-shot medium task — multi-class error classification on Balance Sheets.
Each file has AT MOST ONE violation; model must identify which rule is violated (or NO).
Model: FinGPT/fingpt-mt_llama2-7b_lora
"""

import logging, re, csv, torch, json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR  = REPO_ROOT / "data" / "single_rule_violation" / "BS_data"

BASE_MODEL     = "NousResearch/Llama-2-7b-chat-hf"
PEFT_MODEL     = "FinGPT/fingpt-mt_llama2-7b_lora"
MAX_NEW_TOKENS = 128   # medium needs more tokens for code + explanation

print("Loading FinGPT …")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, PEFT_MODEL)
model = model.eval()
print("Ready.\n")

INPUT_FILES = [
    DATA_DIR / "BS-00.md",
    DATA_DIR / "BS-01.md",
    DATA_DIR / "BS-02.md",
    DATA_DIR / "BS-03.md",
    DATA_DIR / "BS-04.md",
    DATA_DIR / "BS-05.md",
]

GROUND_TRUTH = {
    "BS-00.md": "NO",
    "BS-01.md": "BS01",
    "BS-02.md": "BS02",
    "BS-03.md": "BS03",
    "BS-04.md": "BS04",
    "BS-05.md": "BS05",
}

VALID_CODES = {"BS01", "BS02", "BS03", "BS04", "BS05", "NO"}

PROMPT_TEMPLATE = """You are a strict financial-statement auditor tasked with identifying violations in a balance sheet.

RULES (check in priority order - stop at the first violation found):

BS-00 (Baseline/Compliant): ALL rules below are satisfied.
  - Balance sheet equation holds: Total Assets = Total Liabilities + Total Shareholders' Equity
  - Uses correct "Current" / "Non-Current" terminology
  - Cash is properly labeled
  - Retained Earnings is clearly labeled
  - Treasury stock (if any) is correctly classified in equity as deduction
  - If a statement satisfies ALL rules, respond "NO"

BS-01: Total assets must equal total liabilities plus shareholders' equity.
  - Violation: Total Assets ≠ (Total Liabilities + Total Shareholders' Equity) — even $1 difference.

BS-02: Asset and liability subsections must use EXACTLY "Current" or "Non-Current" terminology.
  - Violation: Using "Present", "Operating", "Working", "Long-term", "Short-term" instead of "Current"/"Non-Current"

BS-03: Cash must be described using precise terminology.
  - Violation: Terms like "Funds", "Resources", "Working Capital", "Capital" used instead of "Cash"
  - Compliant: Uses only "Cash", "Cash and cash equivalents", or "Restricted cash"

BS-04: Retained earnings must be clearly labeled as "Retained Earnings".
  - Violation: "Capital Surplus", "General Reserve", "Accumulated Capital", or similar used instead
  - Violation: "Retained Earnings" line item completely missing from equity section

BS-05: Treasury stock must be reported as a deduction from shareholders' equity.
  - Violation: Treasury stock appears in the Assets section
  - Compliant: Treasury stock appears as a negative line item within equity section

Respond ONLY with ONE of: BS01, BS02, BS03, BS04, BS05, or NO
Followed by a colon and a brief explanation.

BALANCE SHEET DATA (begin):
{table_data}
BALANCE SHEET DATA (end)

Your response (ONE line only):"""


def parse_markdown_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f: content = f.read()
    tables, lines, current_company, current_date, table_lines, in_table = [], content.split("\n"), None, None, [], False
    for line in lines:
        if line.startswith("## ") or line.startswith("### "):
            if table_lines and current_company:
                tables.append({"company_name": current_company, "date": current_date, "data": "\n".join(table_lines)}); table_lines = []
            header = line.lstrip("#").strip()
            if "—" in header: parts = header.split("—"); current_company = parts[0].strip(); current_date = parts[1].strip()
            else: current_company = header; current_date = ""
            in_table = False
        elif line.strip().startswith("|"): in_table = True; table_lines.append(line)
        elif in_table and line.strip() == "": in_table = False
        elif in_table: table_lines.append(line)
    if table_lines and current_company:
        tables.append({"company_name": current_company, "date": current_date, "data": "\n".join(table_lines)})
    return tables


def extract_table_data(table_text):
    clean = []
    for line in table_text.split("\n"):
        if not line.strip(): continue
        line = line.strip().lstrip("|").rstrip("|")
        if re.match(r"^-+\s*\|\s*-+", line): continue
        clean.append(" | ".join(c.strip() for c in line.split("|")))
    return "\n".join(clean)


def generate_response(prompt_text):
    fingpt_prompt = f"Human: {prompt_text}\n\nAssistant:"
    inputs = tokenizer(fingpt_prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    new_tokens = output_ids.shape[1] - input_len
    response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    return response, {"prompt_tokens": input_len, "completion_tokens": new_tokens, "total_tokens": output_ids.shape[1]}


def parse_code_from_response(raw_reply):
    """Extract violation code from model response."""
    r = raw_reply.strip().upper()
    for code in ["BS01", "BS02", "BS03", "BS04", "BS05", "NO"]:
        if r.startswith(code):
            return code
    for code in ["BS01", "BS02", "BS03", "BS04", "BS05"]:
        if code in r:
            return code
    if "NO" in r.split(":")[0]:
        return "NO"
    return "NO"  # default: assume compliant if unclear


def calculate_multiclass_metrics(results_by_file):
    """Compute per-class and macro metrics for multi-class classification."""
    all_codes = ["NO", "BS01", "BS02", "BS03", "BS04", "BS05"]
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in all_codes}
    correct = total = 0

    for gt_code, pred_code in results_by_file:
        total += 1
        if pred_code == gt_code:
            correct += 1
        per_class[gt_code]["fn"] += (1 if pred_code != gt_code else 0)
        per_class[gt_code]["tp"] += (1 if pred_code == gt_code else 0)
        if pred_code != gt_code:
            per_class[pred_code]["fp"] += 1

    accuracy = correct / total if total > 0 else 0
    macro_p = macro_r = macro_f1 = 0
    n_classes = 0
    for c in all_codes:
        tp, fp, fn = per_class[c]["tp"], per_class[c]["fp"], per_class[c]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_class[c]["precision"] = p * 100
        per_class[c]["recall"] = r * 100
        per_class[c]["f1"] = f * 100
        if per_class[c]["tp"] + per_class[c]["fn"] > 0:
            macro_p += p; macro_r += r; macro_f1 += f; n_classes += 1

    if n_classes:
        macro_p /= n_classes; macro_r /= n_classes; macro_f1 /= n_classes

    return {"accuracy": accuracy * 100, "macro_precision": macro_p * 100,
            "macro_recall": macro_r * 100, "macro_f1": macro_f1 * 100,
            "per_class": per_class, "correct": correct, "total": total}


def main():
    import time
    log_file     = Path("zero_shot_medium_bs_fingpt_eval.log")
    csv_file     = Path("zero_shot_medium_bs_fingpt_eval_results.csv")
    metrics_file = Path("zero_shot_medium_bs_fingpt_eval_metrics.txt")
    logging.basicConfig(filename=str(log_file), level=logging.INFO, format="%(asctime)s - %(message)s")

    print("\n" + "=" * 110)
    print("ZERO-SHOT MEDIUM BS — FinGPT EVALUATION (Multi-class)")
    print("=" * 110)

    all_rows = []
    gt_pred_pairs = []
    total_tok = total_ptok = total_ctok = 0

    for filepath in INPUT_FILES:
        fp = Path(filepath)
        gt_code = GROUND_TRUTH.get(fp.name, "NO")
        print(f"\n{'-' * 110}\nFile: {fp.name}  (GT: {gt_code})\n{'-' * 110}")
        if not fp.exists():
            print(f"ERROR: {fp}"); continue
        tables = parse_markdown_file(fp)
        if not tables:
            print("WARNING: No tables"); continue
        print(f"Found {len(tables)} balance sheets")

        for idx, tbl in enumerate(tables, 1):
            comp = tbl.get("company_name", "Unknown"); date = tbl.get("date", "")
            print(f"  [{idx:3d}/{len(tables)}] {comp:30s} GT:{gt_code}...", end=" ", flush=True)
            try:
                ts = extract_table_data(tbl["data"])
                if not ts.strip(): print("SKIP(empty)"); continue
                prompt = PROMPT_TEMPLATE.format(table_data=ts)
                raw_reply, tok = generate_response(prompt)
                pred_code = parse_code_from_response(raw_reply)
                total_tok += tok["total_tokens"]; total_ptok += tok["prompt_tokens"]; total_ctok += tok["completion_tokens"]
                status = "✓" if pred_code == gt_code else "✗"
                print(f"{status} Pred:{pred_code}  Raw:{raw_reply[:60]}")
                gt_pred_pairs.append((gt_code, pred_code))
                all_rows.append({"file": fp.name, "company": comp, "date": date,
                                  "ground_truth": gt_code, "prediction": pred_code,
                                  "correct": int(pred_code == gt_code),
                                  "raw_response": raw_reply[:200], **tok})
                logging.info(f"{fp.name}|{comp}|GT:{gt_code}|Pred:{pred_code}")
            except Exception as e:
                print(f"ERROR:{e}"); logging.error(f"{fp.name}|{comp}|{e}")

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file","company","date","ground_truth","prediction","correct","raw_response","prompt_tokens","completion_tokens","total_tokens"])
        w.writeheader(); w.writerows(all_rows)

    print(f"\n{'=' * 110}\nSUMMARY\n{'=' * 110}")
    if gt_pred_pairs:
        metrics = calculate_multiclass_metrics(gt_pred_pairs)
        print(f"Accuracy: {metrics['accuracy']:.2f}%  Macro-Prec: {metrics['macro_precision']:.2f}%  Macro-Rec: {metrics['macro_recall']:.2f}%  Macro-F1: {metrics['macro_f1']:.2f}%")
        print(f"Correct: {metrics['correct']}/{metrics['total']}")
        print("\nPer-class breakdown:")
        for code in ["NO","BS01","BS02","BS03","BS04","BS05"]:
            pc = metrics["per_class"][code]
            print(f"  {code}: TP={pc['tp']} FP={pc['fp']} FN={pc['fn']}  P={pc['precision']:.1f}%  R={pc['recall']:.1f}%  F1={pc['f1']:.1f}%")
        print(f"\nTokens — Total:{total_tok:,}  Prompt:{total_ptok:,}  Completion:{total_ctok:,}")

        summary_lines = ["=" * 110, "ZERO-SHOT MEDIUM BS — FinGPT EVALUATION SUMMARY", "=" * 110,
                         f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
                         f"Accuracy: {metrics['accuracy']:.2f}%",
                         f"Macro-Precision: {metrics['macro_precision']:.2f}%",
                         f"Macro-Recall: {metrics['macro_recall']:.2f}%",
                         f"Macro-F1: {metrics['macro_f1']:.2f}%",
                         f"Correct: {metrics['correct']}/{metrics['total']}", "",
                         "Per-class:"]
        for code in ["NO","BS01","BS02","BS03","BS04","BS05"]:
            pc = metrics["per_class"][code]
            summary_lines.append(f"  {code}: TP={pc['tp']} FP={pc['fp']} FN={pc['fn']} P={pc['precision']:.1f}% R={pc['recall']:.1f}% F1={pc['f1']:.1f}%")
        summary_lines += ["", f"Tokens: Total:{total_tok:,} Prompt:{total_ptok:,} Completion:{total_ctok:,}", "=" * 110]
        with open(metrics_file, "w", encoding="utf-8") as f: f.write("\n".join(summary_lines))
        logging.info("\n" + "\n".join(summary_lines))
    print(f"Outputs: {csv_file} | {metrics_file} | {log_file}")


if __name__ == "__main__":
    main()
