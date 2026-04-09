#!/usr/bin/env python3
"""
fewshot_counterfactual_easy_CF_fingpt.py

FinGPT few-shot + counterfactual evaluation on Cash Flow statements (easy task).
Model: FinGPT/fingpt-mt_llama2-7b_lora
"""

import logging
import re
import csv
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR  = REPO_ROOT / "data" / "single_rule_violation" / "CF_data"

BASE_MODEL     = "NousResearch/Llama-2-7b-chat-hf"
PEFT_MODEL     = "FinGPT/fingpt-mt_llama2-7b_lora"
MAX_NEW_TOKENS = 64

print("Loading FinGPT tokenizer and model …")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, PEFT_MODEL)
model = model.eval()
print("Model ready.\n")

PROMPT_SUFFIX = (
    "\nRespond with 'T' if compliant, 'F' if non-compliant.\n"
    "Ensure your decision is consistent with the counterfactual test.\nAnswer: "
)

RULES = {
    "CF00": {
        "filename": DATA_DIR / "CF-00.md",
        "ground_truth": "T",
        "prompt_prefix": (
            "You are a strict financial auditor evaluating ALL cash flow statement rules.\n\n"
            "EXAMPLES:\n\n"
            "✓ COMPLIANT - All rules satisfied:\n"
            "All three activity sections present: Operating, Investing, Financing (CF-01 ✓)\n"
            "No \"Cash Flow per Share\" line (CF-02 ✓)\n"
            "Uses \"Cash and cash equivalents\" terminology (CF-03 ✓)\n"
            "→ T (Any rule violation → F)\n\n"
            "✗ NON-COMPLIANT - Example violations:\n"
            "Missing \"Investing Activities\" section (CF-01 ✗)\n"
            "OR contains \"Cash Flow per Share (non-GAAP)\" line (CF-02 ✗)\n"
            "OR uses \"Funds at end of period\" instead of \"Cash\" (CF-03 ✗)\n"
            "→ F (Fixing all violations → T)\n\n"
            "Evaluate:\n\nSTATEMENT OF CASH FLOWS DATA:\n"
        ),
    },
    "CF01": {
        "filename": DATA_DIR / "CF-01.md",
        "ground_truth": "F",
        "prompt_prefix": (
            "Financial auditor evaluating: Cash flow statement MUST have Operating, Investing, and Financing sections.\n\n"
            "EXAMPLES:\n\n"
            "✓ COMPLIANT:\n"
            "Statement has all three sections: Operating, Investing, Financing Activities ✓\n"
            "→ T (If any section removed → violation)\n\n"
            "✗ NON-COMPLIANT:\n"
            "Statement missing \"Investing Activities\" section ✗\n"
            "→ F (If \"Investing Activities\" section added back → compliant)\n\n"
            "Evaluate:\n\nSTATEMENT OF CASH FLOWS DATA:\n"
        ),
    },
    "CF02": {
        "filename": DATA_DIR / "CF-02.md",
        "ground_truth": "F",
        "prompt_prefix": (
            "Financial auditor evaluating: \"Cash Flow per Share\" is a PROHIBITED non-GAAP metric.\n\n"
            "EXAMPLES:\n\n"
            "✓ COMPLIANT:\n"
            "Statement has no \"Cash Flow per Share\" line ✓\n"
            "→ T (If \"Cash Flow per Share\" line added → violation)\n\n"
            "✗ NON-COMPLIANT:\n"
            "Statement contains \"Cash Flow per Share (non-GAAP): 3.45\" ✗\n"
            "→ F (If that line removed → compliant)\n\n"
            "Evaluate:\n\nSTATEMENT OF CASH FLOWS DATA:\n"
        ),
    },
    "CF03": {
        "filename": DATA_DIR / "CF-03.md",
        "ground_truth": "F",
        "prompt_prefix": (
            "Financial auditor evaluating: MUST use \"Cash\" or \"Cash and cash equivalents\" (NOT Funds/Resources).\n\n"
            "EXAMPLES:\n\n"
            "✓ COMPLIANT:\n"
            "Uses \"Net increase in cash and cash equivalents: 1,234\" ✓\n"
            "→ T (If \"cash\" replaced with \"funds\" → violation)\n\n"
            "✗ NON-COMPLIANT:\n"
            "Uses \"Net increase in funds and fund equivalents: 1,234\" ✗\n"
            "→ F (If \"funds\" replaced with \"cash\" → compliant)\n\n"
            "Evaluate:\n\nSTATEMENT OF CASH FLOWS DATA:\n"
        ),
    },
}


def parse_markdown_file(filepath: Path) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    tables, lines = [], content.split("\n")
    current_company = current_date = None
    table_lines, in_table = [], False
    for line in lines:
        if line.startswith("## ") or line.startswith("### "):
            if table_lines and current_company:
                tables.append({"company_name": current_company, "date": current_date,
                                "data": "\n".join(table_lines)})
                table_lines = []
            header = line.lstrip("#").strip()
            if "—" in header:
                parts = header.split("—"); current_company = parts[0].strip(); current_date = parts[1].strip()
            else:
                current_company = header; current_date = ""
            in_table = False
        elif line.strip().startswith("|"):
            in_table = True; table_lines.append(line)
        elif in_table and line.strip() == "":
            in_table = False
        elif in_table:
            table_lines.append(line)
    if table_lines and current_company:
        tables.append({"company_name": current_company, "date": current_date, "data": "\n".join(table_lines)})
    return tables


def extract_table_data(table_text: str) -> str:
    clean_lines = []
    for line in table_text.split("\n"):
        if not line.strip():
            continue
        line = line.strip().lstrip("|").rstrip("|")
        if re.match(r"^-+\s*\|\s*-+", line):
            continue
        clean_lines.append(" | ".join(c.strip() for c in line.split("|")))
    return "\n".join(clean_lines)


def generate_response(prompt_text: str):
    fingpt_prompt = f"Human: {prompt_text}\n\nAssistant:"
    inputs = tokenizer(fingpt_prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids.shape[1] - input_len
    response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    return response, {"prompt_tokens": input_len, "completion_tokens": new_tokens,
                      "total_tokens": output_ids.shape[1]}


def check_table(table_info: dict, rule_code: str) -> tuple:
    table_string = extract_table_data(table_info["data"])
    if not table_string.strip():
        raise ValueError("Table data is empty")
    prompt = RULES[rule_code]["prompt_prefix"] + table_string + PROMPT_SUFFIX
    raw_reply, token_usage = generate_response(prompt)
    tf = raw_reply[0].upper() if raw_reply and raw_reply[0].upper() in ("T", "F") else "F"
    return tf, token_usage


def calculate_metrics(tp, tn, fp, fn):
    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"accuracy": accuracy * 100, "precision": precision * 100,
            "recall": recall * 100, "f1": f1 * 100}


def main():
    import time
    log_file     = Path("fewshot_cf_easy_cf_fingpt_eval.log")
    csv_file     = Path("fewshot_cf_easy_cf_fingpt_eval_results.csv")
    metrics_file = Path("fewshot_cf_easy_cf_fingpt_eval_metrics.txt")
    logging.basicConfig(filename=str(log_file), level=logging.INFO, format="%(asctime)s - %(message)s")
    rule_order = ["CF00", "CF01", "CF02", "CF03"]

    print("\n" + "=" * 110)
    print("FEW-SHOT + COUNTERFACTUAL EASY CF — FinGPT EVALUATION")
    print("=" * 110)

    rule_results, all_rows = {}, []

    for rule_code in rule_order:
        rule_info    = RULES[rule_code]
        input_path   = Path(rule_info["filename"])
        ground_truth = rule_info["ground_truth"]
        print(f"\n{'-' * 110}\nProcessing: {rule_code}  (GT: {ground_truth})\n{'-' * 110}")
        if not input_path.exists():
            print(f"ERROR: File not found: {input_path}"); continue
        tables = parse_markdown_file(input_path)
        if not tables:
            print("WARNING: No tables found"); continue
        print(f"Found {len(tables)} cash flow statements")
        rule_results[rule_code] = {"processed": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0,
                                   "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        for idx, tbl in enumerate(tables, 1):
            comp = tbl.get("company_name", "Unknown"); date = tbl.get("date", "")
            print(f"  [{idx:3d}/{len(tables)}] {comp:30s} ({date:20s}) GT:{ground_truth}...", end=" ", flush=True)
            rule_results[rule_code]["processed"] += 1
            try:
                pred, tok = check_table(tbl, rule_code)
                for k in ("total_tokens", "prompt_tokens", "completion_tokens"):
                    rule_results[rule_code][k] += tok[k]
                if   ground_truth == "T" and pred == "T": rule_results[rule_code]["tn"] += 1; status = "✓TN"
                elif ground_truth == "T" and pred == "F": rule_results[rule_code]["fp"] += 1; status = "✗FP"
                elif ground_truth == "F" and pred == "F": rule_results[rule_code]["tp"] += 1; status = "✓TP"
                else:                                      rule_results[rule_code]["fn"] += 1; status = "✗FN"
                print(f"{status} Pred:{pred}")
                all_rows.append({"rule": rule_code, "company": comp, "date": date,
                                  "ground_truth": ground_truth, "prediction": pred,
                                  "tp": int(ground_truth=="F" and pred=="F"),
                                  "tn": int(ground_truth=="T" and pred=="T"),
                                  "fp": int(ground_truth=="T" and pred=="F"),
                                  "fn": int(ground_truth=="F" and pred=="T"),
                                  **tok})
                logging.info(f"{rule_code}|{comp}|{date}|GT:{ground_truth}|Pred:{pred}")
            except Exception as e:
                print(f"ERROR: {e}"); logging.error(f"{rule_code}|{comp}|{e}")
        s = rule_results[rule_code]
        if s["processed"]:
            m = calculate_metrics(s["tp"], s["tn"], s["fp"], s["fn"])
            print(f"\n  {rule_code}: Acc={m['accuracy']:.1f}% F1={m['f1']:.1f}%")

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rule","company","date","ground_truth","prediction",
                                               "tp","tn","fp","fn","prompt_tokens","completion_tokens","total_tokens"])
        writer.writeheader(); writer.writerows(all_rows)

    total_p = total_tp = total_tn = total_fp = total_fn = 0
    total_tok = total_ptok = total_ctok = 0
    print(f"\n{'=' * 110}\nSUMMARY\n{'=' * 110}")
    for rule_code in rule_order:
        if rule_code not in rule_results: continue
        s = rule_results[rule_code]
        total_p += s["processed"]; total_tp += s["tp"]; total_tn += s["tn"]
        total_fp += s["fp"]; total_fn += s["fn"]
        total_tok += s["total_tokens"]; total_ptok += s["prompt_tokens"]; total_ctok += s["completion_tokens"]
        if s["processed"]:
            m = calculate_metrics(s["tp"], s["tn"], s["fp"], s["fn"])
            print(f"{rule_code}: Acc={m['accuracy']:5.1f}% F1={m['f1']:5.1f}%")
    if total_p:
        om = calculate_metrics(total_tp, total_tn, total_fp, total_fn)
        print(f"OVERALL: Acc={om['accuracy']:.2f}% F1={om['f1']:.2f}%  Tokens:{total_tok:,}")

    summary_lines = ["=" * 110, "FEW-SHOT + CF EASY CF — FinGPT SUMMARY", "=" * 110,
                     f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for rule_code in rule_order:
        if rule_code not in rule_results: continue
        s = rule_results[rule_code]
        if not s["processed"]: continue
        m = calculate_metrics(s["tp"], s["tn"], s["fp"], s["fn"])
        summary_lines += [f"{rule_code}: Acc:{m['accuracy']:.2f}% Prec:{m['precision']:.2f}% Rec:{m['recall']:.2f}% F1:{m['f1']:.2f}%",
                          f"  Tokens:{s['total_tokens']:,}", ""]
    if total_p:
        om = calculate_metrics(total_tp, total_tn, total_fp, total_fn)
        summary_lines += ["OVERALL:", f"  Acc:{om['accuracy']:.2f}% Prec:{om['precision']:.2f}% Rec:{om['recall']:.2f}% F1:{om['f1']:.2f}%",
                          f"  Tokens:{total_tok:,}", "=" * 110]
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    logging.info("\n" + "\n".join(summary_lines))
    print(f"Outputs: {csv_file}  |  {metrics_file}  |  {log_file}")


if __name__ == "__main__":
    main()
