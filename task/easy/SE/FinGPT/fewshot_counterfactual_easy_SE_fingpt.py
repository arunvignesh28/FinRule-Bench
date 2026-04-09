#!/usr/bin/env python3
"""
fewshot_counterfactual_easy_SE_fingpt.py — FinGPT few-shot + counterfactual on SE (easy).
Model: FinGPT/fingpt-mt_llama2-7b_lora
"""

import logging, re, csv, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR  = REPO_ROOT / "data" / "single_rule_violation" / "SE_data"
BASE_MODEL = "NousResearch/Llama-2-7b-chat-hf"
PEFT_MODEL = "FinGPT/fingpt-mt_llama2-7b_lora"
MAX_NEW_TOKENS = 64

print("Loading FinGPT …")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, PEFT_MODEL)
model = model.eval()
print("Ready.\n")

PROMPT_SUFFIX = "\nRespond 'T' if compliant, 'F' if non-compliant. Answer: "

RULES = {
    "SE00": {"filename": DATA_DIR / "SE-00.md", "ground_truth": "T",
             "prompt_prefix": (
                 "Financial auditor evaluating ALL Stockholders' Equity rules.\n\n"
                 "✓ COMPLIANT: Totals balance, Net income present, OCI present → T (any violation → F)\n"
                 "✗ NON-COMPLIANT: Total ≠ components OR missing Net income OR missing OCI → F (fix all → T)\n\n"
                 "Evaluate:\n\nSTATEMENT OF EQUITY DATA:\n")},
    "SE01": {"filename": DATA_DIR / "SE-01.md", "ground_truth": "F",
             "prompt_prefix": (
                 "Financial auditor: Total Equity MUST equal sum of components.\n\n"
                 "✓ Total = 12,000 = components sum → T (If total changed → violation)\n"
                 "✗ Total = 12,500 ≠ components (off by 500) → F (If total corrected → compliant)\n\n"
                 "Evaluate:\n\nSTATEMENT OF EQUITY DATA:\n")},
    "SE02": {"filename": DATA_DIR / "SE-02.md", "ground_truth": "F",
             "prompt_prefix": (
                 "Financial auditor: Retained earnings MUST show \"Net income (loss)\" line.\n\n"
                 "✓ Retained earnings shows \"Net income: 1,234\" ✓ → T (If removed → violation)\n"
                 "✗ No \"Net income\" line in retained earnings section ✗ → F (If added → compliant)\n\n"
                 "Evaluate:\n\nSTATEMENT OF EQUITY DATA:\n")},
    "SE03": {"filename": DATA_DIR / "SE-03.md", "ground_truth": "F",
             "prompt_prefix": (
                 "Financial auditor: AOCI section MUST show \"Other comprehensive income\" line.\n\n"
                 "✓ AOCI shows \"Other comprehensive income: 456\" ✓ → T (If removed → violation)\n"
                 "✗ No OCI line in AOCI reconciliation ✗ → F (If added → compliant)\n\n"
                 "Evaluate:\n\nSTATEMENT OF EQUITY DATA:\n")},
}

def parse_markdown_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f: content = f.read()
    tables, lines, current_company, current_date, table_lines, in_table = [], content.split("\n"), None, None, [], False
    for line in lines:
        if line.startswith("## ") or line.startswith("### "):
            if table_lines and current_company:
                tables.append({"company_name": current_company, "date": current_date, "data": "\n".join(table_lines)})
                table_lines = []
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
    clean_lines = []
    for line in table_text.split("\n"):
        if not line.strip(): continue
        line = line.strip().lstrip("|").rstrip("|")
        if re.match(r"^-+\s*\|\s*-+", line): continue
        clean_lines.append(" | ".join(c.strip() for c in line.split("|")))
    return "\n".join(clean_lines)

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

def check_table(table_info, rule_code):
    table_string = extract_table_data(table_info["data"])
    if not table_string.strip(): raise ValueError("Empty table")
    prompt = RULES[rule_code]["prompt_prefix"] + table_string + PROMPT_SUFFIX
    raw_reply, token_usage = generate_response(prompt)
    tf = raw_reply[0].upper() if raw_reply and raw_reply[0].upper() in ("T", "F") else "F"
    return tf, token_usage

def calculate_metrics(tp, tn, fp, fn):
    d = tp + tn + fp + fn
    accuracy  = (tp + tn) / d if d > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {k: v * 100 for k, v in zip(["accuracy","precision","recall","f1"],[accuracy,precision,recall,f1])}

def main():
    import time
    log_file = Path("fewshot_cf_easy_se_fingpt_eval.log")
    csv_file = Path("fewshot_cf_easy_se_fingpt_eval_results.csv")
    metrics_file = Path("fewshot_cf_easy_se_fingpt_eval_metrics.txt")
    logging.basicConfig(filename=str(log_file), level=logging.INFO, format="%(asctime)s - %(message)s")
    rule_order = ["SE00","SE01","SE02","SE03"]
    print("\n" + "="*110 + "\nFEW-SHOT + COUNTERFACTUAL EASY SE — FinGPT\n" + "="*110)
    rule_results, all_rows = {}, []
    for rule_code in rule_order:
        info = RULES[rule_code]; input_path = Path(info["filename"]); gt = info["ground_truth"]
        print(f"\nProcessing: {rule_code} (GT:{gt})")
        if not input_path.exists(): print(f"ERROR: {input_path}"); continue
        tables = parse_markdown_file(input_path)
        if not tables: print("WARNING: No tables"); continue
        rule_results[rule_code] = {"processed":0,"tp":0,"tn":0,"fp":0,"fn":0,"total_tokens":0,"prompt_tokens":0,"completion_tokens":0}
        for idx, tbl in enumerate(tables, 1):
            comp = tbl.get("company_name","Unknown"); date = tbl.get("date","")
            print(f"  [{idx:3d}/{len(tables)}] {comp:30s} GT:{gt}...", end=" ", flush=True)
            rule_results[rule_code]["processed"] += 1
            try:
                pred, tok = check_table(tbl, rule_code)
                for k in ("total_tokens","prompt_tokens","completion_tokens"): rule_results[rule_code][k] += tok[k]
                if   gt=="T" and pred=="T": rule_results[rule_code]["tn"]+=1; status="✓TN"
                elif gt=="T" and pred=="F": rule_results[rule_code]["fp"]+=1; status="✗FP"
                elif gt=="F" and pred=="F": rule_results[rule_code]["tp"]+=1; status="✓TP"
                else:                       rule_results[rule_code]["fn"]+=1; status="✗FN"
                print(f"{status} Pred:{pred}")
                all_rows.append({"rule":rule_code,"company":comp,"date":date,"ground_truth":gt,"prediction":pred,
                                  "tp":int(gt=="F"and pred=="F"),"tn":int(gt=="T"and pred=="T"),
                                  "fp":int(gt=="T"and pred=="F"),"fn":int(gt=="F"and pred=="T"),**tok})
                logging.info(f"{rule_code}|{comp}|GT:{gt}|Pred:{pred}")
            except Exception as e: print(f"ERROR:{e}"); logging.error(f"{rule_code}|{comp}|{e}")
        s = rule_results[rule_code]
        if s["processed"]:
            m = calculate_metrics(s["tp"],s["tn"],s["fp"],s["fn"])
            print(f"  {rule_code}: Acc={m['accuracy']:.1f}% F1={m['f1']:.1f}%")
    with open(csv_file,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["rule","company","date","ground_truth","prediction","tp","tn","fp","fn","prompt_tokens","completion_tokens","total_tokens"])
        w.writeheader(); w.writerows(all_rows)
    total_p=total_tp=total_tn=total_fp=total_fn=total_tok=0
    for rc in rule_order:
        if rc not in rule_results: continue
        s=rule_results[rc]
        total_p+=s["processed"];total_tp+=s["tp"];total_tn+=s["tn"];total_fp+=s["fp"];total_fn+=s["fn"];total_tok+=s["total_tokens"]
        if s["processed"]:
            m=calculate_metrics(s["tp"],s["tn"],s["fp"],s["fn"])
            print(f"{rc}: Acc={m['accuracy']:5.1f}% F1={m['f1']:5.1f}%")
    if total_p:
        om=calculate_metrics(total_tp,total_tn,total_fp,total_fn)
        print(f"OVERALL: Acc={om['accuracy']:.2f}% F1={om['f1']:.2f}% Tokens:{total_tok:,}")
    summary=["="*110,"FEW-SHOT+CF EASY SE — FinGPT","="*110,f"Timestamp:{time.strftime('%Y-%m-%d %H:%M:%S')}",""]
    for rc in rule_order:
        if rc not in rule_results: continue
        s=rule_results[rc]
        if not s["processed"]: continue
        m=calculate_metrics(s["tp"],s["tn"],s["fp"],s["fn"])
        summary+=[f"{rc}: Acc:{m['accuracy']:.2f}% Prec:{m['precision']:.2f}% Rec:{m['recall']:.2f}% F1:{m['f1']:.2f}%",f"  Tokens:{s['total_tokens']:,}",""]
    if total_p:
        om=calculate_metrics(total_tp,total_tn,total_fp,total_fn)
        summary+=["OVERALL:",f"  Acc:{om['accuracy']:.2f}% F1:{om['f1']:.2f}%",f"  Tokens:{total_tok:,}","="*110]
    with open(metrics_file,"w",encoding="utf-8") as f: f.write("\n".join(summary))
    logging.info("\n"+"\n".join(summary))
    print(f"Outputs: {csv_file} | {metrics_file} | {log_file}")

if __name__ == "__main__":
    main()
