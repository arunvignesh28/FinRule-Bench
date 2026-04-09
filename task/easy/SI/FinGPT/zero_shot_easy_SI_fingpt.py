#!/usr/bin/env python3
"""
zero_shot_easy_SI_fingpt.py — FinGPT zero-shot evaluation on Income Statements (easy).
- SI-00.md: COMPLIANT (T), SI-01 to SI-05.md: ERROR-INJECTED (F)
Model: FinGPT/fingpt-mt_llama2-7b_lora
"""

import logging, re, csv, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR  = REPO_ROOT / "data" / "single_rule_violation" / "SI_data" / "SI"
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

RULES = {
    "SI00": {"filename": DATA_DIR/"SI-00.md", "ground_truth": "T",
             "prompt": ("You are a strict financial auditor checking Income Statements.\n\n"
                        "RULE: ALL requirements must be satisfied:\n"
                        "1. Non-operating items appear AFTER operating expenses section\n"
                        "2. Depreciation and amortization expense is within operating expenses\n"
                        "3. Both basic AND diluted EPS presented on the face\n"
                        "4. No vague \"Miscellaneous\" line items\n"
                        "5. Interest expense presented separately as non-operating\n\n"
                        "Answer T if all satisfied, F if any violated.\n\n"
                        "INCOME STATEMENT DATA:\n{table_data}\n\nRespond ONLY with 'T' or 'F':")},
    "SI01": {"filename": DATA_DIR/"SI-01.md", "ground_truth": "F",
             "prompt": ("Financial auditor: Non-operating items MUST be separated from and appear AFTER operating expenses.\n\n"
                        "INCOME STATEMENT DATA:\n{table_data}\n\nRespond ONLY with 'T' or 'F':")},
    "SI02": {"filename": DATA_DIR/"SI-02.md", "ground_truth": "F",
             "prompt": ("Financial auditor: Depreciation expense MUST be in Operating Expenses, NOT in Non-Operating section.\n\n"
                        "INCOME STATEMENT DATA:\n{table_data}\n\nRespond ONLY with 'T' or 'F':")},
    "SI03": {"filename": DATA_DIR/"SI-03.md", "ground_truth": "F",
             "prompt": ("Financial auditor: Both Basic AND Diluted EPS MUST be presented on the income statement.\n\n"
                        "INCOME STATEMENT DATA:\n{table_data}\n\nRespond ONLY with 'T' or 'F':")},
    "SI04": {"filename": DATA_DIR/"SI-04.md", "ground_truth": "F",
             "prompt": ("Financial auditor: Vague \"Miscellaneous other income\" line items are prohibited — "
                        "all non-operating items must be clearly identified.\n\n"
                        "INCOME STATEMENT DATA:\n{table_data}\n\nRespond ONLY with 'T' or 'F':")},
    "SI05": {"filename": DATA_DIR/"SI-05.md", "ground_truth": "F",
             "prompt": ("Financial auditor: Interest expense MUST be presented separately in the Non-Operating section.\n\n"
                        "INCOME STATEMENT DATA:\n{table_data}\n\nRespond ONLY with 'T' or 'F':")},
}

def parse_markdown_file(filepath):
    with open(filepath,"r",encoding="utf-8") as f: content=f.read()
    tables,lines,current_company,current_date,table_lines,in_table=[],content.split("\n"),None,None,[],False
    for line in lines:
        if line.startswith("## ") or line.startswith("### "):
            if table_lines and current_company:
                tables.append({"company_name":current_company,"date":current_date,"data":"\n".join(table_lines)}); table_lines=[]
            header=line.lstrip("#").strip()
            if "—" in header: parts=header.split("—"); current_company=parts[0].strip(); current_date=parts[1].strip()
            else: current_company=header; current_date=""
            in_table=False
        elif line.strip().startswith("|"): in_table=True; table_lines.append(line)
        elif in_table and line.strip()=="": in_table=False
        elif in_table: table_lines.append(line)
    if table_lines and current_company:
        tables.append({"company_name":current_company,"date":current_date,"data":"\n".join(table_lines)})
    return tables

def extract_table_data(table_text):
    clean=[]
    for line in table_text.split("\n"):
        if not line.strip(): continue
        line=line.strip().lstrip("|").rstrip("|")
        if re.match(r"^-+\s*\|\s*-+",line): continue
        clean.append(" | ".join(c.strip() for c in line.split("|")))
    return "\n".join(clean)

def generate_response(prompt_text):
    fp=f"Human: {prompt_text}\n\nAssistant:"
    inputs=tokenizer(fp,return_tensors="pt",truncation=True,max_length=4096)
    inputs={k:v.to(model.device) for k,v in inputs.items()}
    il=inputs["input_ids"].shape[1]
    with torch.no_grad():
        oids=model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id)
    nt=oids.shape[1]-il
    resp=tokenizer.decode(oids[0][il:],skip_special_tokens=True).strip()
    return resp,{"prompt_tokens":il,"completion_tokens":nt,"total_tokens":oids.shape[1]}

def check_table(table_info,rule_code):
    ts=extract_table_data(table_info["data"])
    if not ts.strip(): raise ValueError("Empty table")
    prompt=RULES[rule_code]["prompt"].format(table_data=ts)
    raw,tok=generate_response(prompt)
    tf=raw[0].upper() if raw and raw[0].upper() in ("T","F") else "F"
    return tf,tok

def calculate_metrics(tp,tn,fp,fn):
    d=tp+tn+fp+fn
    a=(tp+tn)/d if d>0 else 0; p=tp/(tp+fp) if (tp+fp)>0 else 0
    r=tp/(tp+fn) if (tp+fn)>0 else 0; f=2*p*r/(p+r) if (p+r)>0 else 0
    return {k:v*100 for k,v in zip(["accuracy","precision","recall","f1"],[a,p,r,f])}

def main():
    import time
    log_file=Path("zero_shot_easy_si_fingpt_eval.log"); csv_file=Path("zero_shot_easy_si_fingpt_eval_results.csv"); metrics_file=Path("zero_shot_easy_si_fingpt_eval_metrics.txt")
    logging.basicConfig(filename=str(log_file),level=logging.INFO,format="%(asctime)s - %(message)s")
    rule_order=["SI00","SI01","SI02","SI03","SI04","SI05"]
    print("\n"+"="*110+"\nZERO-SHOT EASY SI — FinGPT EVALUATION\n"+"="*110)
    rule_results,all_rows={},[]
    for rc in rule_order:
        info=RULES[rc]; ip=Path(info["filename"]); gt=info["ground_truth"]
        print(f"\nProcessing: {rc} (GT:{gt})")
        if not ip.exists(): print(f"ERROR:{ip}"); continue
        tables=parse_markdown_file(ip)
        if not tables: print("WARNING:No tables"); continue
        rule_results[rc]={"processed":0,"tp":0,"tn":0,"fp":0,"fn":0,"total_tokens":0,"prompt_tokens":0,"completion_tokens":0}
        for idx,tbl in enumerate(tables,1):
            comp=tbl.get("company_name","Unknown"); date=tbl.get("date","")
            print(f"  [{idx:3d}/{len(tables)}] {comp:30s} GT:{gt}...",end=" ",flush=True)
            rule_results[rc]["processed"]+=1
            try:
                pred,tok=check_table(tbl,rc)
                for k in ("total_tokens","prompt_tokens","completion_tokens"): rule_results[rc][k]+=tok[k]
                if   gt=="T" and pred=="T": rule_results[rc]["tn"]+=1; status="✓TN"
                elif gt=="T" and pred=="F": rule_results[rc]["fp"]+=1; status="✗FP"
                elif gt=="F" and pred=="F": rule_results[rc]["tp"]+=1; status="✓TP"
                else:                       rule_results[rc]["fn"]+=1; status="✗FN"
                print(f"{status} Pred:{pred}")
                all_rows.append({"rule":rc,"company":comp,"date":date,"ground_truth":gt,"prediction":pred,
                                  "tp":int(gt=="F"and pred=="F"),"tn":int(gt=="T"and pred=="T"),
                                  "fp":int(gt=="T"and pred=="F"),"fn":int(gt=="F"and pred=="T"),**tok})
                logging.info(f"{rc}|{comp}|GT:{gt}|Pred:{pred}")
            except Exception as e: print(f"ERROR:{e}"); logging.error(f"{rc}|{comp}|{e}")
        s=rule_results[rc]
        if s["processed"]:
            m=calculate_metrics(s["tp"],s["tn"],s["fp"],s["fn"])
            print(f"  {rc}: Acc={m['accuracy']:.1f}% F1={m['f1']:.1f}%")
    with open(csv_file,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["rule","company","date","ground_truth","prediction","tp","tn","fp","fn","prompt_tokens","completion_tokens","total_tokens"])
        w.writeheader(); w.writerows(all_rows)
    tp_tot=tn_tot=fp_tot=fn_tot=p_tot=tok_tot=0
    for rc in rule_order:
        if rc not in rule_results: continue
        s=rule_results[rc]; p_tot+=s["processed"]; tp_tot+=s["tp"]; tn_tot+=s["tn"]; fp_tot+=s["fp"]; fn_tot+=s["fn"]; tok_tot+=s["total_tokens"]
        if s["processed"]:
            m=calculate_metrics(s["tp"],s["tn"],s["fp"],s["fn"]); print(f"{rc}: Acc={m['accuracy']:5.1f}% F1={m['f1']:5.1f}%")
    if p_tot:
        om=calculate_metrics(tp_tot,tn_tot,fp_tot,fn_tot); print(f"OVERALL: Acc={om['accuracy']:.2f}% F1={om['f1']:.2f}% Tokens:{tok_tot:,}")
    summary=["="*110,"ZERO-SHOT EASY SI — FinGPT","="*110,f"Timestamp:{time.strftime('%Y-%m-%d %H:%M:%S')}",""]
    for rc in rule_order:
        if rc not in rule_results: continue
        s=rule_results[rc]
        if not s["processed"]: continue
        m=calculate_metrics(s["tp"],s["tn"],s["fp"],s["fn"])
        summary+=[f"{rc}: Acc:{m['accuracy']:.2f}% Prec:{m['precision']:.2f}% Rec:{m['recall']:.2f}% F1:{m['f1']:.2f}%",f"  Tokens:{s['total_tokens']:,}",""]
    if p_tot:
        om=calculate_metrics(tp_tot,tn_tot,fp_tot,fn_tot); summary+=["OVERALL:",f"  Acc:{om['accuracy']:.2f}% F1:{om['f1']:.2f}%",f"  Tokens:{tok_tot:,}","="*110]
    with open(metrics_file,"w",encoding="utf-8") as f: f.write("\n".join(summary))
    logging.info("\n"+"\n".join(summary))
    print(f"Outputs: {csv_file} | {metrics_file} | {log_file}")

if __name__ == "__main__":
    main()
