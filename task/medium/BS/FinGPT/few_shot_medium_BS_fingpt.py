#!/usr/bin/env python3
"""
few_shot_medium_BS_fingpt.py — FinGPT few-shot medium task on Balance Sheets.
Model: FinGPT/fingpt-mt_llama2-7b_lora
"""

import logging, re, csv, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR  = REPO_ROOT / "data" / "single_rule_violation" / "BS_data"
BASE_MODEL = "NousResearch/Llama-2-7b-chat-hf"
PEFT_MODEL = "FinGPT/fingpt-mt_llama2-7b_lora"
MAX_NEW_TOKENS = 128

print("Loading FinGPT …")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, PEFT_MODEL)
model = model.eval()
print("Ready.\n")

INPUT_FILES = [DATA_DIR/f"BS-0{i}.md" for i in range(6)]
GROUND_TRUTH = {"BS-00.md":"NO","BS-01.md":"BS01","BS-02.md":"BS02",
                "BS-03.md":"BS03","BS-04.md":"BS04","BS-05.md":"BS05"}
VALID_CODES = {"BS01","BS02","BS03","BS04","BS05","NO"}

FEW_SHOT_EXAMPLES = """
EXAMPLE 1 — BS01 (Accounting Equation Violation):
Assets | Current_Assets | Cash | 5000
Assets | Non_Current_Assets | Property | 10000
Assets |  | Total Assets | 18500
Liabilities | Current_Liabilities | Payables | 2000
Liabilities | Non_Current_Liabilities | Debt | 6000
Liabilities |  | Total Liabilities | 8000
Equity | Stockholders_Equity | Retained earnings | 10000
Equity |  | Total Equity | 10000
Response: BS01: Total Assets (18500) ≠ Liabilities (8000) + Equity (10000) = 18000. Equation does not balance.

EXAMPLE 2 — BS02 (Wrong Classification):
Assets | Present_Assets | Cash | 5000
Assets |  | Total Assets | 18000
Response: BS02: Uses "Present_Assets" instead of required "Current_Assets".

EXAMPLE 3 — BS03 (Cash Terminology):
Assets | Current_Assets | Funds and cash equivalents | 5000
Response: BS03: Uses "Funds and cash equivalents" instead of "Cash and cash equivalents".

EXAMPLE 4 — BS04 (Retained Earnings Label):
Equity | Stockholders_Equity | General Reserve | 10000
Response: BS04: Uses "General Reserve" instead of required "Retained earnings".

EXAMPLE 5 — NO (Compliant):
All rules satisfied: equation holds, Current labels correct, Cash terminology correct, Retained earnings present, Treasury stock in equity.
Response: NO: All five rules satisfied. Statement is compliant.
"""

PROMPT_TEMPLATE = FEW_SHOT_EXAMPLES + """
Now analyze the balance sheet below. Each statement contains AT MOST ONE violation.

RULES:
BS-01: Total Assets = Total Liabilities + Total Equity (exact)
BS-02: Use "Current_Assets" / "Current_Liabilities" (not Present/Operating/Short-term variants)
BS-03: Use "Cash and cash equivalents" (not Funds/Resources/Capital)
BS-04: Equity must contain "Retained earnings" (not General Reserve/Accumulated Capital)
BS-05: Treasury stock must be in Equity section (not Assets)
NO: All rules satisfied

Respond with ONE line: <CODE>: <brief explanation>

BALANCE SHEET DATA (begin):
{table_data}
BALANCE SHEET DATA (end)

Your response:"""


def parse_markdown_file(filepath):
    with open(filepath,"r",encoding="utf-8") as f: content=f.read()
    tables,lines,cc,cd,tl,it=[],content.split("\n"),None,None,[],False
    for line in lines:
        if line.startswith("## ") or line.startswith("### "):
            if tl and cc: tables.append({"company_name":cc,"date":cd,"data":"\n".join(tl)}); tl=[]
            h=line.lstrip("#").strip()
            if "—" in h: p=h.split("—"); cc=p[0].strip(); cd=p[1].strip()
            else: cc=h; cd=""
            it=False
        elif line.strip().startswith("|"): it=True; tl.append(line)
        elif it and line.strip()=="": it=False
        elif it: tl.append(line)
    if tl and cc: tables.append({"company_name":cc,"date":cd,"data":"\n".join(tl)})
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
    nt=oids.shape[1]-il; resp=tokenizer.decode(oids[0][il:],skip_special_tokens=True).strip()
    return resp,{"prompt_tokens":il,"completion_tokens":nt,"total_tokens":oids.shape[1]}

def parse_code(raw_reply):
    r=raw_reply.strip().upper()
    for c in ["BS01","BS02","BS03","BS04","BS05","NO"]:
        if r.startswith(c): return c
    for c in ["BS01","BS02","BS03","BS04","BS05"]:
        if c in r: return c
    return "NO"

def calculate_multiclass_metrics(gt_pred_pairs):
    all_codes=["NO","BS01","BS02","BS03","BS04","BS05"]
    pc={c:{"tp":0,"fp":0,"fn":0} for c in all_codes}
    correct=total=0
    for gt,pred in gt_pred_pairs:
        total+=1
        if pred==gt: correct+=1; pc[gt]["tp"]+=1
        else: pc[gt]["fn"]+=1; pc[pred]["fp"]+=1
    accuracy=correct/total if total>0 else 0
    macro_f1=0; n=0
    for c in all_codes:
        tp,fp,fn=pc[c]["tp"],pc[c]["fp"],pc[c]["fn"]
        p=tp/(tp+fp) if (tp+fp)>0 else 0; r=tp/(tp+fn) if (tp+fn)>0 else 0
        f=2*p*r/(p+r) if (p+r)>0 else 0
        pc[c]["precision"]=p*100; pc[c]["recall"]=r*100; pc[c]["f1"]=f*100
        if pc[c]["tp"]+pc[c]["fn"]>0: macro_f1+=f; n+=1
    macro_f1=(macro_f1/n*100) if n>0 else 0
    return {"accuracy":accuracy*100,"macro_f1":macro_f1,"correct":correct,"total":total,"per_class":pc}

def main():
    import time
    log_file=Path("few_shot_medium_bs_fingpt_eval.log"); csv_file=Path("few_shot_medium_bs_fingpt_eval_results.csv"); metrics_file=Path("few_shot_medium_bs_fingpt_eval_metrics.txt")
    logging.basicConfig(filename=str(log_file),level=logging.INFO,format="%(asctime)s - %(message)s")
    print("\n"+"="*110+"\nFEW-SHOT MEDIUM BS — FinGPT EVALUATION\n"+"="*110)
    all_rows=[]; gt_pred=[]; total_tok=total_ptok=total_ctok=0
    for filepath in INPUT_FILES:
        fp=Path(filepath); gt_code=GROUND_TRUTH.get(fp.name,"NO")
        print(f"\nFile:{fp.name}  GT:{gt_code}")
        if not fp.exists(): print(f"ERROR:{fp}"); continue
        tables=parse_markdown_file(fp)
        if not tables: print("WARNING:No tables"); continue
        for idx,tbl in enumerate(tables,1):
            comp=tbl.get("company_name","Unknown"); date=tbl.get("date","")
            print(f"  [{idx:3d}/{len(tables)}] {comp:30s} GT:{gt_code}...",end=" ",flush=True)
            try:
                ts=extract_table_data(tbl["data"])
                if not ts.strip(): print("SKIP"); continue
                prompt=PROMPT_TEMPLATE.format(table_data=ts)
                raw,tok=generate_response(prompt)
                pred=parse_code(raw)
                total_tok+=tok["total_tokens"]; total_ptok+=tok["prompt_tokens"]; total_ctok+=tok["completion_tokens"]
                status="✓" if pred==gt_code else "✗"
                print(f"{status} Pred:{pred}")
                gt_pred.append((gt_code,pred))
                all_rows.append({"file":fp.name,"company":comp,"date":date,"ground_truth":gt_code,"prediction":pred,"correct":int(pred==gt_code),"raw_response":raw[:200],**tok})
                logging.info(f"{fp.name}|{comp}|GT:{gt_code}|Pred:{pred}")
            except Exception as e: print(f"ERROR:{e}"); logging.error(f"{fp.name}|{comp}|{e}")
    with open(csv_file,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["file","company","date","ground_truth","prediction","correct","raw_response","prompt_tokens","completion_tokens","total_tokens"])
        w.writeheader(); w.writerows(all_rows)
    if gt_pred:
        m=calculate_multiclass_metrics(gt_pred)
        print(f"\nAcc={m['accuracy']:.2f}%  Macro-F1={m['macro_f1']:.2f}%  Correct={m['correct']}/{m['total']}")
        print(f"Tokens:{total_tok:,}")
        summary=["="*110,"FEW-SHOT MEDIUM BS — FinGPT","="*110,f"Timestamp:{time.strftime('%Y-%m-%d %H:%M:%S')}","",
                 f"Accuracy:{m['accuracy']:.2f}%  Macro-F1:{m['macro_f1']:.2f}%  Correct:{m['correct']}/{m['total']}","","Per-class:"]
        for code in ["NO","BS01","BS02","BS03","BS04","BS05"]:
            pc=m["per_class"][code]; summary.append(f"  {code}: TP={pc['tp']} FP={pc['fp']} FN={pc['fn']} P={pc['precision']:.1f}% R={pc['recall']:.1f}% F1={pc['f1']:.1f}%")
        summary+=["",f"Tokens:{total_tok:,}  Prompt:{total_ptok:,}  Completion:{total_ctok:,}","="*110]
        with open(metrics_file,"w",encoding="utf-8") as f: f.write("\n".join(summary))
        logging.info("\n"+"\n".join(summary))
    print(f"Outputs: {csv_file} | {metrics_file} | {log_file}")

if __name__ == "__main__":
    main()
