#!/usr/bin/env python3
"""
zero_shot_medium_SE_fingpt.py — FinGPT zero-shot medium task on SE statements.
Model: FinGPT/fingpt-mt_llama2-7b_lora
"""
import logging, re, csv, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO_ROOT=Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR=REPO_ROOT/"data"/"single_rule_violation"/"SE_data"
BASE_MODEL="NousResearch/Llama-2-7b-chat-hf"; PEFT_MODEL="FinGPT/fingpt-mt_llama2-7b_lora"; MAX_NEW_TOKENS=128

print("Loading FinGPT …")
tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,trust_remote_code=True); tokenizer.pad_token=tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained(BASE_MODEL,torch_dtype=torch.float16,device_map="auto",trust_remote_code=True)
model=PeftModel.from_pretrained(model,PEFT_MODEL); model=model.eval(); print("Ready.\n")

INPUT_FILES=[DATA_DIR/f"SE-0{i}.md" for i in range(4)]
GROUND_TRUTH={"SE-00.md":"NO","SE-01.md":"SE01","SE-02.md":"SE02","SE-03.md":"SE03"}

PROMPT_TEMPLATE = """You are a strict auditor identifying EXACTLY ONE violation in a stockholders' equity statement.

RULES:
SE-00 (Compliant): All rules satisfied → respond "NO"
SE-01: Total Equity must equal the sum of all components for every year presented.
  Violation: Total ≠ (Common Stock + APIC + Retained Earnings + AOCI + NCI + Treasury Stock) for any year.
SE-02: Net income (or loss) must appear as an explicit line item in the retained earnings reconciliation.
  Violation: "Net income (or loss)" line MISSING from retained earnings section.
SE-03: Other Comprehensive Income (OCI) must appear as an explicit line in the AOCI reconciliation.
  Violation: "Other comprehensive income" line MISSING from AOCI section.

Check sequentially SE-01 → SE-02 → SE-03. Report the FIRST violation found.
Respond: <CODE>: <brief explanation>  (Codes: SE01, SE02, SE03, or NO)

STATEMENT OF EQUITY DATA (begin):
{table_data}
STATEMENT OF EQUITY DATA (end)

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

def extract_table_data(t):
    clean=[]
    for line in t.split("\n"):
        if not line.strip(): continue
        line=line.strip().lstrip("|").rstrip("|")
        if re.match(r"^-+\s*\|\s*-+",line): continue
        clean.append(" | ".join(c.strip() for c in line.split("|")))
    return "\n".join(clean)

def generate_response(prompt_text):
    fp=f"Human: {prompt_text}\n\nAssistant:"
    inputs=tokenizer(fp,return_tensors="pt",truncation=True,max_length=4096)
    inputs={k:v.to(model.device) for k,v in inputs.items()}; il=inputs["input_ids"].shape[1]
    with torch.no_grad():
        oids=model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id)
    nt=oids.shape[1]-il; resp=tokenizer.decode(oids[0][il:],skip_special_tokens=True).strip()
    return resp,{"prompt_tokens":il,"completion_tokens":nt,"total_tokens":oids.shape[1]}

def parse_code(r):
    r=r.strip().upper()
    for c in ["SE01","SE02","SE03","NO"]:
        if r.startswith(c): return c
    for c in ["SE01","SE02","SE03"]:
        if c in r: return c
    return "NO"

def calc_metrics(gtp,codes):
    pc={c:{"tp":0,"fp":0,"fn":0} for c in codes}; correct=total=0
    for gt,pred in gtp:
        total+=1
        if pred==gt: correct+=1; pc[gt]["tp"]+=1
        else: pc[gt]["fn"]+=1
        if pred!=gt and pred in pc: pc[pred]["fp"]+=1
    a=correct/total if total>0 else 0; mf=0; n=0
    for c in codes:
        tp,fp,fn=pc[c]["tp"],pc[c]["fp"],pc[c]["fn"]
        p=tp/(tp+fp) if (tp+fp)>0 else 0; r=tp/(tp+fn) if (tp+fn)>0 else 0; f=2*p*r/(p+r) if (p+r)>0 else 0
        pc[c]["precision"]=p*100; pc[c]["recall"]=r*100; pc[c]["f1"]=f*100
        if pc[c]["tp"]+pc[c]["fn"]>0: mf+=f; n+=1
    return {"accuracy":a*100,"macro_f1":(mf/n*100) if n>0 else 0,"correct":correct,"total":total,"per_class":pc}

def main():
    import time
    log_file=Path("zero_shot_medium_se_fingpt_eval.log"); csv_file=Path("zero_shot_medium_se_fingpt_eval_results.csv"); metrics_file=Path("zero_shot_medium_se_fingpt_eval_metrics.txt")
    logging.basicConfig(filename=str(log_file),level=logging.INFO,format="%(asctime)s - %(message)s")
    print("\n"+"="*110+"\nZERO-SHOT MEDIUM SE — FinGPT\n"+"="*110)
    all_rows=[]; gtp=[]; ttok=tptok=tctok=0
    for filepath in INPUT_FILES:
        fp=Path(filepath); gt=GROUND_TRUTH.get(fp.name,"NO")
        print(f"\nFile:{fp.name}  GT:{gt}")
        if not fp.exists(): print(f"ERROR:{fp}"); continue
        tables=parse_markdown_file(fp)
        if not tables: continue
        for idx,tbl in enumerate(tables,1):
            comp=tbl.get("company_name","Unknown"); date=tbl.get("date","")
            print(f"  [{idx:3d}/{len(tables)}] {comp:30s} GT:{gt}...",end=" ",flush=True)
            try:
                ts=extract_table_data(tbl["data"])
                if not ts.strip(): print("SKIP"); continue
                raw,tok=generate_response(PROMPT_TEMPLATE.format(table_data=ts))
                pred=parse_code(raw); ttok+=tok["total_tokens"]; tptok+=tok["prompt_tokens"]; tctok+=tok["completion_tokens"]
                print(f"{'✓' if pred==gt else '✗'} Pred:{pred}")
                gtp.append((gt,pred))
                all_rows.append({"file":fp.name,"company":comp,"date":date,"ground_truth":gt,"prediction":pred,"correct":int(pred==gt),"raw_response":raw[:200],**tok})
                logging.info(f"{fp.name}|{comp}|GT:{gt}|Pred:{pred}")
            except Exception as e: print(f"ERROR:{e}"); logging.error(f"{fp.name}|{comp}|{e}")
    with open(csv_file,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["file","company","date","ground_truth","prediction","correct","raw_response","prompt_tokens","completion_tokens","total_tokens"]); w.writeheader(); w.writerows(all_rows)
    if gtp:
        codes=["NO","SE01","SE02","SE03"]; m=calc_metrics(gtp,codes)
        print(f"\nAcc={m['accuracy']:.2f}%  Macro-F1={m['macro_f1']:.2f}%  {m['correct']}/{m['total']}  Tokens:{ttok:,}")
        summary=["="*110,"ZERO-SHOT MEDIUM SE — FinGPT","="*110,f"Timestamp:{time.strftime('%Y-%m-%d %H:%M:%S')}","",f"Acc:{m['accuracy']:.2f}%  Macro-F1:{m['macro_f1']:.2f}%","","Per-class:"]
        for c in codes: pc=m["per_class"][c]; summary.append(f"  {c}: TP={pc['tp']} FP={pc['fp']} FN={pc['fn']} F1={pc['f1']:.1f}%")
        summary+=["",f"Tokens:{ttok:,}","="*110]
        with open(metrics_file,"w",encoding="utf-8") as f: f.write("\n".join(summary))
        logging.info("\n"+"\n".join(summary))
    print(f"Outputs: {csv_file} | {metrics_file} | {log_file}")

if __name__ == "__main__":
    main()
