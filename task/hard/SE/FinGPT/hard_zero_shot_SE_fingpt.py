#!/usr/bin/env python3
"""
hard_zero_shot_SE_fingpt.py — FinGPT zero-shot hard multi-error SE evaluation.
Model: FinGPT/fingpt-mt_llama2-7b_lora
"""
import json, re, csv, logging, torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO_ROOT=Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR=REPO_ROOT/"data"/"multi_rule_violation"/"SE"
INPUT_MD=DATA_DIR/"SE-MIXED-100.md"; INPUT_TRUTH=DATA_DIR/"SE-MIXED-100-truth.json"
BASE_MODEL="NousResearch/Llama-2-7b-chat-hf"; PEFT_MODEL="FinGPT/fingpt-mt_llama2-7b_lora"; MAX_NEW_TOKENS=128
VALID_CODES={"SE01","SE02","SE03"}

print("Loading FinGPT …")
tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,trust_remote_code=True); tokenizer.pad_token=tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained(BASE_MODEL,torch_dtype=torch.float16,device_map="auto",trust_remote_code=True)
model=PeftModel.from_pretrained(model,PEFT_MODEL); model=model.eval(); print("Ready.\n")

HARD_PROMPT="""You are a strict financial auditor. Examine the stockholders' equity statement and identify ALL violations.

RULES (check ALL):
SE01: Total Equity must equal the sum of ALL components (Common Stock + APIC + Retained Earnings + AOCI + NCI + Treasury Stock) for every year.
  Violation: Total ≠ sum of components for ANY year presented.

SE02: Net income (or loss) must appear as an EXPLICIT line item in the retained earnings reconciliation.
  Violation: "Net income (or loss)" line is MISSING from the retained earnings section.

SE03: Other Comprehensive Income (OCI) must appear as an EXPLICIT line in the AOCI reconciliation.
  Violation: "Other comprehensive income" line is MISSING from the AOCI section.

MULTIPLE violations may exist simultaneously.

RESPONSE FORMAT (one line only):
YES: [SE01, SE02] — if violations found (list ALL applicable codes)
NO: [] — if fully compliant

STATEMENT OF STOCKHOLDERS' EQUITY (begin):
{table}
STATEMENT OF STOCKHOLDERS' EQUITY (end)

Your response:""".strip()


def parse_markdown_companies(md_path):
    text=md_path.read_text(encoding="utf-8"); lines=text.splitlines()
    entries=[]; cur_header=None; cur_lines=[]
    for line in lines:
        if line.startswith("## ") or line.startswith("### "):
            if cur_header and cur_lines: entries.append({"company":cur_header.strip(),"table":"\n".join(cur_lines).strip()})
            cur_header=line.lstrip("# ").strip(); cur_lines=[]; continue
        if cur_header is not None: cur_lines.append(line)
    if cur_header and cur_lines: entries.append({"company":cur_header.strip(),"table":"\n".join(cur_lines).strip()})
    return entries

def load_ground_truth(json_path):
    j=json.loads(json_path.read_text(encoding="utf-8")); gt=j.get("ground_truth",j); mapping={}
    for comp,info in gt.items():
        if isinstance(info,dict) and "errors" in info: mapping[comp]=info["errors"]
        elif isinstance(info,list): mapping[comp]=info
        else: mapping[comp]=[]
    return mapping

def generate_response(prompt_text):
    fp=f"Human: {prompt_text}\n\nAssistant:"
    inputs=tokenizer(fp,return_tensors="pt",truncation=True,max_length=4096)
    inputs={k:v.to(model.device) for k,v in inputs.items()}; il=inputs["input_ids"].shape[1]
    with torch.no_grad():
        oids=model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id)
    nt=oids.shape[1]-il; resp=tokenizer.decode(oids[0][il:],skip_special_tokens=True).strip()
    return resp,{"prompt_tokens":il,"completion_tokens":nt,"total_tokens":oids.shape[1]}

def extract_codes(reply):
    r=reply.strip().upper(); is_yes=r.startswith("YES")
    found=set(re.findall(r'\bSE0[1-3]\b',r)); return is_yes,found

def classify_error(pred_codes,exp_codes,pred_yes,exp_yes):
    tp=pred_codes&exp_codes; fp=pred_codes-exp_codes; fn=exp_codes-pred_codes
    if pred_yes!=exp_yes: return "STEP1_ERROR"
    if tp and not fn and not fp: return "CORRECT"
    if fn and not tp and not fp: return "FALSE_NEGATIVE"
    if fp and not fn and not tp: return "FALSE_POSITIVE"
    if tp and fn and not fp: return "PARTIAL_DETECTION"
    if tp and fp and not fn: return "PARTIAL_HALLUCINATION"
    return "MIXED_ERRORS"

def main():
    import time
    log_file=Path("hard_zero_shot_se_fingpt_eval.log"); csv_file=Path("hard_zero_shot_se_fingpt_eval_results.csv"); metrics_file=Path("hard_zero_shot_se_fingpt_eval_metrics.txt")
    logging.basicConfig(filename=str(log_file),level=logging.INFO,format="%(asctime)s - %(message)s")
    print("\n"+"="*100+"\nHARD ZERO-SHOT SE — FinGPT\n"+"="*100)
    if not INPUT_MD.exists(): raise FileNotFoundError(INPUT_MD)
    if not INPUT_TRUTH.exists(): raise FileNotFoundError(INPUT_TRUTH)
    companies=parse_markdown_companies(INPUT_MD); truth_map=load_ground_truth(INPUT_TRUTH)
    rows=[]; s1=[]; s2=[]; ttok=0; err_dist=defaultdict(int)
    for idx,entry in enumerate(companies,1):
        comp=entry["company"]; tbl=entry["table"]
        if comp not in truth_map: continue
        exp_codes={c.upper() for c in truth_map[comp]} & VALID_CODES; exp_yes=len(exp_codes)>0
        print(f"[{idx:3d}] {comp:40s} GT:{'YES' if exp_yes else 'NO'} {sorted(exp_codes)}",end=" ",flush=True)
        try:
            raw,tok=generate_response(HARD_PROMPT.format(table=tbl))
            pred_yes,pred_codes=extract_codes(raw); pred_codes&=VALID_CODES; ttok+=tok["total_tokens"]
            tp=pred_codes&exp_codes; fp=pred_codes-exp_codes; fn=exp_codes-pred_codes; exact=pred_codes==exp_codes
            etype=classify_error(pred_codes,exp_codes,pred_yes,exp_yes); err_dist[etype]+=1
            print(f"{'✓' if exact else '✗'} Pred:{'YES' if pred_yes else 'NO'} {sorted(pred_codes)} [{etype}]")
            rows.append({"company":comp,"expected_yes":exp_yes,"predicted_yes":pred_yes,"s1_correct":pred_yes==exp_yes,
                         "step2_expected":",".join(sorted(exp_codes)),"step2_predicted":",".join(sorted(pred_codes)),
                         "exact_match":exact,"tp":len(tp),"fp":len(fp),"fn":len(fn),"error_type":etype,"raw_response":raw[:300],**tok})
            s1.append({"correct":pred_yes==exp_yes}); s2.append({"exact":exact,"tp":len(tp),"fp":len(fp),"fn":len(fn),"prec":len(tp)/(len(tp)+len(fp)) if (len(tp)+len(fp))>0 else (1.0 if not exp_codes else 0.0),"rec":len(tp)/(len(tp)+len(fn)) if (len(tp)+len(fn))>0 else (1.0 if not pred_codes else 0.0)})
            logging.info(f"{comp}|GT:{sorted(exp_codes)}|Pred:{sorted(pred_codes)}|{etype}")
        except Exception as e: print(f"ERROR:{e}"); logging.error(f"{comp}|{e}")
    with open(csv_file,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["company","expected_yes","predicted_yes","s1_correct","step2_expected","step2_predicted","exact_match","tp","fp","fn","error_type","raw_response","prompt_tokens","completion_tokens","total_tokens"])
        w.writeheader(); w.writerows(rows)
    if s1:
        n=len(s1); s1c=sum(r["correct"] for r in s1); s1a=s1c/n
        tot_tp=sum(r["tp"] for r in s2); tot_fp=sum(r["fp"] for r in s2); tot_fn=sum(r["fn"] for r in s2)
        mp=tot_tp/(tot_tp+tot_fp) if (tot_tp+tot_fp)>0 else 0; mr=tot_tp/(tot_tp+tot_fn) if (tot_tp+tot_fn)>0 else 0; mf=2*mp*mr/(mp+mr) if (mp+mr)>0 else 0
        em=sum(r["exact"] for r in s2)/n
        print(f"\nStep1 Acc={s1a*100:.2f}%  ExactMatch={em*100:.2f}%  MicroF1={mf*100:.2f}%  Tokens:{ttok:,}")
        summary=["="*100,"HARD ZERO-SHOT SE — FinGPT","="*100,f"Timestamp:{time.strftime('%Y-%m-%d %H:%M:%S')}","",
                 f"Step1 (YES/NO): Acc={s1a*100:.2f}%  Correct={s1c}/{n}","",
                 f"Step2 (Codes): ExactMatch={em*100:.2f}%  MicroP={mp*100:.2f}%  MicroR={mr*100:.2f}%  MicroF1={mf*100:.2f}%","",
                 f"Tokens:{ttok:,}","","Error Distribution:"]
        for et,cnt in sorted(err_dist.items()): summary.append(f"  {et}: {cnt}")
        summary.append("="*100)
        with open(metrics_file,"w",encoding="utf-8") as f: f.write("\n".join(summary))
        logging.info("\n"+"\n".join(summary))
    print(f"Outputs: {csv_file} | {metrics_file} | {log_file}")

if __name__ == "__main__":
    main()
