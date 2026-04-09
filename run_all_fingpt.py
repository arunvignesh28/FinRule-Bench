#!/usr/bin/env python3
"""
run_all_fingpt.py
Run from the repo root:
    python run_all_fingpt.py
    python run_all_fingpt.py --gpu 2
    python run_all_fingpt.py --gpu 2 --patch        # apply patch_all_fingpt.py first
    python run_all_fingpt.py --skip-to hard/BS      # resume from a specific script

Runs all 36 FinGPT evaluation scripts sequentially from their own directories,
then calls aggregate_results_fingpt.py. Tracks pass/fail, writes a run log.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent

SCRIPTS = [
    # ── EASY ──────────────────────────────────────────────────────────────
    "task/easy/BS/FinGPT/zero_shot_easy_BS_fingpt.py",
    "task/easy/BS/FinGPT/fewshot_easy_BS_fingpt.py",
    "task/easy/BS/FinGPT/fewshot_counterfactual_easy_BS_fingpt.py",
    "task/easy/CF/FinGPT/zero_shot_easy_CF_fingpt.py",
    "task/easy/CF/FinGPT/fewshot_easy_CF_fingpt.py",
    "task/easy/CF/FinGPT/fewshot_counterfactual_easy_CF_fingpt.py",
    "task/easy/SE/FinGPT/zero_shot_easy_SE_fingpt.py",
    "task/easy/SE/FinGPT/fewshot_easy_SE_fingpt.py",
    "task/easy/SE/FinGPT/fewshot_counterfactual_easy_SE_fingpt.py",
    "task/easy/SI/FinGPT/zero_shot_easy_SI_fingpt.py",
    "task/easy/SI/FinGPT/fewshot_easy_SI_fingpt.py",
    "task/easy/SI/FinGPT/fewshot_counterfactual_easy_SI_fingpt.py",
    # ── MEDIUM ────────────────────────────────────────────────────────────
    "task/medium/BS/FinGPT/zero_shot_medium_BS_fingpt.py",
    "task/medium/BS/FinGPT/few_shot_medium_BS_fingpt.py",
    "task/medium/BS/FinGPT/few_shot_counterfactual_medium_BS_fingpt.py",
    "task/medium/CF/FinGPT/zero_shot_medium_CF_fingpt.py",
    "task/medium/CF/FinGPT/few_shot_medium_CF_fingpt.py",
    "task/medium/CF/FinGPT/few_shot_counterfactual_medium_CF_fingpt.py",
    "task/medium/SE/FinGPT/zero_shot_medium_SE_fingpt.py",
    "task/medium/SE/FinGPT/few_shot_medium_SE_fingpt.py",
    "task/medium/SE/FinGPT/few_shot_counterfactual_medium_SE_fingpt.py",
    "task/medium/SI/FinGPT/zero_shot_medium_SI_fingpt.py",
    "task/medium/SI/FinGPT/few_shot_medium_SI_fingpt.py",
    "task/medium/SI/FinGPT/few_shot_counterfactual_medium_SI_fingpt.py",
    # ── HARD ──────────────────────────────────────────────────────────────
    "task/hard/BS/FinGPT/hard_zero_shot_BS_fingpt.py",
    "task/hard/BS/FinGPT/hard_few_shot_BS_fingpt.py",
    "task/hard/BS/FinGPT/hard_counterfactual_few_shot_BS_fingpt.py",
    "task/hard/CF/FinGPT/hard_zero_shot_CF_fingpt.py",
    "task/hard/CF/FinGPT/hard_few_shot_CF_fingpt.py",
    "task/hard/CF/FinGPT/hard_counterfactual_few_shot_CF_fingpt.py",
    "task/hard/SE/FinGPT/hard_zero_shot_SE_fingpt.py",
    "task/hard/SE/FinGPT/hard_few_shot_SE_fingpt.py",
    "task/hard/SE/FinGPT/hard_counterfactual_few_shot_SE_fingpt.py",
    "task/hard/SI/FinGPT/hard_zero_shot_SI_fingpt.py",
    "task/hard/SI/FinGPT/hard_few_shot_SI_fingpt.py",
    "task/hard/SI/FinGPT/hard_counterfactual_few_shot_SI_fingpt.py",
]


def fmt_duration(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_script(script_path, env, python):
    """Run one script from its own directory. Returns (exit_code, duration_sec)."""
    script_path = REPO / script_path
    t0 = time.time()
    result = subprocess.run(
        [python, script_path.name],
        cwd=str(script_path.parent),
        env=env,
    )
    return result.returncode, time.time() - t0


def main():
    parser = argparse.ArgumentParser(description="Run all 36 FinGPT eval scripts")
    parser.add_argument("--gpu", default=os.environ.get("FINGPT_GPU", "0"),
                        help="CUDA_VISIBLE_DEVICES value (default: 0)")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter to use")
    parser.add_argument("--patch", action="store_true",
                        help="Run patch_all_fingpt.py before evaluations")
    parser.add_argument("--skip-to", metavar="SUBSTR", default=None,
                        help="Skip scripts until one whose path contains SUBSTR")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="Skip aggregate_results_fingpt.py at the end")
    args = parser.parse_args()

    log_file = REPO / "fingpt_run_all.log"
    total = len(SCRIPTS)

    # Build environment — inherit everything, override CUDA device
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["REPO_ROOT"] = str(REPO)

    sep = "=" * 72
    banner = [
        sep,
        " FinGPT Full Evaluation Run",
        f" Repo  : {REPO}",
        f" GPU   : {args.gpu}  (CUDA_VISIBLE_DEVICES={args.gpu})",
        f" Python: {args.python}",
        f" Start : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        sep,
    ]
    header = "\n".join(banner)
    print(header)
    with open(log_file, "w") as lf:
        lf.write(header + "\n\n")

    # ── Optional: patch all scripts first ────────────────────────────────
    if args.patch:
        patch_script = REPO / "patch_all_fingpt.py"
        if patch_script.exists():
            print("\n[PATCH] Applying patch_all_fingpt.py ...")
            r = subprocess.run([args.python, str(patch_script)], cwd=str(REPO), env=env)
            if r.returncode != 0:
                print("  WARNING: patch script exited non-zero — continuing anyway")
        else:
            print("  WARNING: patch_all_fingpt.py not found — skipping patch step")

    # ── Run each script ───────────────────────────────────────────────────
    passed, failed, skipped = 0, 0, 0
    skipping = bool(args.skip_to)
    total_start = time.time()

    for idx, rel_path in enumerate(SCRIPTS, 1):
        full_path = REPO / rel_path

        # --skip-to support
        if skipping:
            if args.skip_to in rel_path:
                skipping = False
            else:
                print(f"[{idx:2d}/{total}] SKIP (--skip-to): {rel_path}")
                skipped += 1
                continue

        if not full_path.exists():
            msg = f"[{idx:2d}/{total}] NOT FOUND: {rel_path}"
            print(msg)
            with open(log_file, "a") as lf:
                lf.write(msg + "\n")
            skipped += 1
            continue

        print(f"\n{'-'*72}")
        print(f"[{idx:2d}/{total}] {time.strftime('%H:%M:%S')}  {rel_path}")
        print(f"{'-'*72}")

        code, dur = run_script(rel_path, env, args.python)

        status = "OK" if code == 0 else f"FAILED (exit {code})"
        result_line = f"  --> {status}  ({fmt_duration(dur)})"
        print(result_line)

        with open(log_file, "a") as lf:
            lf.write(f"[{idx:2d}/{total}] {rel_path}\n{result_line}\n\n")

        if code == 0:
            passed += 1
        else:
            failed += 1

    # ── Aggregate ─────────────────────────────────────────────────────────
    agg_status = 0
    if not args.no_aggregate:
        agg_script = REPO / "aggregate_results_fingpt.py"
        print(f"\n{sep}")
        print("[AGGREGATE] Running aggregate_results_fingpt.py ...")
        print(sep)
        agg_env = env.copy()
        agg_env.pop("CUDA_VISIBLE_DEVICES", None)  # no GPU needed for aggregation
        r = subprocess.run([args.python, str(agg_script)], cwd=str(REPO), env=agg_env)
        agg_status = r.returncode
        agg_line = "Aggregate: OK" if agg_status == 0 else f"Aggregate: FAILED (exit {agg_status})"
        print(f"  --> {agg_line}")

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    summary = [
        "",
        sep,
        f" DONE: {time.strftime('%Y-%m-%d %H:%M:%S')}  (wall time: {fmt_duration(elapsed)})",
        f" Passed : {passed} / {total}",
        f" Failed : {failed} / {total}",
        f" Skipped: {skipped} / {total}",
    ]
    if not args.no_aggregate:
        summary.append(f" {agg_line}")
    summary.append(sep)
    final = "\n".join(summary)
    print(final)
    with open(log_file, "a") as lf:
        lf.write(final + "\n")

    print(f"\nLog written to: {log_file}")
    return 1 if (failed > 0 or agg_status != 0) else 0


if __name__ == "__main__":
    sys.exit(main())
