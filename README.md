# FinRule-Bench

**FinRule-Bench** is a benchmark for evaluating **rule-based joint reasoning** over real-world financial tables under **explicit accounting principles**.

Unlike QA or anomaly-detection benchmarks, FinRule-Bench evaluates whether models can *verify, identify, and localize accounting rule violations* with diagnostic completeness.

## What This Benchmark Tests

FinRule-Bench targets **auditing-style reasoning** over structured financial statements:

- Exhaustive violation detection
- Violation detection among accounting principles on the financial statements
- Record-level localization of multiple simultaneous violations

## Financial Statements Covered
- Balance Sheet (BS)
- Income Statement (SI)
- Cash Flow Statement (CF)
- Statement of Equity (SE)

All base tables are **ground-truth financial statements** derived from real corporate filings.

## Tasks

1. **Rule Verification**  
   Binary compliance check for a given accounting rule.

2. **Rule Identification**  
   Identify the *single violated rule* from a provided rule set.

3. **Joint Rule Diagnosis**  
   Detect and localize **multiple violated rules** at record level.

## Key Design Features

- Human-curated accounting rules with **deterministic validators**
- Controlled, minimal **rule-aware error injection**
- Exact-match, validator-based evaluation
- **causal–counterfactual prompting** strategy for diagnostic analysis

## What This Is *Not*

- Not a question-answering benchmark  
- Not an anomaly detection on noisy data  
- Not a deployment-ready auditing system  

This benchmark is strictly for **research and evaluation**.


