# PACE finding validation

Day 1–Day 3 implementation for `PACE_SELECT_STUDENT_EXECUTION.md`.

Install dependencies and reproduce everything from the repository root:

```bash
python -m pip install -r code/finding_validation/requirements.txt
python code/finding_validation/run_all.py
```

`run_all.py` checks scorer coverage, performs offline rescoring only where
needed, rebuilds every CSV, creates the text report, and runs the tests.

The four previously inconsistent MATH-500 runs were rescored offline with the
repository's existing `code/QNN/scripts/rescore_math_benchmarks.py` tool. The
generated `summary.rescored_v2.csv` files are preferred automatically. This
rescoring reads saved completions and does not rerun inference on a phone.

Code lives in `code/finding_validation/`. Generated inventory and experiment
artifacts are written to `result/finding_validation/`; source CSV files under
`results/` are never modified.

The experiment scope includes:

- F1 compares deadline-aware mode selection with the calibration-fastest mode
  within each runtime.
- F3 compares task-constrained deployment selection with decode tokens/s,
  J/token, and request-energy baselines across reference modes on both tasks.
- F2 compares adaptive, fixed-512, and maximum measured budgets wherever at
  least two real budgets are available.
- Joint selection compares the full runtime/mode/budget candidate set with
  fixed-budget, fixed-reference-mode, and request-energy-only designs.

The inventory covers both GSM8K and MATH-500 so scorer and measurement gaps are
visible before Day 2.

