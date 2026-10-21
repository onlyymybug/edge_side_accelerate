# Section 9 optional calibration experiment

Run from the repository root:

```bash
/home/lyyyy/miniconda3/envs/pace/bin/python \
  code/finding_validation/optional_calibration/run.py
```

The script reads the frozen Day 1 records and writes an independent Section 9
replay to `result/finding_validation/optional_calibration/`. It never runs a
phone. The adopted operational definition treats byte-identical saved
completions as a shared generation trajectory. Original generated token IDs
were not recorded and are not claimed separately. The replay also reports the
historical inference-window time and recorded energy consumed by each
calibration allocation; model loading, mode switching, and cooling overheads
were not logged and are explicitly excluded.
