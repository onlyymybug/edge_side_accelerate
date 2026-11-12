"""Run the supplied smoke unchanged and preserve raw output, timing, and SDK JSON."""
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = "/data/local/tmp/qwen3_vl_geniex_sm8750"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["text", "image", "image_info"])
    parser.add_argument("--image")
    parser.add_argument("--log", choices=["none", "error", "warn", "info", "debug", "trace"])
    args = parser.parse_args()
    parent = Path((ROOT / "CURRENT_RUN.txt").read_text().strip())
    out = parent / args.phase
    out.mkdir(exist_ok=False)
    if args.phase.startswith("image") and not args.image:
        parser.error("image phase requires a remote image path")
    command = f"cd {REMOTE} && chmod +x bin/geniex-bench && sh run_smoke.sh"
    if args.image:
        command += " --image " + shlex.quote(args.image)
    if args.log:
        command += " --log " + args.log
    argv = ["adb", "-s", "189f0fad", "shell", command]
    (out / "command.json").write_text(json.dumps({"argv": argv, "remote": REMOTE}, indent=2) + "\n")
    start = time.monotonic()
    first_answer = None
    generated_lines = []
    with (out / "phone_raw.log").open("w") as raw, (out / "line_times.jsonl").open("w") as timed:
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in process.stdout:
            elapsed = time.monotonic() - start
            raw.write(line)
            raw.flush()
            timed.write(json.dumps({"wall_seconds": elapsed, "line": line.rstrip("\n")}, ensure_ascii=False) + "\n")
            timed.flush()
            if "[gen ]" in line:
                generated_lines.append(line.rstrip())
                if first_answer is None:
                    first_answer = elapsed
            if any(marker in line for marker in ("[gen ]", "[ok  ]", "[warn]", "ERROR", "Error", "error", "[info] stop_reason")):
                print(f"{elapsed:.3f}s {line.rstrip()}", flush=True)
        code = process.wait()
    pull = subprocess.run(["adb", "-s", "189f0fad", "pull", REMOTE + "/results/smoke.json", str(out / "smoke.json")],
                          capture_output=True, text=True)
    (out / "pull.log").write_text(pull.stdout + pull.stderr)
    result = {"phase": args.phase, "process_exit_code": code,
              "process_wall_seconds": time.monotonic() - start,
              "first_complete_answer_print_wall_seconds": first_answer,
              "generated_lines": generated_lines, "sdk_json_pulled": pull.returncode == 0,
              "note": "First printed answer is full-answer completion, not SDK TTFT; process wall includes hash verification and startup."}
    (out / "host_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
