#!/usr/bin/env python3
"""Run, resume, smoke-test and aggregate the Bayesmark jobs in jobs.txt."""

import argparse
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent
JOBS_FILE = ROOT / "jobs.txt"
BASELINE = ROOT / "baseline.json"
PID_FILE = ROOT / ".runner.pid"

N_REPEATS = int(os.getenv("BAYESMARK_REPEATS", 20))
N_WORKERS = int(os.getenv("BAYESMARK_WORKERS", 8))
RESULTS_DIR = Path(os.getenv("BAYESMARK_RESULTS_DIR", "results"))


def path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def read_jobs(limit=None):
    jobs = [x.strip() for x in JOBS_FILE.read_text().splitlines()
            if x.strip() and not x.lstrip().startswith("#")]
    return jobs[:limit] if limit else jobs


def make_command(line, results_dir, repeat):
    # jobs.txt is the source of truth; only output location/repeat are replaced.
    line = re.sub(r"(?:^|\s)-(?:dir|db-root)\s+\S+", "", line)
    line = re.sub(r"(?:^|\s)-(?:b|db)\s+\S+", "", line)
    return f"{line.strip()} -dir {shlex.quote(str(path(results_dir)))} -b part{repeat}"


def marker(state_dir, repeat, job_no):
    return path(state_dir) / f"part{repeat:02d}_job{job_no:04d}.done"


def kill_tree(pid):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for child in children:
        child.terminate()
    _, alive = psutil.wait_procs(children, timeout=5)
    for child in alive:
        child.kill()


def stop_handler(signum, _frame):
    print("\nStopping active Bayesmark jobs...", file=sys.stderr)
    kill_tree(os.getpid())
    PID_FILE.unlink(missing_ok=True)
    raise SystemExit(128 + signum)


def run_one(command, log_file):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a") as log:
        log.write(f"\n=== command ===\n{command}\n")
        return subprocess.run(command, shell=True, cwd=ROOT,
                              stdout=log, stderr=subprocess.STDOUT).returncode


def run_benchmark(jobs, repeats, results_dir, logs_dir, state_dir, workers,
                  resume=False, start_repeat=1, dry_run=False):
    results_dir, logs_dir, state_dir = map(path, (results_dir, logs_dir, state_dir))

    dirty = any(results_dir.glob("part*")) or (state_dir.exists() and any(state_dir.iterdir()))
    if dirty and not resume:
        raise SystemExit(
            f"{results_dir.name}/ or {state_dir.name}/ already contains run data. "
            "Use resume, or remove/move it before a fresh run."
        )

    if dry_run:
        total = len(jobs) * (repeats - start_repeat + 1)
        print(f"Would run {total} jobs ({len(jobs)} jobs x {repeats - start_repeat + 1} repeats).")
        for repeat in range(start_repeat, min(repeats, start_repeat + 1) + 1):
            for line in jobs[:10]:
                print(make_command(line, results_dir, repeat))
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop_handler)
        

    try:
        for repeat in range(start_repeat, repeats + 1):
            pending = [(i, make_command(line, results_dir, repeat))
                       for i, line in enumerate(jobs, 1)
                       if not marker(state_dir, repeat, i).exists()]
            print(f"[part{repeat}] {len(pending)}/{len(jobs)} jobs pending")

            failed = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_one, command,
                           logs_dir / f"part{repeat:02d}_job{i:04d}.log"): i
                           for i, command in pending}
                for future in as_completed(futures):
                    i = futures[future]
                    rc = future.result()
                    if rc == 0:
                        marker(state_dir, repeat, i).touch()
                        print(f"[part{repeat}] job {i}: ok", flush=True)
                    else:
                        failed.append(i)
                        print(f"[part{repeat}] job {i}: FAILED ({rc})", flush=True)

            if failed:
                raise SystemExit(f"Failed jobs in part{repeat}: {failed}. Inspect logs and resume.")
    finally:
        PID_FILE.unlink(missing_ok=True)


def aggregate(results_dir):
    results_dir = path(results_dir)
    parts = sorted((p for p in results_dir.glob("part*") if p.is_dir()),
                   key=lambda p: int(p.name[4:]))
    if not parts:
        raise SystemExit(f"No part* directories found in {results_dir}")

    for part in parts:
        derived = part / "derived"
        shutil.rmtree(derived, ignore_errors=True)
        subprocess.run(["bayesmark-agg", "-dir", str(results_dir), "-b", part.name],
                       cwd=ROOT, check=True)
        derived.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASELINE, derived / "baseline.json")
        subprocess.run(["bayesmark-anal", "-dir", str(results_dir), "-b", part.name],
                       cwd=ROOT, check=True)


def stop():
    if not PID_FILE.exists():
        print("No active benchmark runner recorded.")
        return
    try:
        os.kill(int(PID_FILE.read_text()), signal.SIGTERM)
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        print("Recorded runner is no longer active.")


def parse_args():
    argv = sys.argv[1:]
    if argv and argv[0].startswith("-"):
        argv = ["run"] + argv

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int, default=N_WORKERS)
        p.add_argument("--start-repeat", type=int, default=1)
        p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("smoke-test")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("aggregate")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    sub.add_parser("stop")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.command in {"run", "resume"}:
        run_benchmark(read_jobs(), N_REPEATS, RESULTS_DIR, "logs", ".state",
                      args.workers, args.command == "resume", args.start_repeat, args.dry_run)
    elif args.command == "smoke-test":
        # Same runner as the full benchmark; only the job/repeat counts and output path differ.
        run_benchmark(read_jobs(2), 2, "smoke_results", "smoke_logs", ".smoke_state",
                      args.workers, dry_run=args.dry_run)
    elif args.command == "aggregate":
        aggregate(args.results_dir)
    else:
        stop()


if __name__ == "__main__":
    main()
