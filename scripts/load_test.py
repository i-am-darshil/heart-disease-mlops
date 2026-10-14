"""Send realistic traffic to the API so the monitoring dashboard has something to show.

    python3 scripts/load_test.py                    normal traffic for 2 minutes
    python3 scripts/load_test.py --scenario drift   older, sicker patients (drift panels move)
    python3 scripts/load_test.py --scenario errors  ~30% invalid requests (error alert fires)
    python3 scripts/load_test.py --url http://<vm-ip> --rps 20 --duration 300

Patients are sampled from the real (processed) UCI data, with a little noise, and some
fields are dropped to exercise imputation. Standard library only, so it runs anywhere
Python 3 does (no type hints either, so even EL8's system python3.6 runs it).
"""

import argparse
import csv
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "heart.csv"
NUMERIC = {"age": (1, 120), "trestbps": (50, 250), "chol": (50, 700), "thalach": (50, 250),
           "oldpeak": (0, 10)}  # fmt: skip
CODED = ["cp", "restecg", "slope", "ca", "thal", "sex", "fbs", "exang"]


def load_patients():
    with DATA.open() as f:
        rows = list(csv.DictReader(f))
    patients = []
    for row in rows:
        row.pop("target", None)
        patients.append({k: (None if v == "" else float(v)) for k, v in row.items()})
    return patients


def jitter(patient, rng):
    p = dict(patient)
    for name, (low, high) in NUMERIC.items():
        if p[name] is not None:
            p[name] = round(min(high, max(low, p[name] * rng.uniform(0.95, 1.05))), 1)
    for name in CODED:
        if p[name] is not None:
            p[name] = int(p[name])
    if rng.random() < 0.05:  # a few incomplete records: the API imputes them
        p.pop(rng.choice(["chol", "ca", "thal"]))
    return p


def drifted(patient, rng):
    """A population shift: older patients, higher cholesterol, lower max heart rate."""
    p = jitter(patient, rng)
    p["age"] = min(100, (p["age"] or 55) + 15)
    if p.get("chol") is not None:
        p["chol"] = min(700, p["chol"] + 80)
    p["thalach"] = max(60, (p["thalach"] or 150) - 30)
    return p


def invalid(rng):
    return rng.choice([{"age": 500}, {"cp": 9}, {"sex": "male"}, {"cholesterol": 200}])


def build_request(base, scenario, patients, rng):
    if scenario == "errors" and rng.random() < 0.3:
        return base + "/predict", invalid(rng)
    make = drifted if scenario == "drift" else jitter
    if rng.random() < 0.1:
        batch = [make(rng.choice(patients), rng) for _ in range(rng.randint(2, 10))]
        return base + "/predict/batch", {"patients": batch}
    return base + "/predict", make(rng.choice(patients), rng)


def send(url, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
            status = r.status
    except urllib.error.HTTPError as err:
        status = err.code
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        status = 0  # couldn't connect / timed out
    return status, time.perf_counter() - start


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1", help="API base URL")
    parser.add_argument("--rps", type=float, default=10, help="requests per second")
    parser.add_argument("--duration", type=float, default=120, help="seconds to run")
    parser.add_argument("--scenario", choices=["normal", "drift", "errors"], default="normal")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    try:
        urllib.request.urlopen(base + "/ready", timeout=5).read()
    except (urllib.error.URLError, OSError) as err:
        print(f"API not ready at {base}/ready: {err}", file=sys.stderr)
        return 1

    patients = load_patients()
    rng = random.Random(args.seed)
    statuses = Counter()
    latencies = []
    lock = threading.Lock()
    deadline = time.monotonic() + args.duration
    stopped = threading.Event()
    interval = args.workers / args.rps  # each worker sends one request per interval

    def worker(seed):
        wrng = random.Random(seed)
        next_at = time.monotonic() + wrng.uniform(0, interval)  # stagger workers
        while next_at < deadline and not stopped.is_set():
            time.sleep(max(0.0, next_at - time.monotonic()))
            url, body = build_request(base, args.scenario, patients, wrng)
            status, seconds = send(url, body)
            with lock:
                statuses[status] += 1
                latencies.append(seconds)
            next_at += interval

    print(f"{args.scenario} traffic to {base}: {args.rps:g} req/s for {args.duration:g} s "
          "(Ctrl+C to stop early)")  # fmt: skip
    threads = [threading.Thread(target=worker, args=(rng.random(),), daemon=True)
               for _ in range(args.workers)]  # fmt: skip
    for t in threads:
        t.start()
    started = time.monotonic()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(5)
            with lock:
                done = sum(statuses.values())
            print(f"  {time.monotonic() - started:5.0f} s  {done} requests  {dict(statuses)}",
                  flush=True)  # fmt: skip
    except KeyboardInterrupt:
        print("  stopped early")
        stopped.set()

    total = sum(statuses.values())
    if not total:
        print("No requests sent.")
        return 1
    latencies.sort()
    p50, p95 = latencies[len(latencies) // 2], latencies[int(len(latencies) * 0.95)]
    ok = sum(n for s, n in statuses.items() if 200 <= s < 300)
    print(f"Done: {total} requests, {ok / total:.0%} OK, p50 {p50 * 1000:.0f} ms, "
          f"p95 {p95 * 1000:.0f} ms, status codes {dict(sorted(statuses.items()))}")  # fmt: skip
    # Exit non-zero only if the API was unreachable, not for the 422s the error scenario asks for
    return 1 if statuses.get(0) else 0


if __name__ == "__main__":
    sys.exit(main())
