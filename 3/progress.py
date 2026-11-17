# Check how far experiment.py has come while it runs.
# uv run python progress.py
#
# experiment.py only saves results.csv after each model, so instead this
# counts the answered requests in Ollama's server log since the run started.

import subprocess
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import pandas as pd
import yaml

LOG = Path.home() / ".ollama/logs/server.log"


def run_start():
    # Start time of the running experiment.py process (None if not running)
    pids = subprocess.run(["pgrep", "-f", "python.*experiment.py"],
                          capture_output=True, text=True, check=False).stdout.split()
    if not pids:
        return None
    started = subprocess.run(["ps", "-o", "lstart=", "-p", pids[-1]],
                             capture_output=True, text=True, check=False).stdout.strip()
    return datetime.strptime(started, "%a %b %d %H:%M:%S %Y")


def request_times(since):
    # Times of all answered /api/chat requests in the log after `since`
    times = []
    for line in LOG.read_text(errors="ignore").splitlines():
        if '"/api/chat"' in line and "| 200 |" in line:
            stamp = line.split("|")[0].replace("[GIN]", "").strip()
            t = datetime.strptime(stamp, "%Y/%m/%d - %H:%M:%S")
            if t >= since:
                times.append(t)
    return times


def main():
    start = run_start()
    if start is None:
        print("experiment.py is not running.")
        return

    # Total number of prompts, from the same files experiment.py uses
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    per_model = (len(pd.read_csv("data/names.csv")) * len(pd.read_csv("data/cvs.csv"))
                 * config["repetitions"])
    total = per_model * len(config["models"])

    times = request_times(start)
    done = len(times)
    now = datetime.now()
    print(f"Started {start:%H:%M}, running for {str(now - start).split('.')[0]}")
    print(f"Overall: {done} / {total} prompts ({done / total:.0%})")

    # Prompts are sent model by model, so the count tells which model is busy
    for i, model in enumerate(config["models"]):
        n = min(max(done - i * per_model, 0), per_model)
        print(f"  {model}: {n} / {per_model}")

    # Speed over the last 100 answers, and estimated finish time
    if done >= 2:
        recent = times[-100:]
        sec_per_prompt = (recent[-1] - recent[0]).total_seconds() / (len(recent) - 1)
        left = (total - done) * sec_per_prompt
        finish = now + pd.Timedelta(seconds=left)
        print(f"Speed: {sec_per_prompt:.1f} s per prompt (last {len(recent)} answers)")
        print(f"Estimated finish: {finish:%H:%M} ({left / 3600:.1f} hours left)")

    # Pauses longer than a minute usually mean the Mac went to sleep
    pauses = [(a, b) for a, b in pairwise(times) if (b - a).total_seconds() > 60]
    if pauses:
        lost = sum((b - a).total_seconds() for a, b in pauses) / 60
        print(f"Pauses over 1 min: {len(pauses)} (about {lost:.0f} min lost), "
              f"last one {pauses[-1][0]:%H:%M}-{pauses[-1][1]:%H:%M}")
    if times and (now - times[-1]).total_seconds() > 60:
        print(f"Warning: no answer since {times[-1]:%H:%M}. Is the Mac asleep or Ollama stuck?")


if __name__ == "__main__":
    main()
