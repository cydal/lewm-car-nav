"""Append-only JSONL store for probe results, so cross-run comparisons
(e.g. "does heading erode faster at 900 or 8100 episodes?") don't require
re-running every checkpoint again or scraping stdout logs.

One line per (run_name, epoch). Append-only -- re-running a probe for an
existing (run_name, epoch) adds a new line rather than overwriting, so the
file is also a history of probe-code changes, not just training runs.
`load()` dedups to the latest line per key for callers that want "current
belief" rather than full history.
"""
import json
import os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_results.jsonl")


def append(record):
    """record must include at least 'run_name' and 'epoch'."""
    assert "run_name" in record and "epoch" in record
    with open(PATH, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load(dedup_latest=True):
    """Returns list of records. If dedup_latest, keeps only the last-written
    line per (run_name, epoch) -- later appends win."""
    if not os.path.exists(PATH):
        return []
    records = {}
    order = []
    with open(PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["run_name"], rec["epoch"])
            if dedup_latest:
                records[key] = rec
            else:
                order.append(rec)
    if dedup_latest:
        return list(records.values())
    return order
