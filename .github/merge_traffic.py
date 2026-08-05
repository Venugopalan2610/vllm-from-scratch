"""Merge today's 14-day traffic window into a permanent history file.

GitHub only keeps 14 days of traffic data. This runs daily, merges the new
window into .traffic/history.json keyed by date, and never drops old entries --
so the history grows past GitHub's window instead of rolling off it.
"""
import json
import os
import pathlib

OUT = pathlib.Path(".traffic")
OUT.mkdir(exist_ok=True)
hist_path = OUT / "history.json"
hist = json.loads(hist_path.read_text()) if hist_path.exists() else {}

for kind in ("clones", "views"):
    payload = json.load(open(f"/tmp/{kind}.json"))
    bucket = hist.setdefault(kind, {})
    for day in payload.get(kind, []):
        # key by date; later snapshots of the same day overwrite with the
        # more complete count
        bucket[day["timestamp"][:10]] = {
            "count": day["count"],
            "uniques": day["uniques"],
        }

summary = {}
for kind in ("clones", "views"):
    days = hist.get(kind, {})
    summary[kind] = {
        "days_recorded": len(days),
        "total": sum(d["count"] for d in days.values()),
        "total_uniques": sum(d["uniques"] for d in days.values()),
    }
hist["summary"] = summary
hist_path.write_text(json.dumps(hist, indent=2, sort_keys=True))

print(json.dumps(summary, indent=2))
