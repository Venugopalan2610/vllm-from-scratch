"""Merge one 14-day traffic window into a permanent history file.

    merge_traffic.py <history.json> <clones.json> <views.json>

GitHub only exposes the last 14 days of traffic. This runs daily, folds the
new window into the history keyed by date, and never drops old entries -- so
the record grows past GitHub's window instead of rolling off it.
"""

import json
import pathlib
import sys

hist_path = pathlib.Path(sys.argv[1])
raw = hist_path.read_text().strip() if hist_path.exists() else ""
hist = json.loads(raw) if raw else {}

for kind, src in (("clones", sys.argv[2]), ("views", sys.argv[3])):
    payload = json.load(open(src))
    bucket = hist.setdefault(kind, {})
    for day in payload.get(kind, []):
        # Key by date. A later snapshot of the same day overwrites the earlier
        # one, because GitHub's count for "today" is still climbing when we
        # first see it.
        bucket[day["timestamp"][:10]] = {
            "count": day["count"],
            "uniques": day["uniques"],
        }

hist["summary"] = {
    kind: {
        "days_recorded": len(hist.get(kind, {})),
        "total": sum(d["count"] for d in hist.get(kind, {}).values()),
        "total_uniques": sum(d["uniques"] for d in hist.get(kind, {}).values()),
    }
    for kind in ("clones", "views")
}

hist_path.write_text(json.dumps(hist, indent=2, sort_keys=True) + "\n")
print(json.dumps(hist["summary"], indent=2))
