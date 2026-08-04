"""vc - the stage runner.

    vc list              show the ladder and where you are
    vc info              environment + roofline facts for your GPU
    vc test [stage]      run the current stage's tests (or a named one)
    vc pass              mark current stage complete, advance
    vc bench             run the benchmark for the current stage
    vc lore [stage]      print the insight you're supposed to walk away with
    vc math [B] [batch]  read-vs-compute timings for a B-billion-param model,
                         every division shown. e.g. `vc math 7 128`
    vc cliff             measure the L2-vs-VRAM bandwidth cliff (LORE Appendix A)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROGRESS = ROOT / ".progress.json"
PY = ROOT / ".venv" / "bin" / "python"


def load_stages():
    import yaml

    data = yaml.safe_load((ROOT / "stages.yaml").read_text())
    flat = []
    for arc in data["arcs"]:
        for s in arc["stages"]:
            s["arc"] = arc["name"]
            s["arc_id"] = arc["id"]
            flat.append(s)
    return data, flat


def progress():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {"completed": []}


def save_progress(p):
    PROGRESS.write_text(json.dumps(p, indent=2))


def current_stage(flat, p):
    for s in flat:
        if s["id"] not in p["completed"]:
            return s
    return None


def resolve(flat, p, stage_id):
    """Find the stage the user meant. Accepts '1', '01', '01-forward-pass',
    'forward', 'kv-cache'. Returns (stage, error_message)."""
    if stage_id is None:
        s = current_stage(flat, p)
        return s, None if s else "All stages complete."

    q = stage_id.strip().lower()
    for pred in (
        lambda s: s["id"] == q,                              # exact
        lambda s: s["id"].split("-")[0] == q.zfill(2),        # 1 -> 01
        lambda s: s["id"].startswith(q),                      # prefix
        lambda s: q in s["id"] or q in s["name"].lower(),      # substring
    ):
        hits = [s for s in flat if pred(s)]
        if len(hits) == 1:
            return hits[0], None
        if len(hits) > 1:
            ids = ", ".join(h["id"] for h in hits)
            return None, f"{C['y']}'{stage_id}' is ambiguous:{C['x']} {ids}"

    return None, (
        f"{C['y']}No stage matching '{stage_id}'.{C['x']}\n"
        f"{C['dim']}Try a number (1, 7), an id (07-paged-attention), or a word "
        f"(prefix). `vc list` shows them all.{C['x']}"
    )


C = {
    "dim": "\033[2m", "b": "\033[1m", "g": "\033[32m", "y": "\033[33m",
    "c": "\033[36m", "r": "\033[31m", "x": "\033[0m",
}


def cmd_list(flat, p):
    _, cur = None, current_stage(flat, p)
    arc = None
    for s in flat:
        if s["arc"] != arc:
            arc = s["arc"]
            print(f"\n{C['b']}{C['c']}{s['arc_id']}  {arc}{C['x']}")
        done = s["id"] in p["completed"]
        is_cur = cur and s["id"] == cur["id"]
        mark = f"{C['g']}[x]{C['x']}" if done else (f"{C['y']}[>]{C['x']}" if is_cur else "[ ]")
        name = f"{C['b']}{s['name']}{C['x']}" if is_cur else s["name"]
        stars = "*" * s["difficulty"]
        print(f"  {mark} {s['id']:<26} {name}  {C['dim']}{stars}{C['x']}")
    n = len(p["completed"])
    print(f"\n{C['dim']}{n}/{len(flat)} stages complete{C['x']}\n")


def cmd_lore(flat, p, stage_id=None):
    s, err = resolve(flat, p, stage_id)
    if not s:
        print(err)
        return 1
    print(f"\n{C['b']}{C['c']}{s['id']}  {s['name']}{C['x']}  {C['dim']}({'*' * s['difficulty']}){C['x']}")
    print(f"\n{C['b']}The insight{C['x']}\n  " + s["insight"].strip().replace("\n", "\n  "))
    print(f"\n{C['b']}Deliver{C['x']}\n  {s['deliver']}")
    print(f"\n{C['b']}Measure{C['x']}\n  {s['measure']}\n")


def cmd_test(flat, p, stage_id=None):
    s, err = resolve(flat, p, stage_id)
    if not s:
        print(err)
        return 1
    tdir = ROOT / "tests" / f"stage_{s['id'].replace('-', '_')}"
    if not tdir.exists():
        print(f"{C['y']}No tests authored yet for {s['id']}.{C['x']}")
        print(f"{C['dim']}Expected at: {tdir.relative_to(ROOT)}{C['x']}")
        print(f"{C['dim']}Run `vc lore` for the spec, then write the test first.{C['x']}")
        return 1
    print(f"{C['b']}{C['c']}== {s['id']}  {s['name']} =={C['x']}\n")
    r = subprocess.run(
        [str(PY), "-m", "pytest", str(tdir), "-q", "--timeout=600", "-x"],
        cwd=ROOT,
    )
    if r.returncode == 0:
        print(f"\n{C['g']}{C['b']}PASS{C['x']}  -- `vc pass` to advance to the next stage.")
    return r.returncode


def cmd_pass(flat, p):
    s = current_stage(flat, p)
    if not s:
        print("All stages complete.")
        return
    p["completed"].append(s["id"])
    save_progress(p)
    print(f"{C['g']}Completed {s['id']} - {s['name']}{C['x']}")
    nxt = current_stage(flat, p)
    if nxt:
        print(f"\nNext up:")
        cmd_lore(flat, p)
    else:
        print(f"\n{C['b']}You built vLLM.{C['x']}")


def cmd_bench(flat, p):
    s = current_stage(flat, p)
    script = ROOT / "bench" / f"{s['id']}.py"
    if not script.exists():
        print(f"{C['y']}No benchmark for {s['id']} yet.{C['x']}  Measure: {s['measure']}")
        return 1
    return subprocess.run([str(PY), str(script)], cwd=ROOT).returncode


def cmd_info():
    subprocess.run([str(PY), str(ROOT / "runner" / "envinfo.py")], cwd=ROOT)


def main():
    args = sys.argv[1:]
    cmd = args[0] if args else "list"
    arg = args[1] if len(args) > 1 else None
    if cmd == "info":
        return cmd_info()
    if cmd == "cliff":
        return subprocess.run(
            [str(PY), str(ROOT / "runner" / "cliff.py")], cwd=ROOT
        ).returncode
    if cmd == "math":
        return subprocess.run(
            [str(PY), str(ROOT / "runner" / "timings.py")] + args[1:], cwd=ROOT
        ).returncode
    data, flat = load_stages()
    p = progress()
    return {
        "list": lambda: cmd_list(flat, p),
        "ls": lambda: cmd_list(flat, p),
        "lore": lambda: cmd_lore(flat, p, arg),
        "test": lambda: sys.exit(cmd_test(flat, p, arg) or 0),
        "pass": lambda: cmd_pass(flat, p),
        "bench": lambda: cmd_bench(flat, p),
    }.get(cmd, lambda: print(__doc__))()


if __name__ == "__main__":
    main()
