"""vc - the stage runner.

  THE LOOP
    vc               where you are
    vc guide         what to build for the current stage, and why
    vc test          run the current stage's checks
    vc submit        run the checks; if green, bank it and open the next stage

  REFERENCE
    vc list          the whole ladder
    vc lore [stage]  the one-paragraph insight for any stage
    vc info          your GPU's roofline
    vc math [B] [n]  read-vs-compute timings, every division shown
    vc cliff         the L2-vs-VRAM bandwidth cliff
    vc peek [stage]  show the reference solution (from the solutions branch)
    vc peek [n] --apply   write it straight into the stage's file
    vc reset <n>     rewind to stage n (your code is untouched)
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROGRESS = ROOT / ".progress.json"
PY = ROOT / ".venv" / "bin" / "python"

C = {
    "dim": "\033[2m", "b": "\033[1m", "g": "\033[32m", "y": "\033[33m",
    "c": "\033[36m", "r": "\033[31m", "x": "\033[0m", "u": "\033[4m",
}


def load_stages():
    import yaml

    data = yaml.safe_load((ROOT / "stages.yaml").read_text())
    flat = []
    for arc in data["arcs"]:
        for s in arc["stages"]:
            s["arc"], s["arc_id"] = arc["name"], arc["id"]
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
    if stage_id is None:
        s = current_stage(flat, p)
        return s, None if s else "All stages complete."
    q = stage_id.strip().lower()
    for pred in (
        lambda s: s["id"] == q,
        lambda s: s["id"].split("-")[0] == q.zfill(2),
        lambda s: s["id"].startswith(q),
        lambda s: q in s["id"] or q in s["name"].lower(),
    ):
        hits = [s for s in flat if pred(s)]
        if len(hits) == 1:
            return hits[0], None
        if len(hits) > 1:
            return None, (f"{C['y']}'{stage_id}' is ambiguous:{C['x']} "
                          + ", ".join(h["id"] for h in hits))
    return None, (
        f"{C['y']}No stage matching '{stage_id}'.{C['x']}\n"
        f"{C['dim']}Try a number (1, 7), an id (07-paged-attention), or a word. "
        f"`vc list` shows them all.{C['x']}"
    )


def test_dir(s):
    return ROOT / "tests" / f"stage_{s['id'].replace('-', '_')}"


def stage_index(flat, s):
    return next(i for i, x in enumerate(flat) if x["id"] == s["id"]) + 1


def checks_for(s):
    """(name, one-line description) for every check in the stage."""
    d = test_dir(s)
    out = []
    for f in sorted(d.glob("test_*.py")) if d.exists() else []:
        src = f.read_text()
        for m in re.finditer(r"^def (test_\w+)\(", src, re.M):
            dm = re.match(r'[^)]*\):\n\s*"""(.*?)(?:\n|""")', src[m.end():], re.S)
            desc = dm.group(1).strip().rstrip('"').strip() if dm else ""
            out.append((m.group(1), desc))
    return out


def wrap(text, width=72, indent="  "):
    lines, cur = [], ""
    for w in text.split():
        if len(cur) + len(w) + 1 > width:
            lines.append(indent + cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(indent + cur)
    return "\n".join(lines)


def bar(n, total, width=28):
    filled = int(width * n / total)
    return f"{C['g']}[{'#' * filled}{'.' * (width - filled)}]{C['x']}"


# ---------------------------------------------------------------- commands

def cmd_guide(flat, p, stage_id=None):
    s, err = resolve(flat, p, stage_id)
    if not s:
        print(err)
        return 1
    idx, total = stage_index(flat, s), len(flat)
    stars = "*" * s["difficulty"] + "." * (5 - s["difficulty"])
    done = s["id"] in p["completed"]

    print(f"\n{C['b']}{C['c']}{'=' * 74}{C['x']}")
    print(f"{C['b']}{C['c']}  Stage {idx:02d}/{total}   {s['name']}{C['x']}"
          f"   {C['dim']}[{stars}]{C['x']}")
    print(f"{C['dim']}  {s['arc_id']} - {s['arc']}"
          + ("   (already completed)" if done else "") + f"{C['x']}")
    print(f"{C['b']}{C['c']}{'=' * 74}{C['x']}")

    print(f"\n{C['b']}WHY THIS STAGE EXISTS{C['x']}")
    print(wrap(" ".join(s["insight"].split())))

    print(f"\n{C['b']}WHAT YOU'RE BUILDING{C['x']}")
    print(wrap(s["deliver"]))
    print(f"\n  {C['u']}{s['file']}{C['x']}"
          f"   {C['dim']}<- edit this; the full spec is in its docstrings{C['x']}")

    print(f"\n{C['b']}HOW YOU'LL KNOW IT WORKED{C['x']}")
    print(wrap(s["measure"]))

    checks = checks_for(s)
    if checks:
        print(f"\n{C['b']}THE CHECKS{C['x']} {C['dim']}({len(checks)}){C['x']}")
        for name, desc in checks:
            print(f"  {C['dim']}[ ]{C['x']} {name[5:].replace('_', ' ')}")
            if desc:
                d = " ".join(desc.split())
                print(f"      {C['dim']}{d[:63] + '...' if len(d) > 66 else d}{C['x']}")

    print(f"\n{C['b']}NEXT{C['x']}")
    print(f"  {C['c']}./vc test{C['x']}     run the checks, as often as you like")
    print(f"  {C['c']}./vc submit{C['x']}   bank it and open stage {idx + 1:02d}")
    print(f"  {C['dim']}./vc peek     reveal the reference solution{C['x']}\n")
    return 0


def run_checks(s):
    d = test_dir(s)
    if not d.exists():
        return 1, 0, 0, f"No checks authored for {s['id']}."
    r = subprocess.run(
        [str(PY), "-m", "pytest", str(d), "-q", "--timeout=900",
         "--no-header", "-rN"],
        cwd=ROOT, capture_output=True, text=True,
    )
    out = r.stdout + r.stderr
    m = re.search(r"(\d+) passed", out)
    passed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) failed", out)
    failed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) error", out)
    failed += int(m.group(1)) if m else 0
    return r.returncode, passed, failed, out


def show_failures(out, passed, failed, s):
    body = out.split("= FAILURES =")[-1] if "FAILURES" in out else out
    print(body.strip()[:4000])
    print(f"\n{C['r']}{C['b']}  {passed}/{passed + failed} checks passing{C['x']}")
    print(f"\n  {C['dim']}Spec: {s['file']}   |   ./vc guide   |   ./vc peek{C['x']}\n")


def cmd_test(flat, p, stage_id=None):
    s, err = resolve(flat, p, stage_id)
    if not s:
        print(err)
        return 1
    print(f"\n{C['b']}{C['c']}Stage {stage_index(flat, s):02d}  {s['name']}{C['x']}")
    print(f"{C['dim']}running checks...{C['x']}\n")
    rc, passed, failed, out = run_checks(s)
    if rc == 0:
        print(f"{C['g']}{C['b']}  ALL {passed} CHECKS PASSED{C['x']}")
        print(f"\n  {C['c']}./vc submit{C['x']} to bank it and open the next stage.\n")
    else:
        show_failures(out, passed, failed, s)
    return rc


def cmd_submit(flat, p):
    s = current_stage(flat, p)
    if not s:
        print(f"\n{C['b']}All stages complete. You built vLLM.{C['x']}\n")
        return 0
    idx = stage_index(flat, s)
    print(f"\n{C['b']}{C['c']}Submitting stage {idx:02d}  {s['name']}{C['x']}")
    print(f"{C['dim']}running checks...{C['x']}\n")
    rc, passed, failed, out = run_checks(s)

    if rc != 0:
        show_failures(out, passed, failed, s)
        print(f"{C['r']}{C['b']}  NOT SUBMITTED{C['x']}"
              f"  {C['dim']}fix them and run ./vc submit again{C['x']}\n")
        return 1

    print(f"{C['g']}{C['b']}  ALL {passed} CHECKS PASSED{C['x']}")

    # Commit ONLY the learner's work. Never `git add -A` here -- that sweeps up
    # harness edits and anything else lying around, and a later rewind would
    # throw them away along with the stage.
    subprocess.run(["git", "add", "app/"], cwd=ROOT, capture_output=True)
    msg = f"stage {idx:02d} complete: {s['name']}"
    cm = subprocess.run(
        ["git", "-c", "user.email=you@localhost", "-c", "user.name=you",
         "commit", "-m", msg],
        cwd=ROOT, capture_output=True, text=True,
    )
    if cm.returncode == 0:
        print(f"{C['dim']}  committed: {msg}{C['x']}")

    p["completed"].append(s["id"])
    save_progress(p)

    n, total = len(p["completed"]), len(flat)
    print(f"\n  {bar(n, total)}  {n}/{total} stages\n")

    if current_stage(flat, p):
        return cmd_guide(flat, p)
    print(f"{C['b']}{C['g']}  You built vLLM. Every stage green.{C['x']}\n")
    return 0


def cmd_status(flat, p):
    s = current_stage(flat, p)
    n, total = len(p["completed"]), len(flat)
    print(f"\n  {bar(n, total)}  {n}/{total} stages complete")
    if not s:
        print(f"\n  {C['b']}All done. You built vLLM.{C['x']}\n")
        return 0
    print(f"\n  {C['b']}Current: stage {stage_index(flat, s):02d}  {s['name']}{C['x']}")
    print(f"  {C['dim']}{s['file']}{C['x']}")
    print(f"\n  {C['c']}./vc guide{C['x']}   what to build and why")
    print(f"  {C['c']}./vc test{C['x']}    run the checks")
    print(f"  {C['c']}./vc submit{C['x']}  bank it, open the next stage\n")
    return 0


def cmd_list(flat, p):
    cur = current_stage(flat, p)
    arc = None
    for i, s in enumerate(flat, 1):
        if s["arc"] != arc:
            arc = s["arc"]
            print(f"\n{C['b']}{C['c']}{s['arc_id']}  {arc}{C['x']}")
        done = s["id"] in p["completed"]
        is_cur = cur and s["id"] == cur["id"]
        mark = (f"{C['g']}[x]{C['x']}" if done
                else f"{C['y']}[>]{C['x']}" if is_cur else "[ ]")
        name = f"{C['b']}{s['name']}{C['x']}" if is_cur else s["name"]
        print(f"  {mark} {i:02d} {s['id']:<26} {name}  "
              f"{C['dim']}{'*' * s['difficulty']}{C['x']}")
    print(f"\n{C['dim']}{len(p['completed'])}/{len(flat)} complete{C['x']}\n")
    return 0


def cmd_lore(flat, p, stage_id=None):
    s, err = resolve(flat, p, stage_id)
    if not s:
        print(err)
        return 1
    print(f"\n{C['b']}{C['c']}{s['id']}  {s['name']}{C['x']}  "
          f"{C['dim']}({'*' * s['difficulty']}){C['x']}")
    print(f"\n{C['b']}The insight{C['x']}\n" + wrap(" ".join(s["insight"].split())))
    print(f"\n{C['b']}Deliver{C['x']}\n  {s['deliver']}")
    print(f"\n{C['b']}Measure{C['x']}\n  {s['measure']}\n")
    return 0


def solution_text(name):
    """Reference solutions live on the `solutions` branch, not on master.

    Order: a local .solutions/ checkout, then the branch, then the remote
    branch, fetching it once if we have never seen it.
    """
    local = ROOT / ".solutions" / name
    if local.exists():
        return local.read_text()

    def show(ref):
        r = subprocess.run(["git", "show", f"{ref}:.solutions/{name}"],
                           cwd=ROOT, capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None

    for ref in ("solutions", "origin/solutions"):
        t = show(ref)
        if t:
            return t

    print(f"{C['dim']}fetching the solutions branch...{C['x']}")
    subprocess.run(
        ["git", "fetch", "origin", "solutions:refs/remotes/origin/solutions"],
        cwd=ROOT, capture_output=True,
    )
    return show("origin/solutions")


def cmd_peek(flat, p, stage_id=None):
    s, err = resolve(flat, p, stage_id)
    if not s:
        print(err)
        return 1
    name = Path(s["file"]).name
    text = solution_text(name)
    if text is None:
        print(f"{C['y']}Could not reach the solutions branch.{C['x']}")
        print(f"{C['dim']}Try: git fetch origin solutions{C['x']}")
        return 1
    print(f"\n{C['y']}Reference solution for {s['id']}{C['x']}")
    print(f"{C['dim']}To use it:  ./vc peek {stage_index(flat, s)} --apply"
          f"   (or copy it by hand){C['x']}\n")
    print(text)
    return 0


def cmd_peek_apply(flat, p, stage_id=None):
    s, err = resolve(flat, p, stage_id)
    if not s:
        print(err)
        return 1
    text = solution_text(Path(s["file"]).name)
    if text is None:
        print(f"{C['y']}Could not reach the solutions branch.{C['x']}")
        return 1
    (ROOT / s["file"]).write_text(text)
    print(f"{C['y']}Wrote the reference solution into {s['file']}.{C['x']}")
    print(f"{C['dim']}Read it before you submit -- the point was the reading.{C['x']}")
    return 0


def cmd_reset(flat, p, arg):
    if not arg:
        print("usage: vc reset <stage-number>")
        return 1
    s, err = resolve(flat, p, arg)
    if not s:
        print(err)
        return 1
    idx = stage_index(flat, s)
    keep = {x["id"] for x in flat[:idx - 1]}
    p["completed"] = [c for c in p["completed"] if c in keep]
    save_progress(p)
    print(f"{C['y']}Rewound to stage {idx:02d}.{C['x']} "
          f"{C['dim']}Your code in app/ is untouched.{C['x']}")
    return cmd_guide(flat, p)


def main():
    args = sys.argv[1:]
    cmd = args[0] if args else "status"

    if cmd in ("info", "cliff"):
        script = {"info": "envinfo.py", "cliff": "cliff.py"}[cmd]
        sys.exit(subprocess.run([str(PY), str(ROOT / "runner" / script)],
                                cwd=ROOT).returncode)
    if cmd == "math":
        sys.exit(subprocess.run(
            [str(PY), str(ROOT / "runner" / "timings.py")] + args[1:],
            cwd=ROOT).returncode)

    arg = next((a for a in args[1:] if not a.startswith("-")), None)
    _, flat = load_stages()
    p = progress()
    table = {
        "status": lambda: cmd_status(flat, p),
        "guide": lambda: cmd_guide(flat, p, arg),
        "start": lambda: cmd_guide(flat, p, arg),
        "test": lambda: cmd_test(flat, p, arg),
        "submit": lambda: cmd_submit(flat, p),
        "list": lambda: cmd_list(flat, p),
        "ls": lambda: cmd_list(flat, p),
        "lore": lambda: cmd_lore(flat, p, arg),
        "peek": lambda: (cmd_peek_apply(flat, p, arg)
                         if "--apply" in args else cmd_peek(flat, p, arg)),
        "reset": lambda: cmd_reset(flat, p, arg),
    }
    if cmd not in table:
        print(__doc__)
        sys.exit(1)
    sys.exit(table[cmd]() or 0)


if __name__ == "__main__":
    main()
