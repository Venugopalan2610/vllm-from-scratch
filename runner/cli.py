"""vc - the stage runner.

  THE LOOP
    vc               where you are
    vc guide         what to build for the current stage, and why
    vc test          run the current stage's checks
    vc submit        run the checks; if green, bank it and open the next stage

  BACKENDS
    vc backend       which track you are on: torch or jax
    vc backend jax   switch to it. progress is tracked per track.
    vc <cmd> --jax   one command on the other track, without switching

  REFERENCE
    vc list          the whole ladder
    vc lore [stage]  the one-paragraph insight for any stage
    vc info          your GPU's roofline
    vc math [B] [n]  read-vs-compute timings, every division shown
    vc cliff         the L2-vs-VRAM bandwidth cliff
    vc ncu [stage]   hardware counters for a CUDA stage's kernel
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


def load_stages(backend="torch"):
    """The ladder for one track.

    A stage may carry `tracks: [torch]`, which keeps it off the other
    ladders. The CUDA stages do: a JAX learner has Pallas at stage 08 and no
    reason to write __shfl_xor_sync. A stage with no `tracks` key belongs to
    every track, which is almost all of them.
    """
    import yaml

    data = yaml.safe_load((ROOT / "stages.yaml").read_text())
    flat = []
    for arc in data["arcs"]:
        for s in arc["stages"]:
            if backend not in s.get("tracks", BACKENDS):
                continue
            s["arc"], s["arc_id"] = arc["name"], arc["id"]
            flat.append(s)
    return data, flat


BACKENDS = ("torch", "jax")


def progress():
    """Progress belongs to a TRACK. A pass on stage 08 in CUDA says nothing
    about the same kernel in Pallas. So the two ladders bank separately, and
    you can walk them in either order.

    `p["completed"]` is the active track's list -- the same list object that
    lives in p["tracks"], so mutating it mutates the track.
    """
    p = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
    if "tracks" not in p:
        p = {"backend": "torch", "tracks": {"torch": p.get("completed", [])}}
    p.setdefault("backend", "torch")
    for b in BACKENDS:
        p["tracks"].setdefault(b, [])
    p["completed"] = p["tracks"][p["backend"]]
    p["_persist"] = p["backend"]
    return p


def set_backend(p, name, persist=False):
    """`persist=False` is the `--jax` flag: act on the other track for one
    command, then go back to where you were living."""
    p["backend"] = name
    p["completed"] = p["tracks"][name]
    if persist:
        p["_persist"] = name
    return p


def save_progress(p):
    PROGRESS.write_text(json.dumps(
        {"backend": p.get("_persist", p["backend"]), "tracks": p["tracks"]},
        indent=2))


def stage_file(s, p):
    """The file the learner edits for this stage, on the active track.

    A stage with no `jax_file` is FRAMEWORK-FREE -- the block allocator, the
    scheduler, the detokenizer. There is nothing to port, so both tracks build
    and test the same file.
    """
    if p["backend"] == "jax":
        return s.get("jax_file", s["file"])
    return s["file"]


def stage_files(s, p):
    """Every file the learner edits for this stage.

    The CUDA stages are two: a .cu with the kernel and a .py wrapper that
    builds and calls it. Everything else is one, and `files` is absent.
    """
    if p["backend"] == "torch" and s.get("files"):
        return s["files"]
    return [stage_file(s, p)]


def is_shared(s):
    return "jax_file" not in s


def stage_name(s, p):
    """Two stages read differently per track: 08 is Triton or Pallas, 12 is
    CUDA graphs or shape buckets. Everything else shares a name."""
    if p["backend"] == "jax":
        return s.get("jax_name", s["name"])
    return s["name"]


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
        # "8b" never zfills to "08b", so compare without the leading zeros
        lambda s: s["id"].split("-")[0].lstrip("0") == q.lstrip("0"),
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


def stage_label(s):
    """The number a learner says out loud: "08", "08b", "12".

    Not the position in the ladder. Inserting 08b and 08c would otherwise
    renumber every stage after them on one track but not the other, and the
    two tracks share nine files.
    """
    return s["id"].split("-")[0]


def check_files(s, p):
    """The test files that belong to the active track."""
    d = test_dir(s)
    if not d.exists():
        return []
    files = sorted(d.glob("test_*.py"))
    twin = d / "test_jax.py"
    if p["backend"] == "jax":
        if twin.exists():
            return [twin]
        # test_cuda.py belongs to a torch-only stage. It never reaches here.
        return [f for f in files if f.name != "test_cuda.py"]
    return [f for f in files if f.name != "test_jax.py"]


def checks_for(s, p):
    """(name, one-line description) for every check in the stage."""
    out = []
    for f in check_files(s, p):
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
    print(f"{C['b']}{C['c']}  Stage {stage_label(s)} of {total}   "
          f"{stage_name(s, p)}{C['x']}   {C['dim']}[{stars}]{C['x']}")
    print(f"{C['dim']}  {s['arc_id']} - {s['arc']}   [{p['backend']}]"
          + ("   (already completed)" if done else "") + f"{C['x']}")
    print(f"{C['b']}{C['c']}{'=' * 74}{C['x']}")

    print(f"\n{C['b']}WHY THIS STAGE EXISTS{C['x']}")
    print(wrap(" ".join(s["insight"].split())))

    if p["backend"] == "jax" and s.get("jax_insight"):
        print(f"\n{C['b']}WHAT CHANGES IN JAX{C['x']}")
        print(wrap(" ".join(s["jax_insight"].split())))

    print(f"\n{C['b']}WHAT YOU'RE BUILDING{C['x']}")
    print(wrap(s.get("jax_deliver", s["deliver"])
               if p["backend"] == "jax" else s["deliver"]))
    files = stage_files(s, p)
    print(f"\n  {C['u']}{files[0]}{C['x']}"
          f"   {C['dim']}<- edit this; the full spec is in its docstrings{C['x']}")
    for extra in files[1:]:
        print(f"  {C['u']}{extra}{C['x']}   {C['dim']}<- and this{C['x']}")
    if p["backend"] == "jax" and is_shared(s):
        print(f"  {C['dim']}(framework-free: both tracks build this same file)"
              f"{C['x']}")

    print(f"\n{C['b']}HOW YOU'LL KNOW IT WORKED{C['x']}")
    print(wrap(s["measure"]))

    checks = checks_for(s, p)
    if checks:
        print(f"\n{C['b']}THE CHECKS{C['x']} {C['dim']}({len(checks)}){C['x']}")
        for name, desc in checks:
            print(f"  {C['dim']}[ ]{C['x']} {name[5:].replace('_', ' ')}")
            if desc:
                d = " ".join(desc.split())
                print(f"      {C['dim']}{d[:63] + '...' if len(d) > 66 else d}{C['x']}")

    print(f"\n{C['b']}NEXT{C['x']}")
    print(f"  {C['c']}./vc test{C['x']}     run the checks, as often as you like")
    nxt = flat[idx] if idx < len(flat) else None
    print(f"  {C['c']}./vc submit{C['x']}   bank it"
          + (f" and open stage {stage_label(nxt)}" if nxt else ""))
    print(f"  {C['dim']}./vc peek     reveal the reference solution{C['x']}\n")
    return 0


def run_checks(s, p):
    d = test_dir(s)
    if not d.exists():
        return 1, 0, 0, f"No checks authored for {s['id']}."
    r = subprocess.run(
        [str(PY), "-m", "pytest", str(d), "-q", "--timeout=900",
         "--no-header", "-rN", "--backend", p["backend"]],
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
    print(f"\n  {C['dim']}Spec: {s['_file']}   |   ./vc guide   |   ./vc peek{C['x']}\n")


def cmd_test(flat, p, stage_id=None):
    s, err = resolve(flat, p, stage_id)
    if not s:
        print(err)
        return 1
    s = dict(s, _file=" and ".join(stage_files(s, p)))
    print(f"\n{C['b']}{C['c']}Stage {stage_label(s)}  "
          f"{stage_name(s, p)}{C['x']}   {C['dim']}[{p['backend']}]{C['x']}")
    print(f"{C['dim']}running checks...{C['x']}\n")
    rc, passed, failed, out = run_checks(s, p)
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
    s = dict(s, _file=" and ".join(stage_files(s, p)))
    print(f"\n{C['b']}{C['c']}Submission: stage {stage_label(s)}  "
          f"{stage_name(s, p)}{C['x']}   {C['dim']}[{p['backend']}]{C['x']}")
    print(f"{C['dim']}running checks...{C['x']}\n")
    rc, passed, failed, out = run_checks(s, p)

    if rc != 0:
        show_failures(out, passed, failed, s)
        print(f"{C['r']}{C['b']}  NOT SUBMITTED{C['x']}"
              f"  {C['dim']}fix them and run ./vc submit again{C['x']}\n")
        return 1

    print(f"{C['g']}{C['b']}  ALL {passed} CHECKS PASSED{C['x']}")

    # Commit ONLY the work of the learner. Never use `git add -A` here. It
    # collects harness edits and every other loose change, and a later rewind
    # then throws them away with the stage.
    subprocess.run(["git", "add", "app/"], cwd=ROOT, capture_output=True)
    track = "" if p["backend"] == "torch" else f" [{p['backend']}]"
    msg = f"stage {stage_label(s)} complete: {stage_name(s, p)}{track}"
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
    other = "jax" if p["backend"] == "torch" else "torch"
    print(f"\n  {bar(n, total)}  {n}/{total} stages complete"
          f"  {C['dim']}[{p['backend']}]{C['x']}")
    print(f"  {C['dim']}{other}: {len(p['tracks'][other])}"
          f"/{len(load_stages(other)[1])}   (./vc backend {other}){C['x']}")
    if not s:
        print(f"\n  {C['b']}All done. You built vLLM.{C['x']}\n")
        return 0
    print(f"\n  {C['b']}Current: stage {stage_label(s)}  "
          f"{stage_name(s, p)}{C['x']}")
    for f in stage_files(s, p):
        print(f"  {C['dim']}{f}{C['x']}")
    print(f"\n  {C['c']}./vc guide{C['x']}   what to build and why")
    print(f"  {C['c']}./vc test{C['x']}    run the checks")
    print(f"  {C['c']}./vc submit{C['x']}  bank it, open the next stage\n")
    return 0


def cmd_backend(flat, p, name=None):
    if name is None:
        print(f"\n  backend: {C['b']}{p['backend']}{C['x']}")
        for b in BACKENDS:
            mark = ">" if b == p["backend"] else " "
            total = len(load_stages(b)[1])
            print(f"  {mark} {b:<6} {len(p['tracks'][b])}/{total} stages")
        print(f"\n  {C['dim']}./vc backend jax   switch tracks"
              f"   |   ./vc test --jax   just this once{C['x']}\n")
        return 0
    if name not in BACKENDS:
        print(f"{C['y']}Unknown backend '{name}'.{C['x']} "
              f"Pick one of: {', '.join(BACKENDS)}")
        return 1
    if name == "jax":
        try:
            subprocess.run([str(PY), "-c", "import jax"], check=True,
                           capture_output=True)
        except subprocess.CalledProcessError:
            print(f"{C['y']}JAX is not installed.{C['x']} "
                  f"{C['dim']}Run: ./setup.sh --jax{C['x']}")
            return 1
    save_progress(set_backend(p, name, persist=True))
    print(f"{C['g']}Switched to the {name} track.{C['x']}")
    _, flat = load_stages(name)
    return cmd_status(flat, p)


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
        nm = stage_name(s, p)
        name = f"{C['b']}{nm}{C['x']}" if is_cur else nm
        tag = ""
        if p["backend"] == "jax":
            tag = f"  {C['dim']}{'shared' if is_shared(s) else 'jax'}{C['x']}"
        print(f"  {mark} {stage_label(s):>3} {s['id']:<26} {name}  "
              f"{C['dim']}{'*' * s['difficulty']}{C['x']}{tag}")
    print(f"\n{C['dim']}{len(p['completed'])}/{len(flat)} complete "
          f"on the {p['backend']} track{C['x']}\n")
    return 0


def cmd_lore(flat, p, stage_id=None):
    s, err = resolve(flat, p, stage_id)
    if not s:
        print(err)
        return 1
    print(f"\n{C['b']}{C['c']}{s['id']}  {stage_name(s, p)}{C['x']}  "
          f"{C['dim']}({'*' * s['difficulty']}){C['x']}")
    print(f"\n{C['b']}The insight{C['x']}\n" + wrap(" ".join(s["insight"].split())))
    if p["backend"] == "jax" and s.get("jax_insight"):
        print(f"\n{C['b']}In JAX{C['x']}\n"
              + wrap(" ".join(s["jax_insight"].split())))
    print(f"\n{C['b']}Deliver{C['x']}\n  "
          + (s.get("jax_deliver", s["deliver"]) if p["backend"] == "jax"
             else s["deliver"]))
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

    print(f"{C['dim']}Fetch of the solutions branch...{C['x']}")
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
    files = stage_files(s, p)
    texts = [(f, solution_text(Path(f).name)) for f in files]
    if any(t is None for _, t in texts):
        print(f"{C['y']}Could not reach the solutions branch.{C['x']}")
        print(f"{C['dim']}Try: git fetch origin solutions{C['x']}")
        return 1
    print(f"\n{C['y']}Reference solution for {s['id']}{C['x']}")
    print(f"{C['dim']}To use it:  ./vc peek {stage_label(s)} --apply"
          f"   (or copy it by hand){C['x']}")
    for f, text in texts:
        print(f"\n{C['b']}{C['c']}--- {f} {'-' * max(0, 60 - len(f))}{C['x']}\n")
        print(text)
    return 0


def cmd_peek_apply(flat, p, stage_id=None):
    s, err = resolve(flat, p, stage_id)
    if not s:
        print(err)
        return 1
    targets = stage_files(s, p)
    texts = [(f, solution_text(Path(f).name)) for f in targets]
    if any(t is None for _, t in texts):
        print(f"{C['y']}Could not reach the solutions branch.{C['x']}")
        return 1
    for f, text in texts:
        (ROOT / f).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / f).write_text(text)
        print(f"{C['y']}Wrote the reference solution into {f}.{C['x']}")
    print(f"{C['dim']}Read it before you submit. The reading was the point.{C['x']}")
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
    print(f"{C['y']}Rewound to stage {stage_label(s)}.{C['x']} "
          f"{C['dim']}Your code in app/ did not change.{C['x']}")
    return cmd_guide(flat, p)


def main():
    args = sys.argv[1:]
    cmd = args[0] if args else "status"

    if cmd in ("info", "cliff"):
        script = {"info": "envinfo.py", "cliff": "cliff.py"}[cmd]
        sys.exit(subprocess.run([str(PY), str(ROOT / "runner" / script)],
                                cwd=ROOT).returncode)
    if cmd == "ncu":
        sys.exit(subprocess.run(
            [str(PY), str(ROOT / "runner" / "ncu.py")] + args[1:],
            cwd=ROOT).returncode)
    if cmd == "math":
        sys.exit(subprocess.run(
            [str(PY), str(ROOT / "runner" / "timings.py")] + args[1:],
            cwd=ROOT).returncode)

    arg = next((a for a in args[1:] if not a.startswith("-")), None)
    p = progress()

    # `--jax` / `--torch` on any command: run it against the other track
    # WITHOUT switching. Handy for `./vc test 8 --jax` while you live in torch.
    for b in BACKENDS:
        if f"--{b}" in args:
            set_backend(p, b)

    # The ladder itself is per-track: the CUDA stages are not on the JAX one.
    _, flat = load_stages(p["backend"])

    table = {
        "backend": lambda: cmd_backend(flat, p, arg),
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
