"""vc - the stage runner.

  THE LOOP
    vc               where you are
    vc guide         what to build for the current stage, and why
    vc test          run the checks of the current stage
    vc submit        run the checks. If they pass, bank the stage, open the next

  BACKENDS
    vc backend       the track that you are on: torch or jax
    vc backend jax   change to it. Each track has its own progress.
    vc <cmd> --jax   one command on the other track, with no change

  REFERENCE
    vc list          the whole ladder
    vc lore [stage]  the one-paragraph insight of a stage
    vc info          the roofline of your GPU
    vc math [B] [n]  read and compute times, with every division shown
    vc ncu [stage]   the hardware counters of the kernel of a CUDA stage
    vc nsys          timeline trace of the engine using NVIDIA Nsight Systems
    vc bench         your capstone engine against the roof of this card
    vc serve         your capstone server, on localhost:8000
    vc peek [stage]  show the reference solution (from the solutions branch)
    vc peek [n] --apply   write it into the file of the stage
    vc reset <n>     go back to stage n (your code does not change)
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner import momentum  # noqa: E402
from runner.glossary import terms_for_stage  # noqa: E402
from runner.style import paint  # noqa: E402  (needs ROOT on the path)

PROGRESS_FILE = ROOT / ".progress.json"
PYTHON = ROOT / ".venv" / "bin" / "python"
BACKENDS = ("torch", "jax")

# ---------------------------------------------------------------- the ladder


def load_stages(backend="torch"):
    """The ladder of one track.

    A stage can have `tracks: [torch]`, which keeps it off the other ladder.
    The CUDA stages have it: a JAX learner has Pallas at stage 08 and no
    reason to write __shfl_xor_sync. A stage with no `tracks` key belongs to
    every track, and almost all stages are like that.
    """
    import yaml

    data = yaml.safe_load((ROOT / "stages.yaml").read_text())
    ladder = []
    for arc in data["arcs"]:
        for stage in arc["stages"]:
            if backend not in stage.get("tracks", BACKENDS):
                continue
            stage["arc"], stage["arc_id"] = arc["name"], arc["id"]
            stage["arc_blurb"] = arc.get("blurb", "")
            ladder.append(stage)
    return data, ladder


def stage_file(stage, progress):
    """The file that the learner edits for this stage, on the active track.

    A stage with no `jax_file` uses NO FRAMEWORK: the block allocator, the
    scheduler, the detokenizer. It has nothing to port, so both tracks build
    and test the same file.
    """
    if progress["backend"] == "jax":
        return stage.get("jax_file", stage["file"])
    return stage["file"]


def stage_files(stage, progress):
    """All the files that the learner edits for this stage.

    A CUDA stage has two: a .cu with the kernel and a .py wrapper that builds
    and calls it. The other stages have one, and no `files` key.
    """
    if progress["backend"] == "torch" and stage.get("files"):
        return stage["files"]
    return [stage_file(stage, progress)]


def is_shared(stage):
    return "jax_file" not in stage


def stage_name(stage, progress):
    """Two stages have a different name on each track: 08 is CUDA or Pallas,
    12 is CUDA graphs or shape buckets. The other stages share a name."""
    if progress["backend"] == "jax":
        return stage.get("jax_name", stage["name"])
    return stage["name"]


def stage_text(stage, progress, field):
    """`insight` or `deliver`, with the JAX version on the JAX track."""
    if progress["backend"] == "jax" and field == "deliver":
        return stage.get("jax_deliver", stage["deliver"])
    return stage[field]


def current_stage(ladder, progress):
    for stage in ladder:
        if stage["id"] not in progress["completed"]:
            return stage
    return None


def stage_label(stage):
    """The number that a learner says: "08", "08b", "12".

    It is not the position in the ladder. With positions, the insertion of
    08b and 08c changes the number of every later stage on one track but not
    on the other. The two tracks share nine files.
    """
    return stage["id"].split("-")[0]


def stage_index(ladder, stage):
    """The position of the stage in the ladder, from 1."""
    return next(index for index, other in enumerate(ladder, 1)
                if other["id"] == stage["id"])


def _matchers(query):
    """The tests of a stage against a query, from strict to loose."""
    return (
        lambda stage: stage["id"] == query,
        lambda stage: stage_label(stage) == query.zfill(2),
        # "8b" does not zfill to "08b", so compare with no leading zeros.
        lambda stage: stage_label(stage).lstrip("0") == query.lstrip("0"),
        lambda stage: stage["id"].startswith(query),
        lambda stage: query in stage["id"] or query in stage["name"].lower(),
    )


def resolve(ladder, progress, stage_id):
    """-> (stage, None) or (None, the error message)."""
    if stage_id is None:
        stage = current_stage(ladder, progress)
        return stage, None if stage else "All stages complete."
    query = stage_id.strip().lower()
    for matches in _matchers(query):
        hits = [stage for stage in ladder if matches(stage)]
        if len(hits) == 1:
            return hits[0], None
        if len(hits) > 1:
            return None, (paint(f"'{stage_id}' is ambiguous:", "yellow") + " "
                          + ", ".join(hit["id"] for hit in hits))
    return None, (
        paint(f"No stage matches '{stage_id}'.", "yellow") + "\n"
        + paint("Try a number (1, 7), an id (07-paged-attention), or a word. "
                "`vc list` shows them all.", "dim"))


def resolve_or_report(ladder, progress, stage_id):
    """The stage, or None after it prints the error."""
    stage, error = resolve(ladder, progress, stage_id)
    if error:
        print(error)
    return stage


# ---------------------------------------------------------------- progress


def load_progress():
    """Progress belongs to a TRACK. A pass on stage 08 in CUDA says nothing
    about the same kernel in Pallas. So each track banks separately, and you
    can do the two ladders in either order.

    `progress["completed"]` is the list of the active track. It is the same
    list object as the one in progress["tracks"], so a change to it changes
    the track.
    """
    progress = (json.loads(PROGRESS_FILE.read_text())
                if PROGRESS_FILE.exists() else {})
    if "tracks" not in progress:
        progress = {"backend": "torch",
                    "tracks": {"torch": progress.get("completed", [])}}
    progress.setdefault("backend", "torch")
    for backend in BACKENDS:
        progress["tracks"].setdefault(backend, [])
    progress["completed"] = progress["tracks"][progress["backend"]]
    progress["_persist"] = progress["backend"]
    return progress


def set_backend(progress, name, persist=False):
    """`persist=False` is the `--jax` flag: use the other track for one
    command, then go back to the track that you were on."""
    progress["backend"] = name
    progress["completed"] = progress["tracks"][name]
    if persist:
        progress["_persist"] = name
    return progress


def save_progress(progress):
    backend = progress.get("_persist", progress["backend"])
    saved = {"backend": backend, "tracks": progress["tracks"]}
    for key in ("history", "last_seen"):          # runner/momentum.py keeps these
        if key in progress:
            saved[key] = progress[key]
    PROGRESS_FILE.write_text(json.dumps(saved, indent=2))


# ---------------------------------------------------------------- checks


def test_dir(stage):
    return ROOT / "tests" / f"stage_{stage['id'].replace('-', '_')}"


def check_files(stage, progress):
    """The test files that belong to the active track."""
    directory = test_dir(stage)
    if not directory.exists():
        return []
    files = sorted(directory.glob("test_*.py"))
    jax_twin = directory / "test_jax.py"
    if progress["backend"] != "jax":
        return [path for path in files if path.name != "test_jax.py"]
    if jax_twin.exists():
        return [jax_twin]
    # A test_cuda.py belongs to a torch-only stage. It never gets here.
    return [path for path in files if path.name != "test_cuda.py"]


def checks_for(stage, progress):
    """(name, one-line description) for each check of the stage."""
    checks = []
    for path in check_files(stage, progress):
        source = path.read_text()
        for test in re.finditer(r"^def (test_\w+)\(", source, re.M):
            docstring = re.match(r'[^)]*\):\n\s*"""(.*?)(?:\n|""")',
                                 source[test.end():], re.S)
            description = (docstring.group(1).strip().rstrip('"').strip()
                           if docstring else "")
            checks.append((test.group(1), description))
    return checks


def _count(pattern, output):
    found = re.search(pattern, output)
    return int(found.group(1)) if found else 0


def run_checks(stage, progress):
    """-> (return code, passed, failed, pytest output)."""
    directory = test_dir(stage)
    if not directory.exists():
        return 1, 0, 0, f"No checks written for {stage['id']}."
    result = subprocess.run(
        [str(PYTHON), "-m", "pytest", str(directory), "-q", "--timeout=900",
         "--no-header", "-rN", "--backend", progress["backend"]],
        cwd=ROOT, capture_output=True, text=True)
    output = result.stdout + result.stderr
    failed = _count(r"(\d+) failed", output) + _count(r"(\d+) error", output)
    return result.returncode, _count(r"(\d+) passed", output), failed, output


def show_failures(output, passed, failed, stage_files_text):
    body = output.split("= FAILURES =")[-1] if "FAILURES" in output else output
    print(body.strip()[:4000])
    print("\n" + paint(f"  {passed}/{passed + failed} checks pass", "red", "bold"))
    print("\n  " + paint(f"Spec: {stage_files_text}   |   ./vc guide   |   "
                         f"./vc peek", "dim") + "\n")


# ---------------------------------------------------------------- output


def wrap(text, width=72, indent="  "):
    lines, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            lines.append(indent + line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(indent + line)
    return "\n".join(lines)


def progress_bar(done, total, width=28):
    filled = int(width * done / total)
    return paint(f"[{'#' * filled}{'.' * (width - filled)}]", "green")


def heading(text):
    print("\n" + paint(text, "bold"))


def say(lines):
    """Print the (text, styles) lines that runner/momentum.py returns."""
    for text, styles in lines:
        if not text.startswith(momentum.PREFORMATTED):
            text = wrap(text)
        else:
            text = "  " + text
        print(paint(text, *styles) if styles else text)


def print_stage_banner(stage, progress, total):
    stars = "*" * stage["difficulty"] + "." * (5 - stage["difficulty"])
    already = ("   (already completed)"
               if stage["id"] in progress["completed"] else "")
    rule = paint("=" * 74, "bold", "cyan")
    print("\n" + rule)
    print(paint(f"  Stage {stage_label(stage)} of {total}   "
                f"{stage_name(stage, progress)}", "bold", "cyan")
          + "   " + paint(f"[{stars}]", "dim"))
    print(paint(f"  {stage['arc_id']} - {stage['arc']}   "
                f"[{progress['backend']}]{already}", "dim"))
    print(rule)


def print_terms(ladder, stage):
    """The words that this stage teaches, each with its definition and its
    analogy from software. course/GLOSSARY.md is the source."""
    terms = terms_for_stage(stage_label(stage), [stage_label(item) for item in ladder])
    if not terms:
        return
    heading("NEW WORDS IN THIS STAGE")
    for term in terms:
        print("  " + paint(term.name, "bold"))
        print(wrap(term.definition, indent="    "))
        if "Like" in term.fields:
            print(paint(wrap("Like: " + term.fields["Like"], indent="    "), "dim"))
    print("\n  " + paint("course/GLOSSARY.md has all the terms of the course.", "dim"))


def print_files(stage, progress):
    files = stage_files(stage, progress)
    print("\n  " + paint(files[0], "underline") + "   "
          + paint("<- edit this; the full spec is in its docstrings", "dim"))
    for extra in files[1:]:
        print("  " + paint(extra, "underline") + "   " + paint("<- and this", "dim"))
    if progress["backend"] == "jax" and is_shared(stage):
        print("  " + paint("(no framework: both tracks build this same file)",
                           "dim"))


def print_checks(stage, progress):
    checks = checks_for(stage, progress)
    if not checks:
        return
    print("\n" + paint("THE CHECKS", "bold") + " "
          + paint(f"({len(checks)})", "dim"))
    for name, description in checks:
        print("  " + paint("[ ]", "dim") + " " + name[5:].replace("_", " "))
        if description:
            flat = " ".join(description.split())
            short = flat[:63] + "..." if len(flat) > 66 else flat
            print("      " + paint(short, "dim"))


def print_next_steps(ladder, stage):
    index = stage_index(ladder, stage)
    following = ladder[index] if index < len(ladder) else None
    heading("NEXT")
    print("  " + paint("./vc test", "cyan") + "     run the checks, as often "
          "as you like")
    print("  " + paint("./vc submit", "cyan") + "   bank it"
          + (f" and open stage {stage_label(following)}" if following else ""))
    print("  " + paint("./vc peek     show the reference solution", "dim") + "\n")


# ---------------------------------------------------------------- commands


def cmd_guide(ladder, progress, stage_id=None):
    stage = resolve_or_report(ladder, progress, stage_id)
    if not stage:
        return 1
    print_stage_banner(stage, progress, len(ladder))
    print_terms(ladder, stage)
    heading("WHY THIS STAGE EXISTS")
    print(wrap(" ".join(stage["insight"].split())))
    if progress["backend"] == "jax" and stage.get("jax_insight"):
        heading("WHAT CHANGES IN JAX")
        print(wrap(" ".join(stage["jax_insight"].split())))
    heading("WHAT YOU ARE BUILDING")
    print(wrap(stage_text(stage, progress, "deliver")))
    print_files(stage, progress)
    heading("HOW YOU WILL KNOW IT WORKED")
    print(wrap(stage["measure"]))
    print_checks(stage, progress)
    print_next_steps(ladder, stage)
    return 0


def print_run_header(title, stage, progress):
    print("\n" + paint(f"{title} {stage_label(stage)}  "
                       f"{stage_name(stage, progress)}", "bold", "cyan")
          + "   " + paint(f"[{progress['backend']}]", "dim"))
    print(paint("running checks...", "dim") + "\n")


def run_and_record(stage, progress):
    """Run the checks, and add the run to the history of the stage.
    -> (return code, passed, failed, output, RunResult or None)."""
    return_code, passed, failed, output = run_checks(stage, progress)
    result = None
    if passed + failed:          # 0 and 0 means that pytest did not start
        result = momentum.record_run(progress, stage["id"], passed, passed + failed)
        save_progress(progress)
    return return_code, passed, failed, output, result


def report_failure(stage, progress, passed, failed, output, result):
    show_failures(output, passed, failed, " and ".join(stage_files(stage, progress)))
    if result is None:
        return
    say(momentum.after_run(result))
    if result.is_stuck:
        print()
        say(momentum.when_stuck(result, stage_label(stage), stage["arc_id"]))
    print()


def cmd_test(ladder, progress, stage_id=None):
    stage = resolve_or_report(ladder, progress, stage_id)
    if not stage:
        return 1
    print_run_header("Stage", stage, progress)
    return_code, passed, failed, output, result = run_and_record(stage, progress)
    if return_code != 0:
        report_failure(stage, progress, passed, failed, output, result)
        return return_code
    print(paint(f"  ALL {passed} CHECKS PASSED", "green", "bold"))
    print("\n  " + paint("./vc submit", "cyan") + " to bank it and open the "
          "next stage.\n")
    return 0


def commit_stage(stage, progress):
    """Commit ONLY the work of the learner. Never use `git add -A` here. It
    collects harness edits and every other loose change, and a later rewind
    then throws them away with the stage."""
    subprocess.run(["git", "add", "app/"], cwd=ROOT, capture_output=True)
    track = "" if progress["backend"] == "torch" else f" [{progress['backend']}]"
    message = (f"stage {stage_label(stage)} complete: "
               f"{stage_name(stage, progress)}{track}")
    result = subprocess.run(
        ["git", "-c", "user.email=you@localhost", "-c", "user.name=you",
         "commit", "-m", message],
        cwd=ROOT, capture_output=True, text=True)
    if result.returncode == 0:
        print(paint(f"  committed: {message}", "dim"))


def cmd_submit(ladder, progress):
    stage = current_stage(ladder, progress)
    if not stage:
        print("\n" + paint("All stages complete. You built a single-GPU "
                           "inference engine inspired by vLLM.", "bold") + "\n")
        return 0
    print_run_header("Submission: stage", stage, progress)
    return_code, passed, failed, output, result = run_and_record(stage, progress)
    if return_code != 0:
        report_failure(stage, progress, passed, failed, output, result)
        print(paint("  NOT SUBMITTED", "red", "bold") + "  "
              + paint("fix them and run ./vc submit again", "dim") + "\n")
        return 1

    print(paint(f"  ALL {passed} CHECKS PASSED", "green", "bold"))
    commit_stage(stage, progress)
    progress["completed"].append(stage["id"])
    save_progress(progress)
    done, total = len(progress["completed"]), len(ladder)
    print(f"\n  {progress_bar(done, total)}  {done}/{total} stages\n")
    say(momentum.after_submit(momentum.stage_record(progress, stage["id"]),
                              stage_label(stage)))
    following = current_stage(ladder, progress)
    if following is None or following["arc_id"] != stage["arc_id"]:
        print()
        say(momentum.after_arc(stage["arc_id"], stage["arc"], stage["arc_blurb"]))
    if following:
        return cmd_guide(ladder, progress)
    print()
    say(momentum.finished(progress))
    print()
    return 0


def print_momentum(stage, progress):
    """On the status screen: a welcome on the first visit, a welcome back
    after a break, or your best result on the current stage."""
    record = momentum.stage_record(progress, stage["id"])
    days = momentum.days_away(progress)
    if days is None and not progress["completed"]:
        print()
        say(momentum.first_visit())
    elif days is not None and days >= momentum.AWAY_DAYS:
        print()
        say(momentum.welcome_back(days, stage_label(stage), record))
    elif record.best >= 0:
        print("  " + paint(f"Your best: {record.best}/{record.total} checks, "
                           f"in {record.runs} runs.", "dim"))


def cmd_status(ladder, progress):
    stage = current_stage(ladder, progress)
    done, total = len(progress["completed"]), len(ladder)
    other = "jax" if progress["backend"] == "torch" else "torch"
    other_total = len(load_stages(other)[1])
    print(f"\n  {progress_bar(done, total)}  {done}/{total} stages complete  "
          + paint(f"[{progress['backend']}]", "dim"))
    print("  " + paint(f"{other}: {len(progress['tracks'][other])}"
                       f"/{other_total}   (./vc backend {other})", "dim"))
    if not stage:
        print()
        say(momentum.finished(progress))
        print()
        return 0
    print("\n  " + paint(f"Current: stage {stage_label(stage)}  "
                         f"{stage_name(stage, progress)}", "bold"))
    for path in stage_files(stage, progress):
        print("  " + paint(path, "dim"))
    print_momentum(stage, progress)
    print("\n  " + paint("./vc guide", "cyan") + "   what to build and why")
    print("  " + paint("./vc test", "cyan") + "    run the checks")
    print("  " + paint("./vc submit", "cyan") + "  bank it, open the next stage")
    print("\n  " + paint(momentum.MOTTO, "dim") + "\n")
    return 0


def jax_is_installed():
    result = subprocess.run([str(PYTHON), "-c", "import jax"],
                            capture_output=True)
    return result.returncode == 0


def print_backends(progress):
    print("\n  backend: " + paint(progress["backend"], "bold"))
    for backend in BACKENDS:
        marker = ">" if backend == progress["backend"] else " "
        total = len(load_stages(backend)[1])
        print(f"  {marker} {backend:<6} "
              f"{len(progress['tracks'][backend])}/{total} stages")
    print("\n  " + paint("./vc backend jax   change the track   |   "
                         "./vc test --jax   one time only", "dim") + "\n")


def cmd_backend(ladder, progress, name=None):
    if name is None:
        print_backends(progress)
        return 0
    if name not in BACKENDS:
        print(paint(f"Unknown backend '{name}'.", "yellow")
              + f" Select one of: {', '.join(BACKENDS)}")
        return 1
    if name == "jax" and not jax_is_installed():
        print(paint("JAX is not installed.", "yellow") + " "
              + paint("Run: ./setup.sh --jax", "dim"))
        return 1
    save_progress(set_backend(progress, name, persist=True))
    print(paint(f"Changed to the {name} track.", "green"))
    _, ladder = load_stages(name)
    return cmd_status(ladder, progress)


def cmd_list(ladder, progress):
    current = current_stage(ladder, progress)
    arc = None
    for stage in ladder:
        if stage["arc"] != arc:
            arc = stage["arc"]
            print("\n" + paint(f"{stage['arc_id']}  {arc}", "bold", "cyan"))
        is_current = current is not None and stage["id"] == current["id"]
        if stage["id"] in progress["completed"]:
            marker = paint("[x]", "green")
        elif is_current:
            marker = paint("[>]", "yellow")
        else:
            marker = "[ ]"
        name = stage_name(stage, progress)
        if is_current:
            name = paint(name, "bold")
        tag = ""
        if progress["backend"] == "jax":
            tag = "  " + paint("shared" if is_shared(stage) else "jax", "dim")
        print(f"  {marker} {stage_label(stage):>3} {stage['id']:<26} {name}  "
              + paint("*" * stage["difficulty"], "dim") + tag)
    print("\n" + paint(f"{len(progress['completed'])}/{len(ladder)} complete "
                       f"on the {progress['backend']} track", "dim") + "\n")
    return 0


def cmd_lore(ladder, progress, stage_id=None):
    stage = resolve_or_report(ladder, progress, stage_id)
    if not stage:
        return 1
    print("\n" + paint(f"{stage['id']}  {stage_name(stage, progress)}",
                       "bold", "cyan")
          + "  " + paint(f"({'*' * stage['difficulty']})", "dim"))
    heading("The insight")
    print(wrap(" ".join(stage["insight"].split())))
    if progress["backend"] == "jax" and stage.get("jax_insight"):
        heading("In JAX")
        print(wrap(" ".join(stage["jax_insight"].split())))
    heading("Deliver")
    print("  " + stage_text(stage, progress, "deliver"))
    heading("Measure")
    print(f"  {stage['measure']}\n")
    return 0


# ---------------------------------------------------------------- solutions


def _git_show(ref, name):
    result = subprocess.run(["git", "show", f"{ref}:.solutions/{name}"],
                            cwd=ROOT, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def solution_text(name):
    """The reference solutions are on the `solutions` branch, not on master.

    The order: a local .solutions/ directory, then the branch, then the
    remote branch. Fetch the remote branch one time if it is not here.
    """
    local = ROOT / ".solutions" / name
    if local.exists():
        return local.read_text()
    for ref in ("solutions", "origin/solutions"):
        text = _git_show(ref, name)
        if text:
            return text
    print(paint("Fetch of the solutions branch...", "dim"))
    subprocess.run(
        ["git", "fetch", "origin", "solutions:refs/remotes/origin/solutions"],
        cwd=ROOT, capture_output=True)
    return _git_show("origin/solutions", name)


def solutions_for(stage, progress):
    """[(path, text)] for each file of the stage, or None if one is missing."""
    solutions = [(path, solution_text(Path(path).name))
                 for path in stage_files(stage, progress)]
    if any(text is None for _, text in solutions):
        print(paint("Could not get the solutions branch.", "yellow"))
        print(paint("Try: git fetch origin solutions", "dim"))
        return None
    return solutions


def cmd_peek(ladder, progress, stage_id=None):
    stage = resolve_or_report(ladder, progress, stage_id)
    solutions = stage and solutions_for(stage, progress)
    if not solutions:
        return 1
    print("\n" + paint(f"Reference solution for {stage['id']}", "yellow"))
    print(paint(f"To use it:  ./vc peek {stage_label(stage)} --apply"
                f"   (or copy it by hand)", "dim"))
    for path, text in solutions:
        print("\n" + paint(f"--- {path} {'-' * max(0, 60 - len(path))}",
                           "bold", "cyan") + "\n")
        print(text)
    say(momentum.after_peek())
    print()
    return 0


def cmd_peek_apply(ladder, progress, stage_id=None):
    stage = resolve_or_report(ladder, progress, stage_id)
    solutions = stage and solutions_for(stage, progress)
    if not solutions:
        return 1
    for path, text in solutions:
        target = ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print(paint(f"Wrote the reference solution into {path}.", "yellow"))
    print(paint("Read it before you submit. The reading was the point.", "dim"))
    say(momentum.after_peek())
    return 0


def cmd_reset(ladder, progress, stage_id):
    if not stage_id:
        print("usage: vc reset <stage-number>")
        return 1
    stage = resolve_or_report(ladder, progress, stage_id)
    if not stage:
        return 1
    earlier = {other["id"]
               for other in ladder[:stage_index(ladder, stage) - 1]}
    progress["completed"][:] = [done for done in progress["completed"]
                                if done in earlier]
    save_progress(progress)
    print(paint(f"Back to stage {stage_label(stage)}.", "yellow") + " "
          + paint("Your code in app/ did not change.", "dim"))
    return cmd_guide(ladder, progress)


# ---------------------------------------------------------------- main

# The commands that run a script of their own, with the other arguments.
SCRIPTS = {"info": "envinfo.py", "cliff": "cliff.py", "ncu": "ncu.py",
           "nsys": "nsys.py",
           "bench": "bench.py", "serve": "serve.py", "math": "timings.py",
           "doctor": "doctor.py"}
NO_ARGUMENTS = ("info", "cliff", "doctor", "nsys")


def run_script(command, arguments):
    extra = [] if command in NO_ARGUMENTS else arguments
    script = ROOT / "runner" / SCRIPTS[command]
    return subprocess.run([str(PYTHON), str(script)] + extra,
                          cwd=ROOT).returncode


def main():
    arguments = sys.argv[1:]
    command = arguments[0] if arguments else "status"
    if command in SCRIPTS:
        sys.exit(run_script(command, arguments[1:]))

    stage_id = next((a for a in arguments[1:] if not a.startswith("-")), None)
    progress = load_progress()
    # `--jax` or `--torch` on a command: use the other track for this command
    # ONLY. Useful for `./vc test 8 --jax` while you work in torch.
    for backend in BACKENDS:
        if f"--{backend}" in arguments:
            set_backend(progress, backend)
    # Each track has its own ladder: the CUDA stages are not on the JAX one.
    _, ladder = load_stages(progress["backend"])

    commands = {
        "backend": lambda: cmd_backend(ladder, progress, stage_id),
        "status": lambda: cmd_status(ladder, progress),
        "guide": lambda: cmd_guide(ladder, progress, stage_id),
        "start": lambda: cmd_guide(ladder, progress, stage_id),
        "test": lambda: cmd_test(ladder, progress, stage_id),
        "submit": lambda: cmd_submit(ladder, progress),
        "list": lambda: cmd_list(ladder, progress),
        "ls": lambda: cmd_list(ladder, progress),
        "lore": lambda: cmd_lore(ladder, progress, stage_id),
        "peek": lambda: (cmd_peek_apply(ladder, progress, stage_id)
                         if "--apply" in arguments
                         else cmd_peek(ladder, progress, stage_id)),
        "reset": lambda: cmd_reset(ladder, progress, stage_id),
    }
    if command not in commands:
        print(__doc__)
        sys.exit(1)
    sys.exit(commands[command]() or 0)


if __name__ == "__main__":
    main()
