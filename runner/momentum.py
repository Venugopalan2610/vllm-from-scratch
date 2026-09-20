"""What ./vc says to keep you going: the motto of the course, applied.

    Do not stop. Continue. Be better than before.

This module keeps a small history for each stage. It counts the runs of the
checks. It keeps your best result, and the runs since that best result.
From that history it chooses one message. It never blocks a command, and it
never changes a result. It only tells you where you are, and what to do next.

The history lives in .progress.json, next to your completed stages.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

MOTTO = "Do not stop. Continue. Be better than before."
STUCK_RUNS = 5          # runs with no new pass before ./vc offers a way out
AWAY_DAYS = 3           # days away before ./vc says "welcome back"
PREFORMATTED = "  "     # a line that starts with this keeps its own spacing
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class StageRecord:
    runs: int = 0
    best: int = -1              # the most checks that passed in one run
    total: int = 0
    runs_since_best: int = 0

    @classmethod
    def load(cls, data):
        return cls(**data) if data else cls()

    def as_dict(self):
        return dict(runs=self.runs, best=self.best, total=self.total,
                    runs_since_best=self.runs_since_best)


@dataclass
class RunResult:
    passed: int
    total: int
    previous_best: int          # -1 on the first run
    runs: int
    runs_since_best: int

    @property
    def is_first(self):
        return self.previous_best < 0

    @property
    def is_better(self):
        return self.passed > self.previous_best

    @property
    def is_stuck(self):
        return self.runs_since_best >= STUCK_RUNS


def records(progress):
    """-> {stage id: record data} for the active track. It is a live dict:
    a change to it changes the progress that save_progress() writes."""
    history = progress.setdefault("history", {})
    return history.setdefault(progress["backend"], {})


def record_run(progress, stage_id, passed, total):
    """Add one run of the checks to the history. -> the RunResult."""
    stage_records = records(progress)
    record = StageRecord.load(stage_records.get(stage_id))
    previous_best = record.best
    record.runs += 1
    record.total = total
    if passed > record.best:
        record.best = passed
        record.runs_since_best = 0
    else:
        record.runs_since_best += 1
    stage_records[stage_id] = record.as_dict()
    progress["last_seen"] = date.today().isoformat()
    return RunResult(passed, total, previous_best, record.runs, record.runs_since_best)


def stage_record(progress, stage_id):
    return StageRecord.load(records(progress).get(stage_id))


def days_away(progress, today=None):
    """-> the days since the last run of the checks, or None if none ran."""
    last_seen = progress.get("last_seen")
    if not last_seen:
        return None
    return ((today or date.today()) - date.fromisoformat(last_seen)).days


def notebooks_for(arc_id):
    """The notebook folder of an arc: arc A2 is course Part 3."""
    part_number = int(arc_id.lstrip("A")) + 1
    folders = sorted((ROOT / "course").glob(f"Part{part_number}_*"))
    return folders[0].relative_to(ROOT) if folders else None


# ---------------------------------------------------------------- messages
# Each function returns lines of (text, styles). The caller paints them, so
# this module does not depend on the terminal.


def after_run(result):
    """One or two lines after `./vc test`, when some checks fail."""
    score = f"{result.passed}/{result.total}"
    best = f"{result.previous_best}/{result.total}"
    if result.is_first:
        return [(f"First run: {score}. Each check that passes is one step "
                 "forward.", ("bold",))]
    if result.is_better:
        return [(f"Better than before: {score}. Your best was {best}.",
                 ("green", "bold"))]
    if result.passed == result.previous_best:
        return [(f"The same as your best: {score}. Read the first failure. It "
                 "names the part of the spec that is not true yet.", ("yellow",))]
    return [(f"{score}. Your best is {best}. A step back is part of the "
             "work. Continue.", ("yellow",))]


def when_stuck(result, stage_label, arc_id):
    """The way out, after STUCK_RUNS runs with no new pass."""
    lines = [(MOTTO, ("bold", "cyan")),
             (f"{result.runs_since_best} runs with no new pass. Change the "
              "approach, not the effort. Do one of these:", ())]
    steps = [(f"./vc guide {stage_label}", "read WHY THIS STAGE EXISTS again, and the new words")]
    notebooks = notebooks_for(arc_id)
    if notebooks:
        steps.append((f"{notebooks}/", "do the notebooks of this part first"))
    steps.append((f"./vc peek {stage_label}", "read the answer, close it, then write it yourself"))
    width = max(len(command) for command, _ in steps)
    lines += [(f"{PREFORMATTED}{number}. {command:<{width}}   {why}", ("dim",))
              for number, (command, why) in enumerate(steps, 1)]
    return lines


def after_submit(record, stage_label):
    """After a stage passes and the harness banks it."""
    runs = "1 run" if record.runs == 1 else f"{record.runs} runs"
    return [(f"Stage {stage_label} done, in {runs} of the checks. You are "
             "better than before.", ("green", "bold"))]


def after_arc(arc_id, arc_name, blurb):
    """When the submitted stage was the last stage of its arc."""
    return [(f"Arc {arc_id} complete: {arc_name}.", ("green", "bold")),
            (blurb, ("dim",))]


def first_visit():
    """For `./vc` before the first run of any check."""
    return [("Welcome. Stage 01 is the slowest inference engine that you will "
             "ever write. That is the plan: each stage after it must be "
             "better than before, and your GPU measures it.", ("bold",))]


def welcome_back(days, stage_label, record):
    """For `./vc` after some days away."""
    lines = [(f"Welcome back. {days} days since your last run.", ("bold",))]
    if record.best >= 0:
        lines.append((f"Stage {stage_label}: your best is {record.best}/"
                      f"{record.total} checks. Continue from there.", ()))
    return lines


def after_peek():
    return [("Read it. Close it. Write it yourself, in your own words. Then "
             "continue.", ("cyan",))]


def finished(progress):
    total_runs = sum(record["runs"] for record in records(progress).values())
    return [("Every stage passes. You built a single-GPU inference engine "
             "inspired by vLLM.", ("bold", "green")),
            (f"{total_runs} runs of the checks. Each stage was better than the "
             "one before it.", ()),
            (MOTTO, ("cyan",))]
