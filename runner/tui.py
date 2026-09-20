"""Terminal User Interface (TUI) for vc - the stage runner.

Run with:
  ./vc tui
  ./vc ui
"""

import curses
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.cli import (
    BACKENDS,
    checks_for,
    commit_stage,
    load_progress,
    load_stages,
    run_checks,
    save_progress,
    stage_files,
    stage_label,
    stage_name,
)
from runner.glossary import terms_for_stage


def is_completed(stage, progress):
    return stage["id"] in progress["completed"]


def get_gpu_info():
    """Extract brief GPU device string."""
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            mem_gb = p.total_memory / 1e9
            return f"{p.name} ({mem_gb:.1f} GB) · CC {p.major}.{p.minor}"
        return "CPU / No CUDA"
    except Exception:
        return "GPU unavailable"


class VcTUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.progress = load_progress()
        self.backend = self.progress["backend"]
        self.data, self.ladder = load_stages(self.backend)
        self.selected_idx = 0
        self.active_tab = 0  # 0: Guide, 1: Lore, 2: Checks, 3: Solution, 4: Console
        self.tabs = ["1: Guide", "2: Lore", "3: Checks", "4: Peek Solution", "5: Console"]
        self.console_lines = [
            "Welcome to the vc Stage Runner TUI.",
            "Select a stage on the left and press [t] to test, or [s] to submit.",
            "Use [1-5] or [h]/[l] to switch inspector tabs.",
        ]
        self.is_running = False
        self.status_message = "Ready. [t] Test · [s] Submit · [Tab] Switch Track · [q] Quit"
        self.right_scroll = 0
        self.gpu_str = get_gpu_info()

        # Find initial active stage
        for i, stg in enumerate(self.ladder):
            if not is_completed(stg, self.progress):
                self.selected_idx = i
                break

    def init_colors(self):
        if not curses.has_colors():
            return
        try:
            curses.start_color()
        except Exception:
            pass
        try:
            curses.use_default_colors()
            bg = -1
        except Exception:
            bg = curses.COLOR_BLACK

        pairs = [
            (1, curses.COLOR_CYAN, bg),
            (2, curses.COLOR_GREEN, bg),
            (3, curses.COLOR_YELLOW, bg),
            (4, curses.COLOR_RED, bg),
            (5, curses.COLOR_MAGENTA, bg),
            (6, curses.COLOR_BLACK, curses.COLOR_CYAN),
            (7, curses.COLOR_WHITE, bg),
        ]
        for pair_id, fg, b in pairs:
            try:
                curses.init_pair(pair_id, fg, b)
            except Exception:
                try:
                    curses.init_pair(pair_id, fg, curses.COLOR_BLACK)
                except Exception:
                    pass

    def draw_box(self, y, x, h, w, title=""):
        """Draw a sleek box with optional title."""
        try:
            # Horizontal borders
            self.stdscr.addstr(y, x, "╭" + "─" * (w - 2) + "╮", curses.color_pair(1))
            for row in range(y + 1, y + h - 1):
                self.stdscr.addstr(row, x, "│", curses.color_pair(1))
                self.stdscr.addstr(row, x + w - 1, "│", curses.color_pair(1))
            self.stdscr.addstr(y + h - 1, x, "╰" + "─" * (w - 2) + "╯", curses.color_pair(1))
            if title and len(title) < w - 4:
                self.stdscr.addstr(y, x + 2, f" {title} ", curses.color_pair(1) | curses.A_BOLD)
        except curses.error:
            pass

    def run_stage_test(self, stage):
        """Run pytest for the stage and update console."""
        self.is_running = True
        self.status_message = f"Testing stage {stage_label(stage)}... please wait"
        self.active_tab = 4  # Switch to console
        self.console_lines.append(f"\n--- Testing Stage {stage_label(stage)}: {stage_name(stage, self.progress)} ---")

        def worker():
            code, passed, failed, output = run_checks(stage, self.progress)
            self.console_lines.extend(output.splitlines())
            if code == 0:
                self.console_lines.append(f"\n>>> SUCCESS: ALL {passed} CHECKS PASSED! <<<")
                self.status_message = f"Stage {stage_label(stage)} passed ({passed} checks). Ready to submit [s]!"
            else:
                self.console_lines.append(f"\n>>> FAILED: {failed} checks failed. <<<")
                self.status_message = f"Stage {stage_label(stage)} failed. Fix errors and retry [t]."
            self.is_running = False

        threading.Thread(target=worker, daemon=True).start()

    def submit_stage(self, stage):
        """Submit the current stage."""
        self.is_running = True
        self.status_message = f"Verifying and submitting stage {stage_label(stage)}..."
        self.active_tab = 4

        def worker():
            code, passed, failed, output = run_checks(stage, self.progress)
            self.console_lines.extend(output.splitlines())
            if code == 0:
                commit_stage(stage, self.progress)
                if stage["id"] not in self.progress["completed"]:
                    self.progress["completed"].append(stage["id"])
                    save_progress(self.progress)
                self.console_lines.append(f"\n>>> BANKED: Stage {stage_label(stage)} committed and submitted! <<<")
                self.status_message = f"Stage {stage_label(stage)} complete! Next stage unlocked."
                # Advance to next incomplete stage
                for i, stg in enumerate(self.ladder):
                    if not is_completed(stg, self.progress):
                        self.selected_idx = i
                        break
            else:
                self.console_lines.append(f"\n>>> SUBMISSION REJECTED: {failed} checks failed. <<<")
                self.status_message = f"Submission failed for stage {stage_label(stage)}."
            self.is_running = False

        threading.Thread(target=worker, daemon=True).start()

    def render(self):
        self.stdscr.clear()
        max_y, max_x = self.stdscr.getmaxyx()

        if max_y < 24 or max_x < 80:
            msg = "Terminal window too small (minimum 80x24). Please resize."
            self.stdscr.addstr(max_y // 2, max(0, (max_x - len(msg)) // 2), msg, curses.color_pair(4))
            self.stdscr.refresh()
            return

        # 1. Header Banner
        header_h = 3
        done = len(self.progress["completed"])
        total = len(self.ladder)
        pct = int((done / total) * 100) if total else 0
        pbar_len = 16
        filled = int(pbar_len * (done / total)) if total else 0
        pbar = "█" * filled + "░" * (pbar_len - filled)

        self.draw_box(0, 0, header_h, max_x, title="⚡ vLLM-FROM-SCRATCH STAGE RUNNER ⚡")
        header_text = (
            f" Track: [{self.backend.upper()}]  ·  Progress: [{pbar}] {pct}% ({done}/{total})  ·  GPU: {self.gpu_str}"
        )
        self.stdscr.addstr(1, 2, header_text[: max_x - 4], curses.color_pair(1) | curses.A_BOLD)

        # 2. Main Layout Split
        footer_h = 2
        body_y = header_h
        body_h = max_y - header_h - footer_h
        left_w = min(42, max_x // 3)
        right_w = max_x - left_w

        # Left Box: The Ladder
        self.draw_box(body_y, 0, body_h, left_w, title="The Ladder (Stages)")

        # Populate left list
        row_limit = body_h - 2
        scroll_start = max(0, self.selected_idx - row_limit // 2)
        visible_ladder = self.ladder[scroll_start : scroll_start + row_limit]

        current_arc = None
        curr_y = body_y + 1
        for idx_offset, stage in enumerate(visible_ladder):
            abs_idx = scroll_start + idx_offset
            if curr_y >= body_y + body_h - 1:
                break

            # Arc header line if arc changed
            if stage["arc_id"] != current_arc and curr_y < body_y + body_h - 2:
                current_arc = stage["arc_id"]
                arc_label = f"[{stage['arc_id']}] {stage['arc']}"
                self.stdscr.addstr(curr_y, 2, arc_label[: left_w - 4], curses.color_pair(5) | curses.A_BOLD)
                curr_y += 1
                if curr_y >= body_y + body_h - 1:
                    break

            completed = is_completed(stage, self.progress)
            is_active = not completed and all(
                is_completed(self.ladder[j], self.progress) for j in range(abs_idx)
            )

            icon = "✓" if completed else ("▶" if is_active else "·")
            icon_color = (
                curses.color_pair(2)
                if completed
                else (curses.color_pair(1) if is_active else curses.color_pair(7))
            )

            label = stage_label(stage)
            name = stage_name(stage, self.progress)
            line = f" {icon} {label:>3} {name}"
            truncated = line[: left_w - 4]

            if abs_idx == self.selected_idx:
                self.stdscr.addstr(curr_y, 2, f"{truncated:<{left_w-4}}", curses.color_pair(6) | curses.A_BOLD)
            else:
                self.stdscr.addstr(curr_y, 2, f" {icon} ", icon_color | curses.A_BOLD)
                self.stdscr.addstr(curr_y, 5, f"{label:>3} {name}"[: left_w - 7], curses.color_pair(7))
            curr_y += 1

        # Right Box: Dynamic Inspector
        cur_stage = self.ladder[self.selected_idx]
        tab_bar = "  " + "   ".join(
            f"[{t}]" if i == self.active_tab else f" {t} " for i, t in enumerate(self.tabs)
        )
        self.draw_box(body_y, left_w, body_h, right_w, title=f"Stage {stage_label(cur_stage)}")
        self.stdscr.addstr(body_y, left_w + 2, tab_bar[: right_w - 4], curses.color_pair(3) | curses.A_BOLD)

        # Right Pane Content based on active tab
        content_y = body_y + 2
        content_h = body_h - 3
        inner_w = right_w - 4

        if self.active_tab == 0:  # Guide
            lines = [
                f"NAME:       {stage_name(cur_stage, self.progress)}",
                f"ARC:        {cur_stage['arc_id']} - {cur_stage['arc']}",
                f"DIFFICULTY: {'★' * cur_stage['difficulty']}{'☆' * (5 - cur_stage['difficulty'])}",
                f"FILES:      {', '.join(stage_files(cur_stage, self.progress))}",
                "",
                "THE INSIGHT:",
                f"  {cur_stage.get('insight', '').strip()}",
                "",
                "DELIVER:",
                f"  {cur_stage.get('deliver', '').strip()}",
                "",
                "MEASURE:",
                f"  {cur_stage.get('measure', '').strip()}",
            ]
            terms = terms_for_stage(
                stage_label(cur_stage), [stage_label(item) for item in self.ladder]
            )
            if terms:
                lines.append("")
                lines.append("NEW TERMS INTRODUCED:")
                for t in terms:
                    lines.append(f"  • {t.name}: {t.definition}")

            for i, line in enumerate(lines[:content_h]):
                self.stdscr.addstr(content_y + i, left_w + 2, line[:inner_w], curses.color_pair(7))

        elif self.active_tab == 1:  # Lore
            insight = cur_stage.get("insight", "No lore recorded for this stage.")
            lore_lines = [
                f"LORE INSIGHT: Stage {stage_label(cur_stage)}",
                "=" * len(f"LORE INSIGHT: Stage {stage_label(cur_stage)}"),
                "",
            ]
            lore_lines.extend(insight.splitlines() or [insight])
            lore_lines.append("")
            lore_lines.append("See LORE.md in the repo root for the complete mathematical derivation.")
            for i, line in enumerate(lore_lines[:content_h]):
                self.stdscr.addstr(content_y + i, left_w + 2, line[:inner_w], curses.color_pair(7))

        elif self.active_tab == 2:  # Checks
            checks = checks_for(cur_stage, self.progress)
            check_lines = [
                f"TEST CHECKS ({len(checks)} total):",
                "------------------------------",
            ]
            for name, desc in checks:
                check_lines.append(f"• {name}")
                if desc:
                    check_lines.append(f"    {desc[:inner_w-6]}")
            for i, line in enumerate(check_lines[:content_h]):
                self.stdscr.addstr(content_y + i, left_w + 2, line[:inner_w], curses.color_pair(7))

        elif self.active_tab == 3:  # Solution Peek
            target_path = Path(".solutions") / f"s{cur_stage['id'].replace('-', '_').split('_')[0]}_{cur_stage['id'].split('-')[-1]}.py"
            # Fallback searching .solutions
            sol_content = "Loading reference solution from solutions branch..."
            try:
                out = subprocess.run(
                    ["git", "show", f"solutions:.solutions/{cur_stage['file'].split('/')[-1]}"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                if out.returncode == 0:
                    sol_content = out.stdout
                else:
                    sol_content = "Solution file not available or not yet fetched."
            except Exception as e:
                sol_content = f"Error reading solution: {e}"

            sol_lines = sol_content.splitlines()
            for i, line in enumerate(sol_lines[:content_h]):
                self.stdscr.addstr(content_y + i, left_w + 2, line[:inner_w], curses.color_pair(7))

        elif self.active_tab == 4:  # Console
            visible_console = self.console_lines[-content_h:]
            for i, line in enumerate(visible_console):
                color = curses.color_pair(7)
                if "SUCCESS" in line or "ALL" in line:
                    color = curses.color_pair(2) | curses.A_BOLD
                elif "FAILED" in line or "ERROR" in line:
                    color = curses.color_pair(4) | curses.A_BOLD
                elif line.startswith("---"):
                    color = curses.color_pair(1) | curses.A_BOLD
                self.stdscr.addstr(content_y + i, left_w + 2, line[:inner_w], color)

        # 3. Footer Bar
        try:
            self.stdscr.addstr(max_y - 2, 2, self.status_message[: max_x - 4], curses.color_pair(3) | curses.A_BOLD)
            help_bar = " [↑/↓/j/k] Select · [t] Test · [s] Submit · [1-5] Tab · [Tab] Backend · [q] Quit"
            self.stdscr.addstr(max_y - 1, 0, help_bar[: max_x - 1], curses.color_pair(6))
        except curses.error:
            pass

        self.stdscr.refresh()

    def run(self):
        self.init_colors()
        try:
            curses.curs_set(0)
        except Exception:
            pass
        self.stdscr.timeout(100)  # non-blocking for responsive UI/threads

        while True:
            self.render()
            try:
                ch = self.stdscr.getch()
            except curses.error:
                continue

            if ch == -1:
                continue

            if ch in (ord("q"), ord("Q")):
                break
            elif ch in (curses.KEY_UP, ord("k")):
                if self.selected_idx > 0:
                    self.selected_idx -= 1
            elif ch in (curses.KEY_DOWN, ord("j")):
                if self.selected_idx < len(self.ladder) - 1:
                    self.selected_idx += 1
            elif ch in (ord("\t"),):  # Tab toggles backend
                new_backend = "jax" if self.backend == "torch" else "torch"
                self.backend = new_backend
                self.progress["backend"] = new_backend
                self.data, self.ladder = load_stages(new_backend)
                self.selected_idx = min(self.selected_idx, len(self.ladder) - 1)
                self.status_message = f"Switched to {new_backend.upper()} track."
            elif ch in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                self.active_tab = ch - ord("1")
            elif ch in (ord("h"), curses.KEY_LEFT):
                self.active_tab = max(0, self.active_tab - 1)
            elif ch in (ord("l"), curses.KEY_RIGHT):
                self.active_tab = min(len(self.tabs) - 1, self.active_tab + 1)
            elif ch in (ord("t"), ord("T")):
                if not self.is_running:
                    self.run_stage_test(self.ladder[self.selected_idx])
            elif ch in (ord("s"), ord("S")):
                if not self.is_running:
                    self.submit_stage(self.ladder[self.selected_idx])


def main():
    try:
        curses.wrapper(lambda stdscr: VcTUI(stdscr).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
