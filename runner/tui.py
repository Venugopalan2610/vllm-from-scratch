"""Terminal User Interface (TUI) for vc - the stage runner.

Run with:
  ./vc tui
  ./vc ui
"""

import curses
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.cli import (
    BACKENDS,
    checks_for,
    commit_stage,
    current_stage,
    load_progress,
    load_stages,
    progress_counts,
    run_checks,
    save_progress,
    set_backend,
    stage_files,
    stage_label,
    stage_name,
)
from runner import pictures
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
        # 0: Guide, 1: Picture, 2: Lore, 3: Checks, 4: Console, 5: Solution
        self.active_tab = 0
        self.tabs = ["1:Guide", "2:Picture", "3:Lore", "4:Checks", "5:Console", "6:Peek"]
        self.pictures = pictures.load()
        self._solutions = {}
        self.console_lines = [
            "Welcome to the vc Stage Runner TUI.",
            "Select a stage on the left and press [e] to edit code in Vim.",
            "Press [t] to test, or [s] to submit and advance.",
            "Use [1-6] or [h]/[l] to switch inspector tabs. [p] shows the picture.",
            "[PgUp]/[PgDn] or [u]/[d] scroll a long tab.",
        ]
        self.is_running = False
        self.status_message = "Ready. [e] Edit (Vim) · [t] Test · [s] Submit · [Tab] Track · [q] Quit"
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
        self.console_lines.append("")
        self.console_lines.append(f"--- Testing Stage {stage_label(stage)}: {stage_name(stage, self.progress)} ---")

        def worker():
            code, passed, failed, output = run_checks(stage, self.progress)
            self.console_lines.extend(output.splitlines())
            if code == 0:
                self.console_lines.append("")
                self.console_lines.append(f">>> SUCCESS: ALL {passed} CHECKS PASSED! <<<")
                self.status_message = f"Stage {stage_label(stage)} passed ({passed} checks). Ready to submit [s]!"
            else:
                self.console_lines.append("")
                self.console_lines.append(f">>> FAILED: {failed} checks failed. <<<")
                self.status_message = f"Stage {stage_label(stage)} failed. Fix errors and retry [t]."
            self.is_running = False

        threading.Thread(target=worker, daemon=True).start()

    def submit_stage(self, stage):
        """Submit the current stage. A later stage cannot be banked before the
        stages under it, and a completed stage is already banked."""
        current = current_stage(self.ladder, self.progress)
        if current is None or stage["id"] != current["id"]:
            where = f"stage {stage_label(current)}" if current else "nothing"
            self.status_message = (f"Only the current stage can be submitted: "
                                   f"{where}. [t] tests any stage.")
            return
        self.is_running = True
        self.status_message = f"Verifying and submitting stage {stage_label(stage)}..."
        self.active_tab = 4  # Switch to console

        def worker():
            code, passed, failed, output = run_checks(stage, self.progress)
            self.console_lines.extend(output.splitlines())
            if code == 0:
                commit_stage(stage, self.progress)
                if stage["id"] not in self.progress["completed"]:
                    self.progress["completed"].append(stage["id"])
                    save_progress(self.progress)
                self.console_lines.append("")
                self.console_lines.append(f">>> BANKED: Stage {stage_label(stage)} committed and submitted! <<<")
                self.status_message = f"Stage {stage_label(stage)} complete! Next stage unlocked."
                # Advance to next incomplete stage
                for i, stg in enumerate(self.ladder):
                    if not is_completed(stg, self.progress):
                        self.selected_idx = i
                        break
            else:
                self.console_lines.append("")
                self.console_lines.append(f">>> SUBMISSION REJECTED: {failed} checks failed. <<<")
                self.status_message = f"Submission failed for stage {stage_label(stage)}."
            self.is_running = False

        threading.Thread(target=worker, daemon=True).start()

    def edit_stage(self, stage):
        """Open stage file(s) directly in Vim / $EDITOR with seamless curses resumption."""
        files = stage_files(stage, self.progress)
        if not files:
            self.status_message = "No editable files found for this stage."
            return

        target_files = files
        if len(files) > 1:
            f1_name = Path(files[0]).name
            f2_name = Path(files[1]).name
            self.status_message = f"Edit which file? [1] {f1_name} (Kernel)  [2] {f2_name} (Wrapper)  [b] Both  [Esc] Cancel"
            self.render()
            while True:
                try:
                    c = self.stdscr.getch()
                except curses.error:
                    continue
                if c == -1:
                    time.sleep(0.02)
                    continue
                if c in (ord("1"), ord("\n"), 10, 13, curses.KEY_ENTER):
                    target_files = [files[0]]
                    break
                elif c == ord("2"):
                    target_files = [files[1]]
                    break
                elif c in (ord("b"), ord("B")):
                    target_files = files
                    break
                elif c in (27, ord("q"), ord("Q")):
                    self.status_message = "Edit cancelled."
                    return

        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if not editor:
            for candidate in ["nvim", "vim", "nano", "vi"]:
                if shutil.which(candidate):
                    editor = candidate
                    break
        if not editor:
            editor = "vim"

        file_paths = [str(ROOT / f) for f in target_files]
        cmd = [editor]
        # Open side-by-side vertical splits in vim if multiple files
        if len(file_paths) > 1 and ("vim" in editor or "nvim" in editor):
            cmd.append("-O")
        cmd.extend(file_paths)

        curses.def_prog_mode()
        curses.endwin()
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.console_lines.append(f"Error launching editor {editor}: {e}")
        finally:
            curses.reset_prog_mode()
            try:
                curses.curs_set(0)
            except Exception:
                pass
            self.init_colors()
            self.stdscr.erase()
            self.active_tab = 4  # show console
            files_str = ", ".join(target_files)
            self.console_lines.append("")
            self.console_lines.append(f">>> Returned from {editor} editing {files_str}. Press [t] to test! <<<")
            self.status_message = f"Saved {files_str}. Press [t] to test, or [s] to submit!"

    def picture_lines(self, stage, width):
        """The ASCII picture of the stage, or a note that it has none."""
        picture = pictures.for_stage(stage["id"], self.backend, self.pictures)
        if picture is None:
            return [("No picture for this stage in docs/PICTURES.md.", curses.color_pair(3))]
        lines = [(wl, curses.color_pair(3) | curses.A_BOLD)
                 for wl in textwrap.wrap(picture.caption, width)]
        lines.append(("", 0))
        art = picture.art.splitlines()
        if max((len(line) for line in art), default=0) > width:
            lines.append((f"(widen the terminal to {max(len(l) for l in art) + 50}"
                          " columns for the whole picture)", curses.color_pair(7)))
        lines += [(line, curses.color_pair(1)) for line in art]
        return lines

    def solution(self, stage):
        """The reference files of the stage on this track, cached. It reads
        only what is already here: a fetch would print over the screen."""
        key = (stage["id"], self.backend)
        if key not in self._solutions:
            parts = []
            for path in stage_files(stage, self.progress):
                name = Path(path).name
                local = ROOT / ".solutions" / name
                text = local.read_text() if local.exists() else None
                for ref in ("solutions", "origin/solutions"):
                    if text is None:
                        out = subprocess.run(["git", "show", f"{ref}:.solutions/{name}"],
                                             cwd=ROOT, capture_output=True, text=True)
                        text = out.stdout if out.returncode == 0 else None
                parts.append(f"--- {path} ---")
                parts += (text or "Not here yet. Run in a shell: ./vc peek").splitlines()
                parts.append("")
            self._solutions[key] = parts
        return self._solutions[key]

    def tab_lines(self, stage, inner_w):
        """-> [(text, curses attr)] for the active tab, before the scroll."""
        plain, head = curses.color_pair(7), curses.color_pair(3) | curses.A_BOLD

        def para(text, indent="  "):
            return [(indent + wl, plain)
                    for wl in textwrap.wrap(" ".join(str(text).split()),
                                            width=inner_w - len(indent))]

        if self.active_tab == 0:  # Guide
            lines = [(f"NAME:       {stage_name(stage, self.progress)}", plain),
                     (f"ARC:        {stage['arc_id']} - {stage['arc']}", plain),
                     (f"DIFFICULTY: {'★' * stage['difficulty']}"
                      f"{'☆' * (5 - stage['difficulty'])}", plain),
                     (f"FILES:      {', '.join(stage_files(stage, self.progress))}", plain),
                     (f"YOU ARE HERE: {stage.get('layer', '')}", plain),
                     ("", 0), ("THE PROBLEM THAT ASKS FOR THIS STAGE", head)]
            lines += para(stage.get("because", ""))
            lines += [("", 0), ("WHAT IS GIVEN", head)]
            for item in stage.get("given", []):
                for index, wl in enumerate(textwrap.wrap(item, width=inner_w - 6)):
                    lines.append((("  - " if index == 0 else "    ") + wl, plain))
            deliver = (stage.get("jax_deliver", stage["deliver"])
                       if self.backend == "jax" else stage["deliver"])
            lines += [("", 0), ("WHAT IS ASKED", head)] + para(deliver)
            lines += [("", 0), ("WHY THIS STAGE EXISTS", head)] + para(stage.get("insight", ""))
            if self.backend == "jax" and stage.get("jax_insight"):
                lines += [("", 0), ("WHAT CHANGES IN JAX", head)] + para(stage["jax_insight"])
            lines += [("", 0), ("THE PICTURE", head)] + self.picture_lines(stage, inner_w)
            lines += [("", 0), ("HOW YOU WILL KNOW IT WORKED", head)]
            lines += para(stage.get("measure", ""))
            terms = terms_for_stage(stage_label(stage), [stage_label(s) for s in self.ladder])
            if terms:
                lines += [("", 0), ("NEW WORDS IN THIS STAGE", head)]
                for term in terms:
                    lines += para(f"• {term.name}: {term.definition}")
            return lines

        if self.active_tab == 1:  # Picture
            return ([(f"Stage {stage_label(stage)}  {stage_name(stage, self.progress)}", head),
                     ("", 0)] + self.picture_lines(stage, inner_w))

        if self.active_tab == 2:  # Lore
            lines = [(f"THE INSIGHT OF STAGE {stage_label(stage)}", head), ("", 0)]
            lines += para(stage.get("insight", "No insight recorded."), "")
            if self.backend == "jax" and stage.get("jax_insight"):
                lines += [("", 0), ("IN JAX", head)] + para(stage["jax_insight"], "")
            lines += [("", 0), ("LORE.md in the repo root has the long form, "
                                "with all the arithmetic.", curses.color_pair(1))]
            return lines

        if self.active_tab == 3:  # Checks
            checks = checks_for(stage, self.progress)
            lines = [(f"THE CHECKS ({len(checks)})", head), ("", 0)]
            for name, desc in checks:
                lines.append(("[ ] " + name[5:].replace("_", " "), plain))
                lines += [(line, curses.color_pair(1)) for line in para(desc, "    ")]
            return lines

        if self.active_tab == 4:  # Console
            lines = []
            for item in self.console_lines:
                for line in str(item).splitlines():
                    attr = plain
                    if "SUCCESS" in line or "BANKED" in line or " passed" in line:
                        attr = curses.color_pair(2) | curses.A_BOLD
                    elif "FAILED" in line or "ERROR" in line or "REJECTED" in line:
                        attr = curses.color_pair(4) | curses.A_BOLD
                    elif line.startswith("---") or line.startswith(">>>"):
                        attr = curses.color_pair(1) | curses.A_BOLD
                    lines.append((line, attr))
            return lines

        return [(line, plain) for line in self.solution(stage)]  # Peek

    def render(self):
        self.stdscr.erase()
        max_y, max_x = self.stdscr.getmaxyx()

        if max_y < 24 or max_x < 80:
            msg = "Terminal window too small (minimum 80x24). Please resize."
            self.stdscr.addstr(max_y // 2, max(0, (max_x - len(msg)) // 2), msg, curses.color_pair(4))
            self.stdscr.refresh()
            return

        # 1. Header Banner
        header_h = 3
        done, total = progress_counts(self.ladder, self.progress)[:2]
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
        lines = self.tab_lines(cur_stage, inner_w)   # [(text, curses attr)]

        if self.active_tab == 4:      # the console follows its newest line
            visible = lines[-content_h:]
        else:
            last_top = max(0, len(lines) - content_h)
            self.right_scroll = max(0, min(self.right_scroll, last_top))
            visible = lines[self.right_scroll:self.right_scroll + content_h]
            if len(lines) > content_h:
                more = (f" {self.right_scroll + 1}-{self.right_scroll + len(visible)}"
                        f" of {len(lines)}  [u/d] ")
                try:
                    self.stdscr.addstr(body_y + body_h - 1, left_w + right_w - len(more) - 2,
                                       more, curses.color_pair(3))
                except curses.error:
                    pass
        for i, (line, attr) in enumerate(visible):
            try:
                self.stdscr.addstr(content_y + i, left_w + 2,
                                   line.rstrip("\r\n")[:inner_w], attr)
            except curses.error:
                pass

        # 3. Footer Bar
        try:
            self.stdscr.addstr(max_y - 2, 2, self.status_message[: max_x - 4], curses.color_pair(3) | curses.A_BOLD)
            help_bar = " [↑/↓/j/k] Select · [e] Edit (Vim) · [t] Test · [s] Submit · [1-6] Tab · [p] Picture · [u/d] Scroll · [Tab] Track · [q] Quit"
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
        self.stdscr.timeout(50)  # 50ms polling for responsive inputs and thread state

        dirty = True
        prev_running = False

        while True:
            if dirty or self.is_running or (prev_running != self.is_running):
                self.render()
                dirty = False
                prev_running = self.is_running

            try:
                ch = self.stdscr.getch()
            except curses.error:
                continue

            if ch == -1:
                continue

            dirty = True

            if ch in (ord("q"), ord("Q")):
                break
            elif ch in (curses.KEY_UP, ord("k")):
                if self.selected_idx > 0:
                    self.selected_idx -= 1
                    self.right_scroll = 0
            elif ch in (curses.KEY_DOWN, ord("j")):
                if self.selected_idx < len(self.ladder) - 1:
                    self.selected_idx += 1
                    self.right_scroll = 0
            elif ch in (curses.KEY_NPAGE, ord("d"), ord(" ")):
                self.right_scroll += max(1, self.stdscr.getmaxyx()[0] // 2)
            elif ch in (curses.KEY_PPAGE, ord("u")):
                self.right_scroll = max(0, self.right_scroll - max(1, self.stdscr.getmaxyx()[0] // 2))
            elif ch in (ord("p"), ord("P")):
                self.active_tab = 1
                self.right_scroll = 0
            elif ch in (ord("e"), ord("E")):
                if not self.is_running:
                    self.edit_stage(self.ladder[self.selected_idx])
                    dirty = True
            elif ch in (ord("\t"),):  # Tab toggles backend
                new_backend = "jax" if self.backend == "torch" else "torch"
                self.backend = new_backend
                # set_backend also swaps progress["completed"] to the list of
                # the new track. Without it, a submit banks the wrong track.
                set_backend(self.progress, new_backend)
                self.data, self.ladder = load_stages(new_backend)
                self.selected_idx = min(self.selected_idx, len(self.ladder) - 1)
                self.right_scroll = 0
                self.status_message = f"Switched to {new_backend.upper()} track."
            elif ord("1") <= ch < ord("1") + len(self.tabs):
                self.active_tab = ch - ord("1")
                self.right_scroll = 0
            elif ch in (ord("h"), curses.KEY_LEFT):
                self.active_tab = max(0, self.active_tab - 1)
                self.right_scroll = 0
            elif ch in (ord("l"), curses.KEY_RIGHT):
                self.active_tab = min(len(self.tabs) - 1, self.active_tab + 1)
                self.right_scroll = 0
            elif ch in (ord("t"), ord("T")):
                if not self.is_running:
                    self.run_stage_test(self.ladder[self.selected_idx])
            elif ch in (ord("s"), ord("S")):
                if not self.is_running:
                    self.submit_stage(self.ladder[self.selected_idx])
            elif ch == curses.KEY_RESIZE:
                dirty = True


def main():
    try:
        curses.wrapper(lambda stdscr: VcTUI(stdscr).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
