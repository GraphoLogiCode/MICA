"""The after-session labeling dashboard — a native desktop window (Tkinter).

    python after_session_tool/gui.py

Labeling is the only manual step left in the pipeline: the rig automation processes
every session (evidence, scan, report, readiness) on its own and leaves just the
builder's goal/subtype label. This window turns that step into a click — a list of
sessions needing labels, each with its chain state, preview evidence, and the two
required inputs — and a History tab to review/fix the whole batch before a cascade.

All data + actions live in `sessions.py`; this is just the front-end. Tkinter ships
with Python, so there are no dependencies. It only reads the pipeline's files and, on
a label, shells out to the canonical scripts/after_game.py.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import sessions                                          # noqa: E402

_STATE_TEXT = {
    "ready": "ready to label", "labeling": "labeling…", "processing": "processing…",
    "unprocessed": "not processed", "labeled": "labeled",
    "quarantined": "quarantined", "gate failed": "gate failed",
}
_STATE_COLOR = {
    "ready": "#1f7a3d", "labeling": "#8a6d10", "processing": "#5a6570",
    "unprocessed": "#6b5583", "labeled": "#245b9c",
    "quarantined": "#a12b2b", "gate failed": "#a12b2b",
}
_THUMB_WIDTH = 360


class ScrollFrame(ttk.Frame):
    """A vertically scrollable frame — render into `.inner`. Detail panes can grow
    taller than the window (two pictures + verdict + form), so they scroll."""

    def __init__(self, master):
        super().__init__(master)
        canvas = tk.Canvas(self, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(canvas, padding=10)
        window = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>",
                        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(window, width=e.width))
        # Only the hovered pane scrolls (bind_all would fight between two panes).
        canvas.bind("<Enter>", lambda e: canvas.bind_all(
            "<MouseWheel>", lambda ev: canvas.yview_scroll(int(-ev.delta / 120), "units")))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("MICA — After-Session Labeling")
        root.geometry("1080x720")
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        # Grey chips for the roadmap styles the matcher's templates don't cover yet.
        style.configure("Roadmap.TButton", foreground="#7a8591")
        self._sel: str | None = None                    # inbox selection
        self._hist_sel: str | None = None               # history selection
        self._pending_snap: dict | None = None          # worker -> UI-thread handoff
        self._refreshing = False
        self._refresh_again = False
        self.queue: list[dict] = []                     # staged batch labels

        nb = ttk.Notebook(root)
        self.nb = nb
        nb.pack(fill="both", expand=True)
        self.tab_inbox = ttk.Frame(nb)
        self.tab_status = ttk.Frame(nb)
        self.tab_history = ttk.Frame(nb)
        nb.add(self.tab_inbox, text="Inbox")
        nb.add(self.tab_status, text="Status")
        nb.add(self.tab_history, text="History")
        nb.bind("<<NotebookTabChanged>>", lambda e: self.refresh_all())

        # A bottom activity bar for background jobs. There is no honest percentage
        # (the matcher's model-load + match doesn't report a fraction), so this is
        # an INDETERMINATE bar + a real elapsed timer — it says "working, and for
        # how long", not a made-up number.
        self.statusbar = ttk.Frame(root, padding=(8, 3))
        self.statusbar.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="")
        ttk.Label(self.statusbar, textvariable=self.status_var,
                  foreground="#8a6d10").pack(side="left")
        self.bar = ttk.Progressbar(self.statusbar, mode="indeterminate", length=160)
        self._bar_running = False
        self._had_running = False

        self._build_inbox()
        self._build_status()
        self._build_history()
        self.refresh_all()
        self._poll_jobs()
        self._check_interpreter()

    def _check_interpreter(self) -> None:
        # Every job this window starts runs on THIS python (sys.executable). A
        # machine with several pythons (a fresh 3.14 became the `py` default and
        # broke evidence jobs with "No module named timm") makes that a silent
        # trap — so say it loudly at startup instead of failing one job at a time.
        import importlib.util
        missing = [name for name in ("torch", "timm") if importlib.util.find_spec(name) is None]
        if missing:
            messagebox.showwarning(
                "Wrong Python for the pipeline",
                f"This window is running on {sys.executable}\n\n"
                f"which lacks: {', '.join(missing)}.\n\n"
                "Evidence and label jobs WILL fail on it. Close this window and "
                "launch with the ML Python instead:\n\n"
                "    py -3.12 after_session_tool\\gui.py")

    # ----------------------------------------------------------------- inbox

    def _build_inbox(self) -> None:
        pane = ttk.Panedwindow(self.tab_inbox, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=6, pady=6)
        left = ttk.Frame(pane)
        bar = ttk.Frame(left)
        bar.pack(fill="x")
        ttk.Label(bar, text="Sessions needing a label").pack(side="left")
        ttk.Button(bar, text="Refresh", command=self.refresh_all).pack(side="right")
        self.tree = ttk.Treeview(left, columns=("state",), show="tree headings",
                                 height=24, selectmode="browse")
        self.tree.heading("#0", text="session")
        self.tree.heading("state", text="state")
        self.tree.column("#0", width=210)
        self.tree.column("state", width=110, anchor="center")
        for state, color in _STATE_COLOR.items():
            self.tree.tag_configure(state, foreground=color)
        self.tree.pack(fill="both", expand=True, pady=(4, 0))
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # The batch queue: stage several labels here, then run the matcher ONCE
        # for all of them — one model load instead of one per session.
        box = ttk.LabelFrame(left, text="Batch queue — one matcher run for all", padding=4)
        box.pack(fill="x", pady=(8, 0))
        self.queue_tree = ttk.Treeview(box, columns=("label",), show="tree headings",
                                       height=4, selectmode="browse")
        self.queue_tree.heading("#0", text="session")
        self.queue_tree.heading("label", text="staged label")
        self.queue_tree.column("#0", width=190)
        self.queue_tree.column("label", width=140)
        self.queue_tree.pack(fill="x")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(4, 0))
        self.queue_vlm = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="VLM cross-check", variable=self.queue_vlm).pack(side="left")
        ttk.Button(row, text="Remove", command=self._queue_remove).pack(side="left", padx=6)
        self.queue_btn = ttk.Button(row, text="Label all (0)", command=self._queue_run)
        self.queue_btn.pack(side="left")

        pane.add(left, weight=1)
        self.inbox_detail = ScrollFrame(pane)
        pane.add(self.inbox_detail, weight=2)

    def refresh_inbox(self, cards: list[dict]) -> None:
        keep = self._sel
        self.tree.delete(*self.tree.get_children())
        for c in cards:
            self.tree.insert("", "end", iid=c["sid"], text=c["sid"],
                             values=(_STATE_TEXT.get(c["state"], c["state"]),),
                             tags=(c["state"],))
        if keep and self.tree.exists(keep):
            self.tree.selection_set(keep)
        elif cards:
            self.tree.selection_set(cards[0]["sid"])
        else:
            self._render_detail(None, self.inbox_detail.inner)

    def _on_select(self, _event=None) -> None:
        sel = self.tree.selection()
        self._sel = sel[0] if sel else None
        self._render_detail(self._sel, self.inbox_detail.inner)

    # ---------------------------------------------------------- shared detail

    def _render_detail(self, sid, target=None, allow_relabel: bool = False) -> None:
        if target is None:
            target = self.inbox_detail.inner            # default pane
        for child in target.winfo_children():
            child.destroy()
        # Per-pane image refs (so the two detail panes never GC each other's
        # pictures); self._images aliases the pane being rendered right now.
        target._imgs = []
        self._images = target._imgs
        if sid is None:
            ttk.Label(target, text="Select a session.", foreground="#7a8591").pack(anchor="w")
            return
        view = sessions.session_view(sid)
        state, verdict = view["state"], view.get("verdict")

        head = ttk.Frame(target)
        head.pack(fill="x")
        ttk.Label(head, text=sid, font=("Consolas", 12, "bold")).pack(side="left")
        ttk.Label(head, text="  " + _STATE_TEXT.get(state, state),
                  foreground=_STATE_COLOR.get(state, "#5a6570")).pack(side="left")

        summary, report = view.get("summary") or {}, view.get("report") or {}
        facts = []
        if summary.get("records"):
            facts.append(f'{summary["records"].get("scored", "?")} scored records · '
                         f'{summary.get("built_cells", "?")} built cells')
        if summary.get("scanned_of_built") is not None and summary.get("built_cells"):
            facts.append(f'agent scan covered {summary["scanned_of_built"]}/{summary["built_cells"]}')
        if report.get("gate_file_checks"):
            facts.append(f'B0 gate: {report["gate_file_checks"]}')
        if facts:
            ttk.Label(target, text="   ·   ".join(facts),
                      foreground="#4a5560").pack(anchor="w", pady=(6, 0))

        self._render_build(sid, view["pngs"], target)
        self._render_declare(sid, view.get("declared"), target)
        self._render_verdict(verdict, target)
        self._render_job(view.get("job"), target)
        self._render_form(sid, state, view.get("label") or {}, target, allow_relabel, verdict)

    def _load_image(self, path: str, target_w: int, target):
        """A PNG scaled down to ~target_w px (integer subsample — Tk has no smooth
        resize without extra deps; fine for a preview). Ref kept on the pane."""
        try:
            img = tk.PhotoImage(file=path)
        except tk.TclError:
            return None
        factor = max(1, (img.width() + target_w - 1) // target_w)
        if factor > 1:
            img = img.subsample(factor)
        target._imgs.append(img)
        return img

    def _render_build(self, sid: str, pngs: list[str], target) -> None:
        """What was built — the actual in-game screenshot (works even before a session
        is processed) next to the voxel render of the structure. Both open full-size on
        click. This is the picture the human checks to pick goal + subtype."""
        pov = sessions.pov_frame(sid)
        clouds = sessions.shot_path(sid, "clouds.png")
        items = [x for x in ((pov, "final in-game view"),
                             (clouds, "built structure (voxel)")) if x[0]]
        if not items:
            return
        box = ttk.LabelFrame(target, text="What was built", padding=6)
        box.pack(fill="x", pady=8)
        row = ttk.Frame(box)
        row.pack(anchor="w")
        for path, caption in items:
            img = self._load_image(path, _THUMB_WIDTH, target)
            if img is None:
                continue
            col = ttk.Frame(row)
            col.pack(side="left", padx=(0, 12))
            pic = ttk.Label(col, image=img, cursor="hand2")
            pic.pack()
            pic.bind("<Button-1>", lambda e, p=path: os.startfile(p))
            ttk.Label(col, text=caption, foreground="#7a8591").pack()
        buttons = ttk.Frame(box)
        buttons.pack(anchor="w", pady=(6, 0))
        if pngs:
            ttk.Button(buttons, text="Open all graphs",
                       command=lambda: self._open(sessions.reports_dir(sid))).pack(side="left")
        if pov:
            ttk.Button(buttons, text="Open frames",
                       command=lambda: self._open(sessions.frames_dir(sid))).pack(side="left", padx=6)

    def _render_declare(self, sid, declared: dict | None, target) -> None:
        """What the builder SAYS they are building. A declaration is not a label:
        it feeds material planning and eval only — never the belief (D7)."""
        box = ttk.LabelFrame(target, text="Declared target", padding=6)
        box.pack(fill="x", pady=8)
        goal_var = tk.StringVar(value=(declared or {}).get("goal", sessions.GOALS[0]))
        subtype_var = tk.StringVar(value=(declared or {}).get("subtype", ""))
        row = ttk.Frame(box)
        row.pack(anchor="w")
        ttk.Combobox(row, textvariable=goal_var, values=list(sessions.GOALS),
                     state="readonly", width=16).pack(side="left")
        ttk.Entry(row, textvariable=subtype_var, width=24).pack(side="left", padx=6)
        ttk.Button(row, text="Declare",
                   command=lambda: self._do_declare(sid, goal_var.get(),
                                                    subtype_var.get())).pack(side="left")
        note = (f'declared {declared.get("declared_when", "")} — used for material '
                'planning and eval, never fed to the belief'
                if declared else
                "optional: say what you were building — feeds material planning and "
                "eval, never the belief")
        ttk.Label(box, foreground="#7a8591", wraplength=560, text=note).pack(anchor="w")

    def _do_declare(self, sid, goal, subtype) -> None:
        error = sessions.set_declared_target(sid, goal, subtype)
        if error:
            messagebox.showwarning("Declare", error)
            return
        self.refresh_all()

    def _render_verdict(self, verdict: dict | None, target) -> None:
        if not verdict:
            return
        label, builder = verdict.get("label", {}), verdict.get("builder_label", {})
        agree = verdict.get("agrees_with_builder")
        box = ttk.LabelFrame(target, text="Matcher verdict", padding=8)
        box.pack(fill="x", pady=8)
        ttk.Label(box, wraplength=560, foreground=("#1f7a3d" if agree else "#9a5312"),
                  text=(f'you said {builder.get("goal","?")}/{builder.get("subtype","?")}; '
                        f'the matcher read {label.get("goal","?")}/{label.get("subtype","?")} '
                        f'(score {label.get("score","?")}) — '
                        f'{"AGREES" if agree else "CONTESTS"}')).pack(anchor="w")
        vlm = (verdict.get("vlm") or {}).get("category")
        ttk.Label(box, foreground="#7a8591", text=(
            f'VLM cross-check: {vlm or "n/a"} · pairs: {verdict.get("pairs", 0)} · '
            f'{"counts toward the cascade" if (label.get("kept") and agree) else "not counted"}')
                  ).pack(anchor="w")

    def _render_job(self, job: dict | None, target) -> None:
        if not job:
            return
        if job["state"] == "running":
            kind = "processing" if job.get("kind") == "process" else "labeling"
            ttk.Label(target, foreground="#8a6d10",
                      text=f"{kind}… running in the background (this updates when done)"
                      ).pack(anchor="w", pady=4)
        elif job["state"] == "failed":
            ttk.Label(target, foreground="#a12b2b",
                      text="last run failed: " + " ".join(job.get("tail", [])[-2:])
                      ).pack(anchor="w", pady=4)

    def _render_form(self, sid, state, label, target, allow_relabel, verdict) -> None:
        relabel = allow_relabel and verdict is not None
        can_label = state == "ready" or relabel
        frm = ttk.Frame(target)
        frm.pack(fill="x", pady=8)

        ttk.Label(frm, text="Goal").grid(row=0, column=0, sticky="w")
        goal_var = tk.StringVar(value=label.get("goal", sessions.GOALS[0]))
        goal_box = ttk.Combobox(frm, textvariable=goal_var, values=list(sessions.GOALS),
                                state="readonly", width=20)
        goal_box.grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(frm, text="Subtype").grid(row=1, column=0, sticky="w")
        subtype_var = tk.StringVar(value=label.get("subtype", ""))
        ttk.Entry(frm, textvariable=subtype_var, width=28).grid(row=1, column=1, sticky="w", pady=2)

        chips = ttk.Frame(frm)
        chips.grid(row=2, column=1, sticky="w")

        def rebuild_chips(*_a):
            # Two groups: styles the matcher's templates recognize, and the wider
            # documented styles (grey — fine as labels, agreement is category-level).
            for c in chips.winfo_children():
                c.destroy()
            groups = [("matcher templates:", sessions.subtype_presets(goal_var.get()), "TButton"),
                      ("more styles:", sessions.roadmap_presets(goal_var.get()), "Roadmap.TButton")]
            for caption, values, chip_style in groups:
                if not values:
                    continue
                ttk.Label(chips, text=caption, foreground="#7a8591").pack(anchor="w", pady=(3, 0))
                row = None
                for index, s in enumerate(values):
                    if index % 5 == 0:               # wrap so long lists don't overflow
                        row = ttk.Frame(chips)
                        row.pack(anchor="w")
                    ttk.Button(row, text=s, width=max(6, len(s)), style=chip_style,
                               command=lambda v=s: subtype_var.set(v)).pack(side="left", padx=1, pady=1)
        goal_box.bind("<<ComboboxSelected>>", rebuild_chips)
        rebuild_chips()

        vlm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="VLM cross-check", variable=vlm_var).grid(
            row=3, column=1, sticky="w", pady=(4, 2))

        buttons = ttk.Frame(frm)
        buttons.grid(row=4, column=1, sticky="w", pady=4)
        btn = ttk.Button(buttons, text=("Re-label" if relabel else "Label"),
                         command=lambda: self._do_label(sid, goal_var.get(),
                                                        subtype_var.get(), vlm_var.get()))
        btn.pack(side="left")
        add = ttk.Button(buttons, text="Add to queue",
                         command=lambda: self._queue_add(sid, goal_var.get(),
                                                         subtype_var.get()))
        add.pack(side="left", padx=6)
        if not can_label:
            btn.state(["disabled"])
            add.state(["disabled"])
        if state in ("unprocessed", "processing"):
            ttk.Button(buttons, text="Process evidence",
                       command=lambda: self._do_process(sid)).pack(side="left", padx=6)
        ttk.Button(buttons, text="Skip", command=lambda: self._do_skip(sid)).pack(side="left", padx=6)

        if relabel:
            note = "Re-labeling re-runs the matcher and overwrites the current label."
        elif state == "unprocessed":
            note = ("No evidence generated yet — captured before the automation, or its chain "
                    "never ran. Nothing is processing it now. Skip it, or press Process evidence "
                    "to generate the evidence (then it becomes labelable).")
        elif state == "quarantined":
            note = ("Can't label this one: its structure evidence is quarantined (an unexplained "
                    "voxel divergence, or the build left the capture region), so it isn't "
                    "proof-grade. Skip it.")
        elif state == "gate failed":
            note = "Can't label this one: it failed the B0 capture-integrity gate. Skip it."
        elif state == "labeling":
            note = "Labeling is running in the background…"
        elif state == "processing":
            note = ("Evidence exists but no session report yet — the rig's post-session "
                    "chain is either still running (it becomes labelable when it "
                    "finishes) or it died partway. If nothing changes in a few minutes, "
                    "press Process evidence to redo it — but not while the rig is "
                    "actively working on this session (two runs would race).")
        else:
            note = ""
        if note:
            ttk.Label(target, foreground="#7a8591", wraplength=560, text=note).pack(anchor="w")

    # actions ---------------------------------------------------------------

    def _label_inputs_ok(self, goal, subtype) -> bool:
        # Say WHY a click did nothing instead of silently ignoring it.
        if not subtype.strip():
            messagebox.showwarning("Subtype needed",
                                   "Type a subtype (e.g. 'crop field') first.")
            return False
        if goal not in sessions.GOALS:
            messagebox.showwarning("Goal needed", "Pick a goal from the dropdown.")
            return False
        return True

    def _do_label(self, sid, goal, subtype, vlm) -> None:
        if not self._label_inputs_ok(goal, subtype):
            return
        if not sessions.start_label_job(sid, goal, subtype.strip(), vlm):
            messagebox.showinfo("One at a time",
                                "Another session's job is still running — it must "
                                "finish first (the jobs share the pipeline's files).")
        self.refresh_all()

    # queue actions ----------------------------------------------------------

    def _queue_render(self) -> None:
        self.queue_tree.delete(*self.queue_tree.get_children())
        for e in self.queue:
            self.queue_tree.insert("", "end", iid=e["session"], text=e["session"],
                                   values=(f'{e["goal"]}/{e["subtype"]}',))
        self.queue_btn.configure(text=f"Label all ({len(self.queue)})")

    def _queue_add(self, sid, goal, subtype) -> None:
        if not self._label_inputs_ok(goal, subtype):
            return
        # Re-staging a session replaces its earlier entry (an edited label wins).
        entry = {"session": sid, "goal": goal, "subtype": subtype.strip()}
        self.queue = [e for e in self.queue if e["session"] != sid] + [entry]
        self._queue_render()

    def _queue_remove(self) -> None:
        sel = self.queue_tree.selection()
        if sel:
            self.queue = [e for e in self.queue if e["session"] != sel[0]]
            self._queue_render()

    def _queue_run(self) -> None:
        problem = sessions.batch_problem(self.queue)
        if problem:
            messagebox.showwarning("Queue not ready", problem)
            return
        if not sessions.start_batch_label_job(list(self.queue), self.queue_vlm.get()):
            messagebox.showinfo("One at a time",
                                "Another job is still running — it must finish "
                                "first (the jobs share the pipeline's files).")
            return
        self.queue = []
        self._queue_render()
        self.refresh_all()

    def _do_process(self, sid) -> None:
        if not sessions.start_process_job(sid):
            messagebox.showinfo("One at a time",
                                "Another session's job is still running — it must "
                                "finish first (the jobs share the pipeline's files).")
        self.refresh_all()

    def _do_skip(self, sid) -> None:
        sessions.skip(sid)
        self._sel = None
        self.refresh_all()

    def _open(self, path) -> None:
        if os.path.isdir(path):
            os.startfile(path)                          # native file explorer

    # ----------------------------------------------------------------- status

    def _build_status(self) -> None:
        self.status_body = ScrollFrame(self.tab_status)
        self.status_body.pack(fill="both", expand=True)

    def refresh_status(self, snap: dict) -> None:
        body = self.status_body.inner
        for c in body.winfo_children():
            c.destroy()
        r = snap["status"]
        ttk.Label(body, text="Cascade readiness",
                  font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(body, font=("Segoe UI", 26, "bold"),
                  text=f'{r["agreed_captures_total"]}  ').pack(anchor="w", pady=(8, 0))
        ttk.Label(body, foreground="#7a8591",
                  text="matcher-agreed captures").pack(anchor="w")
        if r["last_cascade"] == "never":
            since = f'{r["pairs_total"]} training pairs banked; no cascade recorded yet'
        else:
            since = (f'since the last cascade ({r["last_cascade"]}): '
                     f'{len(r["new_agreed_since_cascade"])} new agreed captures, '
                     f'+{r["new_pairs_since_cascade"]} pairs')
        ttk.Label(body, text=since).pack(anchor="w", pady=(10, 6))
        ready = r["last_cascade"] == "never" or r["new_agreed_since_cascade"]
        ttk.Label(body, foreground=("#1f7a3d" if ready else "#5a6570"),
                  font=("Segoe UI", 11, "bold"),
                  text=("READY — the batch is worth a retrain" if ready
                        else "nothing new since the last cascade")).pack(anchor="w")
        ttk.Label(body, foreground="#7a8591",
                  text="The cascade retrain stays a manual step. When ready, run in a terminal:"
                  ).pack(anchor="w", pady=(8, 4))
        cmd = ttk.Entry(body, width=44)
        cmd.insert(0, "python scripts/run_cascade_a.py")
        cmd.configure(state="readonly")
        cmd.pack(anchor="w")

        # Which trained models are live on disk — the set a cascade replaces.
        models = ttk.LabelFrame(body, text="Models on disk", padding=8)
        models.pack(fill="x", pady=(16, 0))
        for row in snap["models"]:
            line = ttk.Frame(models)
            line.pack(fill="x")
            ttk.Label(line, text="present" if row["present"] else "MISSING", width=8,
                      foreground=("#1f7a3d" if row["present"] else "#a12b2b")).pack(side="left")
            ttk.Label(line, text=row["name"], width=28).pack(side="left")
            ttk.Label(line, text=row["trained"] or "", width=17,
                      foreground="#4a5560").pack(side="left")
            ttk.Label(line, text=row["facts"], foreground="#7a8591").pack(side="left")

        # Who feeds the next cascade and who doesn't — nothing unlabeled,
        # skipped, quarantined, or contested can slip into a retrain unseen.
        check = snap["checklist"]
        pairs = sum(e["pairs"] for e in check["included"])
        panel = ttk.LabelFrame(
            body, text=(f'Cascade checklist — {len(check["included"])} sessions feed it '
                        f'({pairs} pairs) · {len(check["excluded"])} excluded'), padding=8)
        panel.pack(fill="x", pady=(16, 0))
        for e in check["included"]:
            ttk.Label(panel, foreground="#1f7a3d",
                      text=f'IN   {e["sid"]}   {e["label"]}   ·   {e["pairs"]} pairs'
                      ).pack(anchor="w")
        for e in check["excluded"]:
            ttk.Label(panel, foreground="#7a8591",
                      text=f'out  {e["sid"]}   —   {e["reason"]}').pack(anchor="w")

    # ----------------------------------------------------------------- history

    def _build_history(self) -> None:
        pane = ttk.Panedwindow(self.tab_history, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=6, pady=6)
        left = ttk.Frame(pane)

        ttk.Label(left, text="Labeled — select to review or re-label").pack(anchor="w")
        self.hist_labeled = ttk.Treeview(left, columns=("label", "matcher"),
                                         show="tree headings", height=13, selectmode="browse")
        self.hist_labeled.heading("#0", text="session")
        self.hist_labeled.heading("label", text="your label")
        self.hist_labeled.heading("matcher", text="matcher")
        self.hist_labeled.column("#0", width=190)
        self.hist_labeled.column("label", width=140)
        self.hist_labeled.column("matcher", width=80, anchor="center")
        self.hist_labeled.tag_configure("agrees", foreground="#1f7a3d")
        self.hist_labeled.tag_configure("contests", foreground="#9a5312")
        self.hist_labeled.pack(fill="both", expand=True, pady=(2, 8))
        self.hist_labeled.bind("<<TreeviewSelect>>",
                               lambda e: self._on_history_select(self.hist_labeled, True))

        row = ttk.Frame(left)
        row.pack(fill="x")
        ttk.Label(row, text="Skipped").pack(side="left")
        ttk.Button(row, text="Un-skip selected", command=self._do_unskip).pack(side="right")
        self.hist_skipped = ttk.Treeview(left, columns=(), show="tree", height=7,
                                         selectmode="browse")
        self.hist_skipped.column("#0", width=260)
        self.hist_skipped.pack(fill="x", pady=(2, 0))
        self.hist_skipped.bind("<<TreeviewSelect>>",
                               lambda e: self._on_history_select(self.hist_skipped, False))
        pane.add(left, weight=1)

        self.hist_detail = ScrollFrame(pane)
        pane.add(self.hist_detail, weight=2)

    def refresh_history(self, h: dict) -> None:
        keep = self._hist_sel
        self.hist_labeled.delete(*self.hist_labeled.get_children())
        for e in h["labeled"]:
            v = e.get("verdict") or {}
            tag = "" if not v else ("agrees" if v.get("agrees_with_builder") else "contests")
            self.hist_labeled.insert("", "end", iid=e["sid"], text=e["sid"], tags=(tag,), values=(
                f'{e["label"].get("goal","")}/{e["label"].get("subtype","")}', tag))
        self.hist_skipped.delete(*self.hist_skipped.get_children())
        for e in h["skipped"]:
            self.hist_skipped.insert("", "end", iid=e["sid"], text=e["sid"])
        # re-render the detail for a kept selection so a re-label's verdict shows
        if keep and (self.hist_labeled.exists(keep) or self.hist_skipped.exists(keep)):
            allow = self.hist_labeled.exists(keep)
            self._render_detail(keep, self.hist_detail.inner, allow_relabel=allow)
        else:
            self._hist_sel = None
            self._render_detail(None, self.hist_detail.inner)

    def _on_history_select(self, tree, allow_relabel) -> None:
        sel = tree.selection()
        if not sel:
            return
        self._hist_sel = sel[0]
        self._render_detail(sel[0], self.hist_detail.inner, allow_relabel=allow_relabel)

    def _do_unskip(self) -> None:
        sel = self.hist_skipped.selection()
        if sel:
            sessions.unskip(sel[0])
            self._hist_sel = None
            self.refresh_all()

    # ----------------------------------------------------------------- shared

    def refresh_all(self) -> None:
        # Gather everything in a worker thread — one pass over the pipeline's
        # files — and render when it lands, so the window never blocks on IO.
        # A request made during a running pass coalesces into exactly one
        # follow-up pass instead of piling up.
        if self._refreshing:
            self._refresh_again = True
            return
        self._refreshing = True
        threading.Thread(target=self._gather_snapshot, daemon=True).start()
        self.root.after(60, self._check_snapshot)

    def _gather_snapshot(self) -> None:
        # Worker thread: no Tk calls here (they are only safe on the UI thread) —
        # the result is handed off through an attribute the UI thread polls.
        try:
            self._pending_snap = sessions.snapshot()
        except Exception as error:
            self._pending_snap = {"error": str(error)}

    def _check_snapshot(self) -> None:
        snap = self._pending_snap
        if snap is None:
            self.root.after(60, self._check_snapshot)
            return
        self._pending_snap = None
        self._refreshing = False
        if snap.get("error"):
            self.status_var.set(f"refresh failed: {snap['error']}")
        else:
            self.refresh_inbox(snap["inbox"])
            self.refresh_status(snap)
            self.refresh_history(snap["history"])
        if self._refresh_again:
            self._refresh_again = False
            self.refresh_all()

    def _poll_jobs(self) -> None:
        # Drive the bottom activity bar from any running job, and refresh the view
        # ONCE when a job finishes (not every tick — that reloaded the pictures and
        # flickered). The bar animates + the timer counts up while it runs.
        running = None
        for sid in list(sessions._jobs):
            job = sessions.job(sid)
            if job and job.get("state") == "running":
                running = (sid, job)
                break
        if running:
            sid, job = running
            elapsed = int(time.time() - job.get("started", time.time()))
            if job.get("kind") == "batch":
                what = "batch labeling — one matcher run"
                where = f'{len(job.get("sessions", []))} sessions'
            elif job.get("kind") == "process":
                what, where = "processing evidence (gate + perception)", sid
            else:
                what, where = "labeling — running the matcher", sid
            self.status_var.set(f"{what}  ·  {where}  ·  {elapsed}s elapsed")
            if not self._bar_running:
                self.bar.pack(side="left", padx=10)
                self.bar.start(12)
                self._bar_running = True
            self._had_running = True
        else:
            if self._bar_running:
                self.bar.stop()
                self.bar.pack_forget()
                self._bar_running = False
            if self._had_running:              # a job just finished — show the result
                self._had_running = False
                self.status_var.set("")
                self.refresh_all()
        self.root.after(700, self._poll_jobs)


def main() -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
