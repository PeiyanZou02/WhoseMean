"""Standalone Tkinter interface for the Whose Mean 512px training studio.

Presentation is driven entirely by :mod:`whose_mean.theme`: a monochrome
neutral ramp, one runtime-resolved sans-serif family, an 8px spacing rhythm
and a small set of radii and motion durations. No literal colours, font
families or ad-hoc paddings should appear below.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from . import theme, training
from .data import IMAGES, OUT, load_records, square, thumbnail_atlas
from .theme import BORDER, COLOR, METRIC, MOTION, RADIUS, SPACE


WORKS_PATH = OUT / "works.json"

# Backwards-compatible aliases for the few colours that used to be literals.
BG = COLOR.surface
FG = COLOR.text
MUTED = COLOR.text_muted
LINE = COLOR.divider


# ---------------------------------------------------------------------------
# Small shared widgets
# ---------------------------------------------------------------------------


class Eyebrow(tk.Frame):
    """A section header: a quiet sentence-case label with an optional hint."""

    def __init__(self, master, text, hint=None, background=None):
        background = background or COLOR.surface
        super().__init__(master, bg=background)
        self.label = tk.Label(self, text=text, bg=background, fg=COLOR.text_muted,
                              font=theme.font("label"), anchor="w")
        self.label.pack(side="left")
        self.hint = tk.Label(self, text=hint or "", bg=background, fg=COLOR.text_muted,
                             font=theme.font("caption"), anchor="e")
        self.hint.pack(side="right")

    def set_hint(self, text):
        self.hint.configure(text=text or "")


class KeyValueList(tk.Frame):
    """A compact two-column readout with tabular alignment."""

    def __init__(self, master, background=None):
        background = background or COLOR.surface
        super().__init__(master, bg=background)
        self.background = background
        self.columnconfigure(1, weight=1)
        self._rows = []

    def set_rows(self, rows):
        while len(self._rows) < len(rows):
            index = len(self._rows)
            key = tk.Label(self, bg=self.background, fg=COLOR.text_muted,
                           font=theme.font("caption"), anchor="w")
            value = tk.Label(self, bg=self.background, fg=COLOR.text,
                             font=theme.font("numeric"), anchor="e")
            key.grid(row=index, column=0, sticky="w", pady=SPACE["xxs"])
            value.grid(row=index, column=1, sticky="e", pady=SPACE["xxs"])
            self._rows.append((key, value))
        for index, (key, value) in enumerate(self._rows):
            if index < len(rows):
                key.configure(text=str(rows[index][0]))
                value.configure(text=str(rows[index][1]))
                key.grid()
                value.grid()
            else:
                key.grid_remove()
                value.grid_remove()


class StatusBadge(tk.Canvas):
    """A monochrome state chip: filled when busy, outlined when at rest."""

    TONES = {
        "idle": dict(fill="", outline=COLOR.border, text=COLOR.text_secondary, weight="normal"),
        "active": dict(fill=COLOR.accent, outline=COLOR.accent, text=COLOR.text_inverse, weight="bold"),
        "done": dict(fill="", outline=COLOR.text, text=COLOR.text, weight="bold"),
        "error": dict(fill=COLOR.accent, outline=COLOR.accent, text=COLOR.text_inverse, weight="bold"),
    }

    def __init__(self, master, background=None):
        background = background or COLOR.surface
        super().__init__(master, height=22, bg=background, highlightthickness=0, bd=0)
        self.background = background
        self.text = "IDLE"
        self.tone = "idle"
        self.bind("<Configure>", lambda _event: self.draw())

    def set(self, text, tone="idle"):
        self.text = str(text)
        self.tone = tone if tone in self.TONES else "idle"
        self.draw()

    def draw(self):
        self.delete("all")
        tone = self.TONES[self.tone]
        font = theme.font("label") if tone["weight"] == "bold" else theme.font("caption")
        width = font.measure(self.text) + SPACE["md"] * 2
        self.configure(width=width)
        theme.rounded_rect(self, 1, 1, width - 1, 21, radius=RADIUS["pill"],
                           fill=tone["fill"] or self.background, outline=tone["outline"],
                           width=BORDER["thin"])
        self.create_text(width / 2, 11, text=self.text, fill=tone["text"], font=font)


class ProgressLine(tk.Canvas):
    """A rounded, low-chrome determinate progress track."""

    def __init__(self, master, width=320, background=None):
        background = background or COLOR.surface
        super().__init__(master, width=width, height=METRIC["progress_height"], bg=background,
                         highlightthickness=0, bd=0)
        self.background = background
        self.value = 0
        self.bind("<Configure>", lambda _event: self.draw())

    def set(self, value):
        self.value = max(0, min(1, float(value)))
        self.draw()

    def draw(self):
        self.delete("all")
        width = max(4, self.winfo_width())
        height = METRIC["progress_height"]
        radius = height / 2
        theme.rounded_rect(self, 0, 0, width, height, radius=radius,
                           fill=COLOR.surface_sunken, outline="")
        filled = width * self.value
        if filled > 1:
            theme.rounded_rect(self, 0, 0, max(filled, height), height, radius=radius,
                               fill=COLOR.accent, outline="")


class Slider(tk.Canvas):
    """A keyboard-accessible monochrome slider with hover and focus states."""

    def __init__(self, master, variable, command=None, width=160, background=None):
        background = background or COLOR.surface
        super().__init__(master, width=width, height=METRIC["slider_height"], bg=background,
                         highlightthickness=0, bd=0, cursor="hand2", takefocus=1)
        self.background = background
        self.variable = variable
        self.command = command
        self.hovered = False
        self.focused = False
        self.pressed = False
        self.bind("<Configure>", lambda _event: self.draw())
        self.bind("<Button-1>", self._press)
        self.bind("<B1-Motion>", self.set_from_pointer)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Enter>", lambda _event: self._set_hover(True))
        self.bind("<Leave>", lambda _event: self._set_hover(False))
        self.bind("<FocusIn>", lambda _event: self._set_focus(True))
        self.bind("<FocusOut>", lambda _event: self._set_focus(False))
        for sequence, step in (("<Left>", -0.05), ("<Right>", 0.05),
                               ("<Down>", -0.05), ("<Up>", 0.05)):
            self.bind(sequence, lambda _event, delta=step: self._nudge(delta))
        self.bind("<Home>", lambda _event: self._assign(0.0))
        self.bind("<End>", lambda _event: self._assign(1.0))
        self.draw()

    # -- state ---------------------------------------------------------
    def _set_hover(self, value):
        self.hovered = value
        self.draw()

    def _set_focus(self, value):
        self.focused = value
        self.draw()

    def _press(self, event):
        self.pressed = True
        self.focus_set()
        self.set_from_pointer(event)

    def _release(self, _event):
        self.pressed = False
        self.draw()

    def _nudge(self, delta):
        self._assign(float(self.variable.get()) + delta)

    def _assign(self, value):
        value = max(0.0, min(1.0, float(value)))
        self.variable.set(round(value, 2))
        self.draw()
        if self.command:
            self.command(value)

    def set_from_pointer(self, event):
        inset = METRIC["slider_knob"] / 2
        width = max(2 * inset + 2, self.winfo_width())
        self._assign((event.x - inset) / (width - 2 * inset))

    # -- paint ---------------------------------------------------------
    def draw(self):
        self.delete("all")
        width = max(24, self.winfo_width())
        height = METRIC["slider_height"]
        knob = METRIC["slider_knob"]
        inset = knob / 2
        centre = height / 2
        track = METRIC["slider_track"]
        theme.rounded_rect(self, inset, centre - track / 2, width - inset, centre + track / 2,
                           radius=track / 2, fill=COLOR.surface_sunken, outline="")
        position = inset + float(self.variable.get()) * (width - 2 * inset)
        if position > inset + 1:
            theme.rounded_rect(self, inset, centre - track / 2, position, centre + track / 2,
                               radius=track / 2, fill=COLOR.accent, outline="")
        if self.focused:
            self.create_oval(position - inset - 3, centre - inset - 3,
                             position + inset + 3, centre + inset + 3,
                             outline=COLOR.focus, width=BORDER["thin"])
        fill = COLOR.accent_pressed if self.pressed else (
            COLOR.accent_hover if self.hovered else COLOR.accent)
        self.create_oval(position - inset, centre - inset, position + inset, centre + inset,
                         fill=fill, outline=COLOR.surface, width=BORDER["thick"])


class ScrollImage(tk.Frame):
    """A vertically scrolling image well with a proper empty state."""

    def __init__(self, master, empty_text="Nothing here yet.", empty_hint="", **kwargs):
        super().__init__(master, bg=COLOR.surface, **kwargs)
        self.empty_text = empty_text
        self.empty_hint = empty_hint
        self.canvas = tk.Canvas(self, bg=COLOR.surface, highlightthickness=0, bd=0)
        bar = ttk.Scrollbar(self, orient="vertical", style="Studio.Vertical.TScrollbar",
                            command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda _event: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind("<Button-5>", lambda _event: self.canvas.yview_scroll(1, "units"))
        self.canvas.bind("<Configure>", self._reflow)
        self.photo = None

    def _wheel(self, event):
        self.canvas.yview_scroll(-int(event.delta / 120), "units")

    def _reflow(self, _event=None):
        if self.photo is None:
            self.show(None)

    def show(self, image: Image.Image | None):
        self.canvas.delete("all")
        self.photo = None
        if image is None:
            width = max(200, self.canvas.winfo_width())
            self.canvas.create_text(width / 2, SPACE["xxl"] + SPACE["lg"], text=self.empty_text,
                                    fill=COLOR.text_secondary, font=theme.font("body"),
                                    anchor="center", width=width - SPACE["xl"] * 2)
            if self.empty_hint:
                self.canvas.create_text(width / 2, SPACE["xxl"] + SPACE["xxl"] + SPACE["sm"],
                                        text=self.empty_hint, fill=COLOR.text_muted,
                                        font=theme.font("caption"), anchor="center",
                                        width=width - SPACE["xl"] * 2)
            self.canvas.configure(scrollregion=(0, 0, width, 120))
            return
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(SPACE["sm"], SPACE["sm"], image=self.photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, image.width + SPACE["lg"],
                                            image.height + SPACE["lg"]))


class ScrollColumn(tk.Frame):
    """A vertically scrolling column that only shows its scrollbar when needed."""

    def __init__(self, master, width):
        super().__init__(master, bg=COLOR.surface, width=width)
        self.canvas = tk.Canvas(self, bg=COLOR.surface, highlightthickness=0, bd=0, width=width)
        self.bar = ttk.Scrollbar(self, orient="vertical", style="Studio.Vertical.TScrollbar",
                                 command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._on_scroll)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.body = tk.Frame(self.canvas, bg=COLOR.surface)
        self._window = self.canvas.create_window(0, 0, window=self.body, anchor="nw")
        self.body.bind("<Configure>", self._sync)
        self.canvas.bind("<Configure>", self._resize)
        self.bind_wheel(self.canvas)
        self.bind_wheel(self.body)

    def _on_scroll(self, first, last):
        fits = float(first) <= 0.0 and float(last) >= 1.0
        mapped = bool(self.bar.winfo_ismapped())
        if fits and mapped:
            self.bar.pack_forget()
        elif not fits and not mapped:
            # Packed before the expanding canvas so it is allocated its width.
            self.bar.pack(side="right", fill="y", before=self.canvas)
        self.bar.set(first, last)

    def _sync(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize(self, event):
        self.canvas.itemconfigure(self._window, width=event.width)

    def _wheel(self, event):
        delta = getattr(event, "delta", 0)
        if delta:
            self.canvas.yview_scroll(-int(delta / 120), "units")
        elif getattr(event, "num", None) in (4, 5):
            self.canvas.yview_scroll(-1 if event.num == 4 else 1, "units")
        return "break"

    def bind_wheel(self, widget):
        widget.bind("<MouseWheel>", self._wheel)
        widget.bind("<Button-4>", self._wheel)
        widget.bind("<Button-5>", self._wheel)

    def bind_wheel_tree(self, widget=None, skip=()):
        """Route wheel events from every plain child to the column."""
        widget = self.body if widget is None else widget
        for child in widget.winfo_children():
            if child in skip:
                continue
            self.bind_wheel(child)
            self.bind_wheel_tree(child, skip)


class ZoomPane(tk.Frame):
    """A titled, bordered viewport for a single zoomable image."""

    def __init__(self, master, title, wheel_command):
        super().__init__(master, bg=COLOR.surface)
        tk.Label(self, text=title, bg=COLOR.surface, fg=COLOR.text_muted,
                 font=theme.font("label"), anchor="w").pack(anchor="w", pady=(0, SPACE["sm"]))
        holder = tk.Frame(self, bg=COLOR.surface, highlightbackground=COLOR.border,
                          highlightcolor=COLOR.border, highlightthickness=BORDER["thin"])
        holder.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(holder, bg=COLOR.surface_sunken, highlightthickness=0, bd=0)
        xbar = ttk.Scrollbar(holder, orient="horizontal", style="Studio.Horizontal.TScrollbar",
                             command=self.canvas.xview)
        ybar = ttk.Scrollbar(holder, orient="vertical", style="Studio.Vertical.TScrollbar",
                             command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        holder.rowconfigure(0, weight=1)
        holder.columnconfigure(0, weight=1)
        self.canvas.bind("<MouseWheel>", wheel_command)
        self.photo = None

    def show(self, image, zoom):
        size = max(128, int(512 * zoom))
        rendered = image.resize((size, size), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(rendered)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, size, size))


# ---------------------------------------------------------------------------
# Secondary windows
# ---------------------------------------------------------------------------


class AttachedToplevel(tk.Toplevel):
    """A movable child window that keeps its position relative to the studio."""

    def __init__(self, master, default_width, default_height):
        super().__init__(master, bg=COLOR.surface)
        self.owner = master
        master.update_idletasks()
        owner_width = max(1, master.winfo_width())
        owner_height = max(1, master.winfo_height())
        width = min(default_width, max(520, owner_width - 40))
        height = min(default_height, max(420, owner_height - 70))
        self._relative_position = (SPACE["xl"], SPACE["xxxl"])
        self.geometry(f'{width}x{height}+{master.winfo_x() + SPACE["xl"]}'
                      f'+{master.winfo_y() + SPACE["xxxl"]}')
        self.transient(master)
        self._following_owner = False
        self._owner_binding = master.bind("<Configure>", self._follow_owner, add="+")
        self.bind("<Configure>", self._remember_position, add="+")
        self.bind("<Destroy>", self._detach_owner, add="+")

    def _remember_position(self, event):
        if event.widget is self and not self._following_owner and self.winfo_exists():
            self._relative_position = (self.winfo_x() - self.owner.winfo_x(),
                                       self.winfo_y() - self.owner.winfo_y())

    def _follow_owner(self, event):
        if event.widget is not self.owner or not self.winfo_exists():
            return
        dx, dy = self._relative_position
        self._following_owner = True
        try:
            self.geometry(f"+{self.owner.winfo_x() + dx}+{self.owner.winfo_y() + dy}")
        finally:
            self.after_idle(
                lambda: setattr(self, "_following_owner", False) if self.winfo_exists() else None)

    def _detach_owner(self, event):
        if event.widget is self and self._owner_binding:
            try:
                self.owner.unbind("<Configure>", self._owner_binding)
            except tk.TclError:
                pass
            self._owner_binding = None

    # -- shared chrome -------------------------------------------------
    def build_header(self, title, subtitle=None):
        """A consistent window header: title block left, zoom controls right."""
        header = tk.Frame(self, bg=COLOR.surface)
        header.pack(fill="x", padx=SPACE["xl"], pady=(SPACE["lg"], SPACE["md"]))
        titles = tk.Frame(header, bg=COLOR.surface)
        titles.pack(side="left", fill="x", expand=True)
        tk.Label(titles, text=title, bg=COLOR.surface, fg=COLOR.text,
                 font=theme.font("title"), anchor="w").pack(anchor="w")
        if subtitle:
            tk.Label(titles, text=subtitle, bg=COLOR.surface, fg=COLOR.text_muted,
                     font=theme.font("caption"), anchor="w").pack(anchor="w",
                                                                  pady=(SPACE["xxs"], 0))
        controls = tk.Frame(header, bg=COLOR.surface)
        controls.pack(side="right")
        ttk.Button(controls, text="−", style="Quiet.TButton", width=3,
                   command=lambda: self.set_zoom(self.zoom - .2)).pack(side="left")
        self.zoom_label = tk.Label(controls, text="100%", bg=COLOR.surface, fg=COLOR.text_secondary,
                                   font=theme.font("numeric"), width=5)
        self.zoom_label.pack(side="left", padx=SPACE["xs"])
        ttk.Button(controls, text="+", style="Quiet.TButton", width=3,
                   command=lambda: self.set_zoom(self.zoom + .2)).pack(side="left")
        ttk.Separator(self, orient="horizontal", style="Divider.TSeparator").pack(fill="x")
        return header


class ComparisonViewer(AttachedToplevel):
    def __init__(self, master, title, original, generated):
        super().__init__(master, 1120, 720)
        self.title(title)
        self.minsize(820, 560)
        self.original, self.generated = original, generated
        self.zoom = 1.0
        self.build_header(title, "Original and generated, zoomed together")
        body = tk.Frame(self, bg=COLOR.surface)
        body.pack(fill="both", expand=True, padx=SPACE["xl"], pady=SPACE["xl"])
        self.left = ZoomPane(body, "Original", self.wheel)
        self.left.pack(side="left", fill="both", expand=True, padx=(0, SPACE["sm"]))
        self.right = ZoomPane(body, "pix2pix generated", self.wheel)
        self.right.pack(side="left", fill="both", expand=True, padx=(SPACE["sm"], 0))
        self.bind("<Escape>", lambda _event: self.destroy())
        self.set_zoom(1)

    def wheel(self, event):
        self.set_zoom(self.zoom * math.exp(event.delta / 120 * .12))

    def set_zoom(self, value):
        self.zoom = max(.4, min(4, value))
        self.zoom_label.configure(text=f"{self.zoom:.0%}")
        self.left.show(self.original, self.zoom)
        self.right.show(self.generated, self.zoom)


class ResultViewer(AttachedToplevel):
    def __init__(self, master, title, image):
        super().__init__(master, 860, 800)
        self.title(title)
        self.minsize(600, 520)
        self.image = image
        self.zoom = 1.0
        self.build_header(title, "Scroll to zoom · Esc to close")
        body = tk.Frame(self, bg=COLOR.surface)
        body.pack(fill="both", expand=True, padx=SPACE["xl"], pady=SPACE["xl"])
        self.pane = ZoomPane(body, "pix2pix mean", self.wheel)
        self.pane.pack(fill="both", expand=True)
        self.bind("<Escape>", lambda _event: self.destroy())
        self.set_zoom(1)

    def wheel(self, event):
        self.set_zoom(self.zoom * math.exp(event.delta / 120 * .12))

    def set_zoom(self, value):
        self.zoom = max(.4, min(6, value))
        self.zoom_label.configure(text=f"{self.zoom:.0%}")
        self.pane.show(self.image, self.zoom)


class ArtworkViewer(AttachedToplevel):
    """Original artwork beside a typographically structured metadata panel."""

    FIELDS = (
        ("Title", "title", "Untitled"),
        ("Artist", "attribution", "Unknown artist"),
        ("Date", "displaydate", "Unknown"),
        ("Classification", "classification", "Unknown"),
        ("Medium", "medium", "Unknown"),
        ("Object ID", "objectid", "—"),
        ("Credit line", "creditline", "—"),
        ("Description", "assistivetext", "No description available."),
    )

    def __init__(self, master, record, image):
        super().__init__(master, 1120, 740)
        title = record.get("title") or "Untitled"
        self.title(title)
        self.minsize(820, 560)
        self.image = image
        self.zoom = 1.0
        self.build_header(title, record.get("attribution") or "Unknown artist")
        body = tk.Frame(self, bg=COLOR.surface)
        body.pack(fill="both", expand=True, padx=SPACE["xl"], pady=SPACE["xl"])
        self.pane = ZoomPane(body, "Original artwork · scroll to zoom", self.wheel)
        self.pane.pack(side="left", fill="both", expand=True, padx=(0, SPACE["xl"]))

        info = tk.Frame(body, bg=COLOR.surface, width=320)
        info.pack(side="right", fill="y")
        info.pack_propagate(False)
        tk.Label(info, text="Artwork information", bg=COLOR.surface, fg=COLOR.text_muted,
                 font=theme.font("label"), anchor="w").pack(fill="x", pady=(0, SPACE["sm"]))
        well = tk.Frame(info, bg=COLOR.surface, highlightbackground=COLOR.border,
                        highlightcolor=COLOR.border, highlightthickness=BORDER["thin"])
        well.pack(fill="both", expand=True)
        text = tk.Text(well, bg=COLOR.surface, fg=COLOR.text, insertbackground=COLOR.text,
                       relief="flat", bd=0, wrap="word", font=theme.font("body"),
                       padx=SPACE["lg"], pady=SPACE["md"], cursor="arrow",
                       highlightthickness=0, spacing1=2, spacing3=3)
        scrollbar = ttk.Scrollbar(well, orient="vertical", style="Studio.Vertical.TScrollbar",
                                  command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        text.tag_configure("key", foreground=COLOR.text_muted, font=theme.font("label"),
                           spacing1=SPACE["md"])
        text.tag_configure("value", foreground=COLOR.text, font=theme.font("body"),
                           spacing3=SPACE["xs"], lmargin1=0, lmargin2=0)
        for label, key, default in self.FIELDS:
            value = record.get(key)
            value = str(value) if value not in (None, "") else default
            text.insert("end", f"{label}\n", "key")
            text.insert("end", f"{value}\n", "value")
        text.configure(state="disabled")

        self.bind("<Escape>", lambda _event: self.destroy())
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.set_zoom(1)
        self.lift()
        self.focus_force()

    def wheel(self, event):
        self.set_zoom(self.zoom * math.exp(event.delta / 120 * .12))

    def set_zoom(self, value):
        self.zoom = max(.4, min(6, value))
        self.zoom_label.configure(text=f"{self.zoom:.0%}")
        self.pane.show(self.image, self.zoom)


# ---------------------------------------------------------------------------
# The studio
# ---------------------------------------------------------------------------


class Studio:
    ACTIVE_STATES = {"queued", "preparing", "training", "generating"}
    STATE_TONES = {
        "idle": ("Idle", "idle"),
        "loading": ("Reading", "idle"),
        "queued": ("Queued", "active"),
        "preparing": ("Preparing", "active"),
        "training": ("Training", "active"),
        "generating": ("Generating", "active"),
        "complete": ("Complete", "done"),
        "stopped": ("Stopped", "done"),
        "error": ("Error", "error"),
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Whose Mean? · Peiyan Zou · ADV 9672 · W3 Reading Response")
        width = METRIC["default_window_width"]
        height = METRIC["default_window_height"]
        try:  # never open wider or taller than the display we launched on
            width = min(width, max(METRIC["min_window_width"], root.winfo_screenwidth() - 80))
            height = min(height, max(METRIC["min_window_height"], root.winfo_screenheight() - 120))
        except tk.TclError:  # pragma: no cover - exotic Tk builds
            pass
        root.geometry(f"{width}x{height}")
        root.minsize(METRIC["min_window_width"], METRIC["min_window_height"])
        root.configure(bg=COLOR.app)

        document = json.loads(WORKS_PATH.read_text(encoding="utf-8"))
        self.works = document["works"]
        self.by_id = {int(work["id"]): work for work in self.works}
        self.record_map = {int(record["objectid"]): record for record in load_records()}
        self.removed: set[int] = set()
        self.keyword_matches: set[int] | None = None
        self.keyword_edges = []
        self.mode = "inspect"
        self.hovered = None
        self.selected = None
        self.rot_x, self.rot_y = -0.18, 0.40
        self.scene_zoom, self.output_zoom = 1.0, 1.0
        self.dragging = False
        self.last_pointer = (0, 0)
        self.down_pointer = (0, 0)
        self.projected = []
        self.current_epoch = None
        self.follow_latest = True
        self.last_frames = ()
        self.last_result_run = None
        self.result_ids = []
        self.result_atlas = None
        self.scene_refs = []
        self.epoch_refs = []
        self.output_photo = None
        self.detail_photo = None
        self.detail_window = None

        self._configure_style()
        self._build_layout()
        self._load_atlas()
        self._bind_scene()
        self._bind_shortcuts()
        self.set_mode("inspect")
        self.update_keyword_highlights()
        self.root.after(100, self.poll_training)

    # -- chrome --------------------------------------------------------
    def _configure_style(self):
        self.style = theme.apply_root(self.root)

    def _build_layout(self):
        self._build_header()
        ttk.Separator(self.root, orient="horizontal", style="Divider.TSeparator").pack(fill="x")

        body = tk.Frame(self.root, bg=COLOR.app)
        body.pack(fill="both", expand=True)

        stage = tk.Frame(body, bg=COLOR.stage)
        stage.pack(side="left", fill="both", expand=True)
        self.scene = tk.Canvas(stage, bg=COLOR.stage, highlightthickness=0, bd=0, cursor="crosshair")
        self.scene.pack(fill="both", expand=True)

        divider = tk.Frame(body, bg=COLOR.divider, width=BORDER["thin"])
        divider.pack(side="left", fill="y")

        self.sidebar = ScrollColumn(body, METRIC["sidebar_width"])
        self.sidebar.pack(side="right", fill="y")
        self.sidebar.pack_propagate(False)
        self._build_sidebar(self.sidebar.body)
        self.sidebar.bind_wheel_tree(skip=(self.mean_canvas, self.epochs_view.canvas,
                                           self.results_view.canvas))

        ttk.Separator(self.root, orient="horizontal", style="Divider.TSeparator").pack(fill="x")
        self._build_footer()

    def _build_header(self):
        header = tk.Frame(self.root, bg=COLOR.surface, height=METRIC["header_height"])
        header.pack(fill="x")
        header.pack_propagate(False)

        titles = tk.Frame(header, bg=COLOR.surface)
        titles.pack(side="left", padx=SPACE["xl"])
        tk.Label(titles, text="Whose Mean?", bg=COLOR.surface, fg=COLOR.text,
                 font=theme.font("display"), anchor="w").pack(anchor="w")
        tk.Label(titles, text="Peiyan Zou · ADV 9672 · A negotiated mean image",
                 bg=COLOR.surface, fg=COLOR.text_muted, font=theme.font("caption"),
                 anchor="w").pack(anchor="w")

        actions = tk.Frame(header, bg=COLOR.surface)
        actions.pack(side="right", padx=SPACE["xl"])

        self.removed_label = tk.Label(actions, text="0 removed", bg=COLOR.surface,
                                      fg=COLOR.text_secondary, font=theme.font("caption"))
        self.removed_label.pack(side="left", padx=(0, SPACE["md"]))
        self.clear_button = ttk.Button(actions, text="Clear", style="Quiet.TButton",
                                       command=self.clear_removed)
        self.clear_button.pack(side="left", padx=(0, SPACE["lg"]))
        self.clear_button.state(["disabled"])

        tk.Label(actions, text="MODE", bg=COLOR.surface, fg=COLOR.text_muted,
                 font=theme.font("label")).pack(side="left", padx=(0, SPACE["sm"]))
        segment = tk.Frame(actions, bg=COLOR.surface)
        segment.pack(side="left")
        self.inspect_button = ttk.Button(segment, text="Inspect", style="SegmentActive.TButton",
                                         command=lambda: self.set_mode("inspect"))
        self.inspect_button.pack(side="left")
        self.remove_button = ttk.Button(segment, text="Remove", style="Segment.TButton",
                                        command=lambda: self.set_mode("remove"))
        self.remove_button.pack(side="left", padx=(SPACE["xs"], 0))

    def _section(self, parent, title, hint=None, top=SPACE["xl"]):
        """A labelled sidebar section separated by a hairline."""
        if getattr(self, "_sections_started", False):
            ttk.Separator(parent, orient="horizontal", style="Divider.TSeparator").pack(
                fill="x", padx=SPACE["xl"])
        self._sections_started = True
        block = tk.Frame(parent, bg=COLOR.surface)
        block.pack(fill="x", padx=SPACE["xl"], pady=(top, SPACE["lg"]))
        eyebrow = Eyebrow(block, title, hint)
        eyebrow.pack(fill="x", pady=(0, SPACE["md"]))
        return block, eyebrow

    def _build_sidebar(self, parent):
        self._sections_started = False

        # --- search ---------------------------------------------------
        search_block, _ = self._section(parent, "Concept keywords", "any match", top=SPACE["lg"])
        field = tk.Frame(search_block, bg=COLOR.surface)
        field.pack(fill="x")
        self.prompt = tk.StringVar()
        self.search_entry = ttk.Entry(field, textvariable=self.prompt, style="Search.TEntry",
                                      font=theme.font("body"))
        self.search_entry.pack(side="left", fill="x", expand=True)
        self.search_clear = ttk.Button(field, text="✕", style="Quiet.TButton", width=3,
                                       command=self.clear_search)
        self.search_clear.pack(side="left", padx=(SPACE["xs"], 0))
        self.search_clear.state(["disabled"])
        self.search_placeholder = tk.Label(field, text="e.g. tree, portrait, blue",
                                           bg=COLOR.surface, fg=COLOR.text_disabled,
                                           font=theme.font("body"))
        self.search_placeholder.place(in_=self.search_entry, relx=0, rely=0.5, x=SPACE["sm"] + 1,
                                      anchor="w")
        self.search_placeholder.bind("<Button-1>", lambda _event: self.search_entry.focus_set())
        self.search_entry.bind("<Return>", lambda _event: self.prompt_and_train())
        self.search_entry.bind("<Escape>", lambda _event: self.clear_search())
        self.prompt.trace_add("write", lambda *_args: self.update_keyword_highlights())

        self.search_status = tk.Label(search_block, text="", bg=COLOR.surface,
                                      fg=COLOR.text_muted, font=theme.font("caption"),
                                      anchor="w", justify="left",
                                      wraplength=METRIC["sidebar_width"] - SPACE["xxl"] * 2)
        self.search_status.pack(fill="x", pady=(SPACE["sm"], 0))

        # --- training controls ---------------------------------------
        training_block, _ = self._section(parent, "Training")
        row = tk.Frame(training_block, bg=COLOR.surface)
        row.pack(fill="x")
        tk.Label(row, text="Epochs", bg=COLOR.surface, fg=COLOR.text,
                 font=theme.font("body")).pack(side="left")
        self.epochs = tk.IntVar(value=20)
        self.epoch_spin = ttk.Spinbox(row, from_=1, to=200, textvariable=self.epochs, width=5,
                                      style="Studio.TSpinbox", justify="center",
                                      font=theme.font("numeric_strong"))
        self.epoch_spin.pack(side="right")

        slider_row = tk.Frame(training_block, bg=COLOR.surface)
        slider_row.pack(fill="x", pady=(SPACE["lg"], 0))
        self.structure = tk.DoubleVar(value=.65)
        labels = tk.Frame(slider_row, bg=COLOR.surface)
        labels.pack(fill="x")
        tk.Label(labels, text="Structure anchor", bg=COLOR.surface, fg=COLOR.text,
                 font=theme.font("body")).pack(side="left")
        self.structure_label = tk.Label(labels, text="65%", bg=COLOR.surface, fg=COLOR.text,
                                        font=theme.font("numeric"))
        self.structure_label.pack(side="right")
        self.structure_slider = Slider(slider_row, self.structure, self.update_structure_label,
                                       width=METRIC["sidebar_width"] - SPACE["xxl"] * 2)
        self.structure_slider.pack(fill="x", pady=(SPACE["xs"], 0))
        tk.Label(slider_row, text="Mixes the arithmetic pixel mean with the work nearest the "
                                  "feature centroid.", bg=COLOR.surface, fg=COLOR.text_muted,
                 font=theme.font("caption"), anchor="w", justify="left",
                 wraplength=METRIC["sidebar_width"] - SPACE["xxl"] * 2).pack(fill="x",
                                                                            pady=(SPACE["xs"], 0))

        actions = tk.Frame(training_block, bg=COLOR.surface)
        actions.pack(fill="x", pady=(SPACE["lg"], 0))
        self.generate_button = ttk.Button(actions, text="Generate", style="Primary.TButton",
                                          command=self.prompt_and_train)
        self.generate_button.pack(fill="x")
        self.stop_button = ttk.Button(actions, text="Stop", style="Secondary.TButton",
                                      command=self.stop_training)
        self.stop_button.pack(fill="x", pady=(SPACE["sm"], 0))
        self.stop_button.state(["disabled"])

        # --- run status ----------------------------------------------
        status_block, _ = self._section(parent, "Run status")
        badge_row = tk.Frame(status_block, bg=COLOR.surface)
        badge_row.pack(fill="x")
        self.status_badge = StatusBadge(badge_row)
        self.status_badge.pack(side="left")
        self.progress_value = tk.Label(badge_row, text="0%", bg=COLOR.surface, fg=COLOR.text,
                                       font=theme.font("numeric"))
        self.progress_value.pack(side="right")
        self.progress = ProgressLine(status_block)
        self.progress.pack(fill="x", pady=(SPACE["sm"], SPACE["md"]))
        self.message = tk.Label(status_block, text="Ready.", bg=COLOR.surface,
                                fg=COLOR.text_secondary, justify="left", anchor="w",
                                wraplength=METRIC["sidebar_width"] - SPACE["xxl"] * 2,
                                font=theme.font("body"))
        self.message.pack(fill="x")
        self.stats = KeyValueList(status_block)
        self.stats.pack(fill="x", pady=(SPACE["md"], 0))
        self.stats.set_rows([("Artwork", "—"), ("Train L1", "—")])

        # --- mean preview --------------------------------------------
        mean_block, self.mean_eyebrow = self._section(parent, "pix2pix mean", "no run yet")
        self.mean_title = self.mean_eyebrow.hint
        self.mean_canvas = tk.Canvas(mean_block, height=METRIC["mean_preview_height"],
                                     bg=COLOR.surface, highlightthickness=0, bd=0,
                                     cursor="hand2")
        self.mean_canvas.pack(fill="x")
        self.mean_canvas.bind("<MouseWheel>", self.mean_wheel)
        self.mean_canvas.bind("<Button-1>", lambda _event: self.open_large_result())
        self.mean_canvas.bind("<Configure>", lambda _event: self._redraw_mean())
        result_actions = tk.Frame(mean_block, bg=COLOR.surface)
        result_actions.pack(fill="x", pady=(SPACE["md"], 0))
        self.open_large_button = ttk.Button(result_actions, text="Open large",
                                            style="Secondary.TButton",
                                            command=self.open_large_result)
        self.open_large_button.pack(side="left", fill="x", expand=True, padx=(0, SPACE["xs"]))
        self.export_button = ttk.Button(result_actions, text="Export results",
                                        style="Secondary.TButton", command=self.export_results)
        self.export_button.pack(side="left", fill="x", expand=True, padx=(SPACE["xs"], 0))
        self.open_large_button.state(["disabled"])
        self.export_button.state(["disabled"])

        # --- outputs --------------------------------------------------
        ttk.Separator(parent, orient="horizontal", style="Divider.TSeparator").pack(
            fill="x", padx=SPACE["xl"])
        outputs = tk.Frame(parent, bg=COLOR.surface, height=METRIC["outputs_height"])
        outputs.pack(fill="x", padx=SPACE["lg"], pady=(SPACE["md"], SPACE["lg"]))
        outputs.pack_propagate(False)
        notebook = ttk.Notebook(outputs, style="Studio.TNotebook")
        notebook.pack(fill="both", expand=True)
        self.epochs_view = ScrollImage(notebook, "No epoch images yet.",
                                       "Press Generate to start a run.")
        self.results_view = ScrollImage(notebook, "No generated artworks yet.",
                                        "Results appear once a run finishes.")
        model_view = tk.Frame(notebook, bg=COLOR.surface)
        notebook.add(self.epochs_view, text="EPOCHS")
        notebook.add(self.results_view, text="RESULTS")
        notebook.add(model_view, text="MODEL I/O")
        io_rows = (
            ("Input", "512 × 512 × 3 · grayscale in RGB"),
            ("Target", "512 × 512 × 3 · RGB"),
            ("Output", "512 × 512 × 3 · RGB"),
            ("Encoder", "256²×32 → 128²×64 → 64²×128"),
            ("", "32²×256 → 16²×512 → 8²×512"),
            ("Decoder", "16²×512 → 32²×256 → 64²×128"),
            ("", "128²×64 → 256²×32 → 512²×3"),
            ("Batch", "1 artwork"),
        )
        model_inner = tk.Frame(model_view, bg=COLOR.surface)
        model_inner.pack(fill="both", expand=True, padx=SPACE["lg"], pady=SPACE["lg"])
        tk.Label(model_inner, text="PAIRED PIX2PIX TENSORS", bg=COLOR.surface,
                 fg=COLOR.text_muted, font=theme.font("label"), anchor="w").pack(
            fill="x", pady=(0, SPACE["sm"]))
        model_rows = KeyValueList(model_inner)
        model_rows.pack(fill="x")
        model_rows.set_rows(io_rows)
        self.model_note = tk.Label(model_inner, text="", bg=COLOR.surface, fg=COLOR.text_muted,
                                   font=theme.font("caption"), anchor="w", justify="left",
                                   wraplength=METRIC["sidebar_width"] - SPACE["xxl"] * 2)
        self.model_note.pack(fill="x", pady=(SPACE["md"], 0))
        self._update_model_note(.65)
        self.epochs_view.show(None)
        self.results_view.show(None)
        self._redraw_mean()

    def _build_footer(self):
        footer = tk.Frame(self.root, bg=COLOR.surface, height=METRIC["footer_height"])
        footer.pack(fill="x")
        footer.pack_propagate(False)
        self.footer_hint = tk.Label(footer, text="", bg=COLOR.surface, fg=COLOR.text_muted,
                                    font=theme.font("caption"), anchor="w")
        self.footer_hint.pack(side="left", padx=SPACE["xl"])

        def zoom_group(title, minus, plus):
            group = tk.Frame(footer, bg=COLOR.surface)
            group.pack(side="right", padx=(SPACE["lg"], 0))
            tk.Label(group, text=title, bg=COLOR.surface, fg=COLOR.text_muted,
                     font=theme.font("label")).pack(side="left", padx=(0, SPACE["sm"]))
            ttk.Button(group, text="−", style="Quiet.TButton", width=3, command=minus).pack(
                side="left")
            value = tk.Label(group, text="100%", bg=COLOR.surface, fg=COLOR.text,
                             font=theme.font("numeric"), width=5)
            value.pack(side="left", padx=SPACE["xxs"])
            ttk.Button(group, text="+", style="Quiet.TButton", width=3, command=plus).pack(
                side="left")
            return value

        self.output_zoom_label = zoom_group(
            "Output",
            lambda: self.set_output_zoom(self.output_zoom - .15),
            lambda: self.set_output_zoom(self.output_zoom + .15))
        self.space_zoom_label = zoom_group(
            "Space",
            lambda: self.set_scene_zoom(self.scene_zoom - .15),
            lambda: self.set_scene_zoom(self.scene_zoom + .15))
        # Right edge padding for the outermost group.
        tk.Frame(footer, bg=COLOR.surface, width=SPACE["xl"]).pack(side="right")

    def _bind_shortcuts(self):
        self.root.bind("<Control-f>", lambda _event: self._focus_search())
        self.root.bind("<Control-F>", lambda _event: self._focus_search())

    def _focus_search(self):
        self.search_entry.focus_set()
        self.search_entry.select_range(0, "end")
        return "break"

    # -- data ----------------------------------------------------------
    def _load_atlas(self):
        raw = thumbnail_atlas()
        self.atlas = Image.open(io.BytesIO(raw)).convert("RGB")
        self.small_photos = []
        self.dim_photos = []
        self.source_tiles = []
        self.dim_source_tiles = []
        for index in range(len(self.works)):
            x, y = (index % 20) * 64, (index // 20) * 64
            tile = self.atlas.crop((x, y, x + 64, y + 64))
            dim = Image.blend(Image.new("RGB", tile.size, COLOR.stage), tile, .30)
            self.source_tiles.append(tile)
            self.dim_source_tiles.append(dim)
            self.small_photos.append(ImageTk.PhotoImage(tile.resize((16, 16), Image.Resampling.LANCZOS)))
            self.dim_photos.append(ImageTk.PhotoImage(dim.resize((16, 16), Image.Resampling.LANCZOS)))
        self.draw_scene()

    def _bind_scene(self):
        self.scene.bind("<Configure>", lambda _e: self.draw_scene())
        self.scene.bind("<ButtonPress-1>", self.left_down)
        self.scene.bind("<ButtonRelease-1>", self.pointer_up)
        self.scene.bind("<ButtonPress-3>", self.pointer_down)
        self.scene.bind("<B3-Motion>", self.pointer_drag)
        self.scene.bind("<ButtonRelease-3>", self.right_up)
        self.scene.bind("<Motion>", self.pointer_move)
        self.scene.bind("<Leave>", lambda _e: self.set_hover(None))
        self.scene.bind("<MouseWheel>", self.mouse_wheel)

    # -- scene ---------------------------------------------------------
    def rotate(self, work):
        cy, sy = math.cos(self.rot_y), math.sin(self.rot_y)
        cx, sx = math.cos(self.rot_x), math.sin(self.rot_x)
        x = work["x"] * cy + work["z"] * sy
        z = -work["x"] * sy + work["z"] * cy
        return x, work["y"] * cx - z * sx, work["y"] * sx + z * cx

    def project(self, work):
        width, height = max(1, self.scene.winfo_width()), max(1, self.scene.winfo_height())
        x, y, z = self.rotate(work)
        scale = min(width * .68, height * .76) * self.scene_zoom
        perspective = 3 / (3.8 - z)
        return width * .50 + x * scale * .5 * perspective, height * .5 - y * scale * .5 * perspective, z

    def axis_point(self, x, y, z):
        return self.project({"x": x, "y": y, "z": z})[:2]

    AXES = (((-1.12, 0, 0), (1.12, 0, 0), "Warmth  R−B"),
            ((0, -1.12, 0), (0, 1.12, 0), "Luminance"),
            ((0, 0, -1.12), (0, 0, 1.12), "Edge density"))

    def draw_axis(self, start, end, label):
        """Draw one axis line and return where its label belongs.

        The label itself is drawn in a later pass: the thumbnails are dense
        enough that anything under them becomes unreadable.
        """
        a, b = self.axis_point(*start), self.axis_point(*end)
        self.scene.create_line(*a, *b, fill=COLOR.axis, width=BORDER["thin"], arrow=tk.LAST)
        return (b[0] + SPACE["sm"], b[1] - SPACE["sm"] - 1, label)

    def draw_axis_labels(self, placements):
        """Axis labels, above the collection and set on their own small chips.

        A chip rather than an outline: over a thousand thumbnails, haloed text
        still breaks up, and a quiet filled shape is what reads.
        """
        font = theme.font("label")
        pad_x, pad_y = SPACE["sm"], SPACE["xs"]
        for x, y, label in placements:
            width = font.measure(label)
            height = font.metrics("linespace")
            theme.rounded_rect(self.scene, x - pad_x, y - height / 2 - pad_y,
                               x + width + pad_x, y + height / 2 + pad_y,
                               radius=RADIUS["sm"], fill=COLOR.stage,
                               outline=COLOR.axis, width=BORDER["hairline"])
            self.scene.create_text(x, y, text=label, fill=COLOR.axis_label,
                                   anchor="w", font=font)

    def draw_scene(self):
        if not hasattr(self, "small_photos"):
            return
        self.scene.delete("all")
        self.scene_refs = []
        axis_labels = [self.draw_axis(start, end, label) for start, end, label in self.AXES]
        self.projected = []
        for index, work in enumerate(self.works):
            x, y, z = self.project(work)
            self.projected.append((z, x, y, index, work))
        positions = {int(row[4]["id"]): (row[1], row[2]) for row in self.projected}
        ordered = sorted(self.projected, key=lambda row: row[0])
        for row in ordered:
            work_id = int(row[4]["id"])
            if work_id == self.hovered:
                continue
            if self.keyword_matches is None or work_id not in self.keyword_matches:
                self.draw_work(row)
        for first, second in self.keyword_edges:
            if first in positions and second in positions:
                self.scene.create_line(*positions[first], *positions[second],
                                       fill=COLOR.stage, width=6)
                self.scene.create_line(*positions[first], *positions[second],
                                       fill=COLOR.link, width=BORDER["thin"])
        for row in ordered:
            work_id = int(row[4]["id"])
            if work_id != self.hovered and self.keyword_matches is not None and work_id in self.keyword_matches:
                self.draw_work(row)
        # A hovered work is always the final scene item, so its enlarged
        # thumbnail stays above every artwork, axis, and keyword connection.
        if self.hovered is not None:
            hovered_row = next((row for row in self.projected if int(row[4]["id"]) == self.hovered), None)
            if hovered_row is not None:
                self.draw_work(hovered_row)
        self.draw_axis_labels(axis_labels)
        self.draw_stage_hint()

    def draw_stage_hint(self):
        """A quiet, single-line affordance hint anchored to the stage corner."""
        height = max(1, self.scene.winfo_height())
        if self.mode == "remove":
            text = "Click works to exclude them from the next run · Right-drag to orbit · Scroll to zoom"
        else:
            text = "Click a work to inspect it · Right-drag to orbit · Scroll to zoom"
        self.scene.create_text(SPACE["xl"], height - SPACE["xl"], text=text,
                               fill=COLOR.text_muted, anchor="sw", font=theme.font("caption"))

    def draw_work(self, row):
        _z, x, y, index, work = row
        is_match = self.keyword_matches is not None and int(work["id"]) in self.keyword_matches
        is_dim = self.keyword_matches is not None and not is_match
        source_tile = self.dim_source_tiles[index] if is_dim else self.source_tiles[index]
        photo = self.dim_photos[index] if is_dim else self.small_photos[index]
        if self.hovered == int(work["id"]):
            photo = ImageTk.PhotoImage(source_tile.resize((88, 88), Image.Resampling.LANCZOS))
            self.scene_refs.append(photo)
            self.scene.create_rectangle(x - 46, y - 46, x + 46, y + 46,
                                        outline=COLOR.border_strong, width=BORDER["thin"])
        elif self.selected == int(work["id"]):
            photo = ImageTk.PhotoImage(source_tile.resize((42, 42), Image.Resampling.LANCZOS))
            self.scene_refs.append(photo)
            self.scene.create_rectangle(x - 24, y - 24, x + 24, y + 24,
                                        outline=COLOR.accent, width=BORDER["thin"])
        elif is_match:
            photo = ImageTk.PhotoImage(self.source_tiles[index].resize((28, 28), Image.Resampling.LANCZOS))
            self.scene_refs.append(photo)
        if is_match:
            self.scene.create_rectangle(x - 16, y - 16, x + 16, y + 16,
                                        outline=COLOR.accent, width=BORDER["thick"])
        self.scene.create_image(x, y, image=photo)
        # Keyword search is a visual focus state: unmatched works are dimmed,
        # but no removal marks are shown while that state is active.
        if self.keyword_matches is None and int(work["id"]) in self.removed:
            # A light halo keeps the exclusion mark legible over any artwork.
            self.scene.create_line(x - 7, y - 7, x + 7, y + 7, fill=COLOR.stage, width=3)
            self.scene.create_line(x + 7, y - 7, x - 7, y + 7, fill=COLOR.stage, width=3)
            self.scene.create_line(x - 7, y - 7, x + 7, y + 7, fill=COLOR.mark, width=BORDER["thin"])
            self.scene.create_line(x + 7, y - 7, x - 7, y + 7, fill=COLOR.mark, width=BORDER["thin"])

    def nearest(self, x, y):
        candidate, distance = None, 14
        for _z, px, py, _index, work in self.projected:
            current = math.hypot(px - x, py - y)
            if current < distance:
                candidate, distance = int(work["id"]), current
        return candidate

    # -- pointer -------------------------------------------------------
    def pointer_down(self, event):
        self.dragging = True
        self.last_pointer = self.down_pointer = (event.x, event.y)

    def left_down(self, event):
        self.down_pointer = (event.x, event.y)

    def pointer_drag(self, event):
        dx, dy = event.x - self.last_pointer[0], event.y - self.last_pointer[1]
        self.rot_y += dx * .007
        self.rot_x += dy * .007
        self.last_pointer = (event.x, event.y)
        self.draw_scene()

    def pointer_up(self, event):
        moved = math.hypot(event.x - self.down_pointer[0], event.y - self.down_pointer[1])
        self.dragging = False
        if moved >= 7:
            return
        hit = self.nearest(event.x, event.y)
        if hit is None:
            return
        self.selected = hit
        if self.mode == "remove":
            if hit in self.removed:
                self.removed.remove(hit)
            else:
                self.removed.add(hit)
            self.update_removed_label()
        else:
            self.open_artwork_detail(hit)
        self.draw_scene()

    def right_up(self, _event):
        self.dragging = False

    def pointer_move(self, event):
        if not self.dragging:
            self.set_hover(self.nearest(event.x, event.y))

    def set_hover(self, object_id):
        if object_id != self.hovered:
            self.hovered = object_id
            self.draw_scene()

    def mouse_wheel(self, event):
        factor = math.exp(event.delta / 120 * .12)
        self.set_scene_zoom(self.scene_zoom * factor)

    def mean_wheel(self, event):
        self.set_output_zoom(self.output_zoom * math.exp(event.delta / 120 * .12))

    def set_scene_zoom(self, value):
        self.scene_zoom = max(.45, min(3, value))
        self.space_zoom_label.configure(text=f"{self.scene_zoom:.0%}")
        self.draw_scene()

    def set_output_zoom(self, value):
        self.output_zoom = max(.45, min(2.5, value))
        self.output_zoom_label.configure(text=f"{self.output_zoom:.0%}")
        if self.current_epoch is not None:
            self.load_epoch(self.current_epoch)

    # -- modes and selection -------------------------------------------
    def set_mode(self, mode):
        self.mode = mode
        removing = mode == "remove"
        self.remove_button.configure(style="SegmentActive.TButton" if removing else "Segment.TButton")
        self.inspect_button.configure(style="Segment.TButton" if removing else "SegmentActive.TButton")
        self.scene.configure(cursor="X_cursor" if removing else "hand2")
        self.message.configure(
            text="Remove mode · click artworks to exclude them from the next run."
            if removing else "Inspect mode · click an artwork to open it with its museum record.")
        self.draw_scene()

    def update_removed_label(self):
        count = len(self.removed)
        self.removed_label.configure(text=f"{count:,} removed" if count != 1 else "1 removed")
        self.clear_button.state(["!disabled"] if count else ["disabled"])

    def clear_removed(self):
        self.removed.clear()
        self.selected = None
        self.update_removed_label()
        self.set_mode("inspect")
        self.draw_scene()

    def open_artwork_detail(self, object_id):
        record = self.record_map.get(int(object_id))
        if not record:
            return
        path = IMAGES / record["localfile"]
        if not path.exists():
            messagebox.showerror("Artwork unavailable",
                                 "The local artwork image is unavailable.", parent=self.root)
            return
        if self.detail_window is not None and self.detail_window.winfo_exists():
            self.detail_window.destroy()
        image = Image.fromarray(square(path, 512))
        self.detail_window = ArtworkViewer(self.root, record, image)

    # -- keyword search ------------------------------------------------
    def prompt_matches(self):
        return training.matching_ids(self.prompt.get())

    def clear_search(self):
        self.prompt.set("")
        self.search_entry.focus_set()

    def update_keyword_highlights(self, _event=None):
        query = self.prompt.get().strip()
        self.keyword_matches = self.prompt_matches() if query else None
        self.keyword_edges = self.build_keyword_edges(self.keyword_matches)
        self._update_search_status(query)
        self.draw_scene()

    def _update_search_status(self, query):
        if not hasattr(self, "search_status"):
            return
        if query:
            self.search_placeholder.place_forget()
        else:
            self.search_placeholder.place(in_=self.search_entry, relx=0, rely=0.5,
                                          x=SPACE["sm"] + 1, anchor="w")
        self.search_clear.state(["!disabled"] if query else ["disabled"])
        total = len(self.works)
        if not query:
            self.search_status.configure(
                text=f"{total:,} works in view. Keywords highlight matches and connect them "
                     f"by feature distance.", fg=COLOR.text_muted)
            return
        count = len(self.keyword_matches or ())
        if not count:
            self.search_status.configure(text="No works match these keywords. Generating will "
                                              "not be possible until a keyword matches.",
                                         fg=COLOR.text)
            return
        self.search_status.configure(
            text=f"{count:,} of {total:,} works highlighted · Generate trains on these only.",
            fg=COLOR.text_secondary)

    def build_keyword_edges(self, ids):
        if not ids or len(ids) < 2:
            return []
        points = {object_id: self.by_id[object_id] for object_id in ids if object_id in self.by_id}
        if len(points) < 2:
            return []
        remaining = set(points)
        first = remaining.pop()
        connected = {first}
        edges = []

        def distance(a, b):
            return sum((points[a][axis] - points[b][axis]) ** 2 for axis in ("x", "y", "z"))

        best = {item: (distance(first, item), first) for item in remaining}
        while remaining:
            item = min(remaining, key=lambda candidate: best[candidate][0])
            _score, parent = best.pop(item)
            remaining.remove(item)
            connected.add(item)
            edges.append((parent, item))
            for candidate in remaining:
                score = distance(item, candidate)
                if score < best[candidate][0]:
                    best[candidate] = (score, item)
        return edges

    def update_structure_label(self, value):
        self.structure_label.configure(text=f"{value:.0%}")
        self._update_model_note(value)

    def _update_model_note(self, value):
        if hasattr(self, "model_note"):
            self.model_note.configure(
                text=f"Structured mean input: {1 - float(value):.0%} arithmetic mean "
                     f"+ {float(value):.0%} feature-centroid representative.")

    # -- training ------------------------------------------------------
    def prompt_and_train(self):
        query = self.prompt.get().strip()
        if query:
            matches = self.prompt_matches()
            self.keyword_matches = matches
            self.keyword_edges = self.build_keyword_edges(matches)
            self._update_search_status(query)
            if not matches:
                messagebox.showerror("No matching artworks",
                                     "The keywords do not match the local museum metadata.",
                                     parent=self.root)
                return
            # Keep keyword filtering separate from the user's manual removals.
            # The complement is excluded from this training run without adding
            # crosses or changing the persistent REMOVED counter.
            run_removed = ({int(work["id"]) for work in self.works} - matches) | self.removed
        else:
            self.keyword_matches = None
            self.keyword_edges = []
            self._update_search_status(query)
            run_removed = set(self.removed)
        self.draw_scene()
        self.start_training(run_removed)

    def start_training(self, run_removed=None):
        try:
            epochs = max(1, min(200, int(self.epochs.get())))
            excluded = self.removed if run_removed is None else run_removed
            training.start(sorted(excluded), epochs, self.prompt.get(), self.structure.get())
            self.follow_latest = True
            self.status_badge.set("Queued", "active")
            self.message.configure(text="Training queued.", fg=COLOR.text_secondary)
        except Exception as error:
            messagebox.showerror("Unable to start training", str(error), parent=self.root)

    def stop_training(self):
        if not training.stop():
            self.message.configure(text="No training run is active in this desktop process.",
                                   fg=COLOR.text)

    def run_path(self, state):
        return OUT / "live" / str(state.get("run_id", ""))

    # -- results -------------------------------------------------------
    def _redraw_mean(self):
        """Paint either the current epoch preview or a calm empty state."""
        if not hasattr(self, "mean_canvas"):
            return
        width = max(1, self.mean_canvas.winfo_width())
        height = METRIC["mean_preview_height"]
        self.mean_canvas.delete("all")
        if self.output_photo is None:
            theme.rounded_rect(self.mean_canvas, 1, 1, width - 1, height - 1,
                               radius=RADIUS["md"], fill=COLOR.surface_sunken,
                               outline=COLOR.border, width=BORDER["thin"], dash=(3, 4))
            self.mean_canvas.create_text(width / 2, height / 2 - 8, text="No mean image yet",
                                         fill=COLOR.text_secondary, font=theme.font("body"))
            self.mean_canvas.create_text(width / 2, height / 2 + 12,
                                         text="Press Generate to train a mean",
                                         fill=COLOR.text_muted, font=theme.font("caption"))
            return
        self.mean_canvas.create_image(width / 2, height / 2, image=self.output_photo)

    def load_epoch(self, epoch):
        state = training.status()
        path = self.run_path(state) / "frames" / f"epoch_{int(epoch):03d}.png"
        if not path.exists():
            return
        size = int((METRIC["mean_preview_height"] - SPACE["sm"]) * self.output_zoom)
        image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        self.output_photo = ImageTk.PhotoImage(image)
        self.current_epoch = int(epoch)
        self.follow_latest = False
        self._redraw_mean()
        self.mean_eyebrow.set_hint(f"epoch {self.current_epoch:03d}")

    def rebuild_epoch_view(self, state):
        frames = state.get("frames", [])
        if not frames:
            self.epochs_view.show(None)
            return
        columns, tile, gap = 3, 96, SPACE["sm"]
        rows = math.ceil(len(frames) / columns)
        board = Image.new("RGB", (columns * (tile + gap), rows * (tile + 20 + gap)), COLOR.surface)
        for i, epoch in enumerate(frames):
            path = self.run_path(state) / "frames" / f"epoch_{epoch:03d}.png"
            if path.exists():
                image = Image.open(path).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
                board.paste(image, ((i % columns) * (tile + gap), (i // columns) * (tile + 20 + gap)))
        self.epochs_view.show(board)
        self.epochs_view.canvas.configure(cursor="hand2")
        self.epochs_view.canvas.bind("<Button-1>",
                                     lambda event: self.epoch_click(event, frames, tile, gap, columns))

    def epoch_click(self, event, frames, tile, gap, columns):
        y = self.epochs_view.canvas.canvasy(event.y) - SPACE["sm"]
        column = int((event.x - SPACE["sm"]) // (tile + gap))
        row = int(y // (tile + 20 + gap))
        index = row * columns + column
        if 0 <= index < len(frames):
            self.load_epoch(frames[index])

    def rebuild_results_view(self, state):
        path = self.run_path(state) / "generated_atlas.jpg"
        if not path.exists():
            self.results_view.show(None)
            return
        source = Image.open(path).convert("RGB")
        self.result_atlas = source
        records_path = self.run_path(state) / "records.json"
        self.result_ids = ([int(item["id"]) for item in
                            json.loads(records_path.read_text(encoding="utf-8"))]
                           if records_path.exists() else [])
        count = int(state.get("generated", state.get("artworks", 0)))
        source_columns, columns, tile = 32, 5, 64
        board = Image.new("RGB", (columns * tile, math.ceil(max(count, 1) / columns) * tile),
                          COLOR.surface)
        for index in range(count):
            sx, sy = (index % source_columns) * tile, (index // source_columns) * tile
            board.paste(source.crop((sx, sy, sx + tile, sy + tile)),
                        ((index % columns) * tile, (index // columns) * tile))
        self.results_view.show(board)
        self.results_view.canvas.configure(cursor="hand2")
        self.results_view.canvas.bind("<Button-1>", self.result_click)

    def result_click(self, event):
        column = int((self.results_view.canvas.canvasx(event.x) - SPACE["sm"]) // 64)
        row = int((self.results_view.canvas.canvasy(event.y) - SPACE["sm"]) // 64)
        index = row * 5 + column
        if not 0 <= index < len(self.result_ids):
            return
        object_id = self.result_ids[index]
        record = self.record_map.get(object_id)
        if not record:
            return
        original = Image.fromarray(square(IMAGES / record["localfile"], 512))
        state = training.status()
        path = self.run_path(state) / "generated" / f"{object_id}.jpg"
        if path.exists():
            generated = Image.open(path).convert("RGB")
        elif self.result_atlas is not None:
            sx, sy = (index % 32) * 64, (index // 32) * 64
            generated = self.result_atlas.crop((sx, sy, sx + 64, sy + 64)).resize(
                (512, 512), Image.Resampling.LANCZOS)
        else:
            return
        ComparisonViewer(self.root, record.get("title") or "Untitled", original, generated)

    def export_results(self):
        state = training.status()
        run = self.run_path(state)
        frames = state.get("frames") or []
        if not run.exists() or not frames:
            messagebox.showerror("Nothing to export", "Generate a result before exporting.",
                                 parent=self.root)
            return
        chosen = filedialog.askdirectory(title="Choose an export folder", parent=self.root,
                                         mustexist=True)
        if not chosen:
            return
        base = Path(chosen) / f"WhoseMean_{state.get('run_id', 'results')}"
        destination = base
        copy_number = 2
        while destination.exists():
            destination = Path(f"{base}_{copy_number}")
            copy_number += 1
        destination.mkdir(parents=True)
        latest_epoch = max(int(epoch) for epoch in frames)
        shutil.copy2(run / "frames" / f"epoch_{latest_epoch:03d}.png",
                     destination / "pix2pix_mean.png")
        for filename in ("mean_target.png", "structured_input.png", "representative.png",
                         "generated_atlas.jpg", "records.json"):
            source = run / filename
            if source.exists():
                shutil.copy2(source, destination / filename)
        for folder in ("frames", "generated"):
            source = run / folder
            if source.exists():
                shutil.copytree(source, destination / folder)
        export_state = {key: value for key, value in state.items() if key not in {"frames"}}
        export_state["exported_epoch"] = latest_epoch
        (destination / "export_info.json").write_text(
            json.dumps(export_state, ensure_ascii=False, indent=2), encoding="utf-8")
        generated_count = (len(list((destination / "generated").glob("*.jpg")))
                           if (destination / "generated").exists() else 0)
        messagebox.showinfo("Export complete",
                            f"Saved the final mean, {len(frames)} epoch images, and "
                            f"{generated_count} generated artworks to:\n{destination}",
                            parent=self.root)

    def open_large_result(self):
        state = training.status()
        frames = state.get("frames") or []
        if not frames:
            messagebox.showerror("No mean result",
                                 "Generate a result before opening the large view.",
                                 parent=self.root)
            return
        epoch = self.current_epoch if self.current_epoch in frames else max(frames)
        path = self.run_path(state) / "frames" / f"epoch_{int(epoch):03d}.png"
        if not path.exists():
            messagebox.showerror("Result unavailable", "The selected epoch image is unavailable.",
                                 parent=self.root)
            return
        ResultViewer(self.root, f"pix2pix mean · epoch {int(epoch):03d}",
                     Image.open(path).convert("RGB"))

    # -- polling -------------------------------------------------------
    def poll_training(self):
        try:
            state = training.status()
            name = str(state.get("state", "idle"))
            active = name in self.ACTIVE_STATES
            label, tone = self.STATE_TONES.get(name, (name.title(), "idle"))
            self.status_badge.set(label, tone)
            self.generate_button.state(["disabled"] if active else ["!disabled"])
            self.stop_button.state(["!disabled"] if active else ["disabled"])
            self.epoch_spin.state(["disabled"] if active else ["!disabled"])
            self.structure_slider.configure(cursor="watch" if active else "hand2")
            progress = float(state.get("progress", 0) or 0)
            self.progress.set(progress)
            self.progress_value.configure(text=f"{progress:.0%}")
            self.message.configure(text=state.get("message", "Ready."),
                                   fg=COLOR.text if tone == "error" else COLOR.text_secondary)
            history = state.get("history") or []
            loss = f"{history[-1]['train_l1']:.5f}" if history else "—"
            batch = state.get("batch", 0)
            total = state.get("total_batches", state.get("artworks", len(self.works)))
            rows = [("Artwork", f"{batch:,} / {total:,}"), ("Train L1", loss)]
            epoch_number = state.get("epoch")
            if epoch_number:
                rows.insert(0, ("Epoch", f"{epoch_number:,} / {state.get('epochs', '—'):,}"
                                if isinstance(state.get("epochs"), int)
                                else f"{epoch_number:,}"))
            anchor = state.get("representative_title")
            if anchor:
                rows.append(("Anchor", anchor[:24]))
            device = state.get("device")
            if device:
                rows.append(("Device", str(device).upper()))
            self.stats.set_rows(rows)
            frames = tuple(state.get("frames", []))
            self.open_large_button.state(["!disabled"] if frames else ["disabled"])
            self.export_button.state(["!disabled"] if frames else ["disabled"])
            if frames != self.last_frames:
                self.last_frames = frames
                self.rebuild_epoch_view(state)
                if self.follow_latest or self.current_epoch is None:
                    self.load_epoch(frames[-1])
                    self.follow_latest = True
            if state.get("has_generated_atlas") and state.get("run_id") != self.last_result_run:
                self.last_result_run = state.get("run_id")
                self.rebuild_results_view(state)
        except Exception as error:
            self.status_badge.set("Error", "error")
            self.message.configure(text=f"Status error: {error}", fg=COLOR.text)
        self.root.after(MOTION["poll"], self.poll_training)


def smoke_test():
    document = json.loads(WORKS_PATH.read_text(encoding="utf-8"))
    atlas = Image.open(io.BytesIO(thumbnail_atlas()))
    state = training.status()
    result = {"works": len(document["works"]), "atlas": list(atlas.size), "training_state": state.get("state")}
    if result["works"] != 1000 or result["atlas"] != [1280, 3200]:
        raise SystemExit(f"Unexpected desktop data: {result}")
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--ui-smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return
    root = tk.Tk()
    studio = Studio(root)
    if args.ui_smoke_test:
        root.update_idletasks()
        root.update()
        studio.prompt.set("tree")
        studio.update_keyword_highlights()
        print(json.dumps({"works": len(studio.works), "canvas_items": len(studio.scene.find_all()),
                          "tree_matches": len(studio.prompt_matches()),
                          "keyword_edges": len(studio.keyword_edges),
                          "window": [root.winfo_width(), root.winfo_height()]}))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
