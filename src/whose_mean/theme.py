"""Design tokens and Tk/ttk styling helpers for the Whose Mean studio.

The studio uses a strictly monochrome system: a single neutral tonal ramp,
one sans-serif family resolved at runtime, an 8px spacing rhythm, and a small
set of radii, border widths and motion durations. Nothing in the interface
carries hue; emphasis is expressed through value, weight, fill and border.

Everything here is pure standard library (tkinter/ttk) so the module can be
imported and partially exercised without a display.
"""
from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk


# ---------------------------------------------------------------------------
# Colour: one neutral ramp, 0 (white) to 1000 (black).
# ---------------------------------------------------------------------------

NEUTRAL: dict[int, str] = {
    0: "#ffffff",
    25: "#fbfbfa",
    50: "#f7f7f6",
    100: "#f1f1f0",
    150: "#e8e8e6",
    200: "#dededc",
    300: "#c9c9c6",
    400: "#adadaa",
    500: "#8c8c89",
    600: "#6d6d6a",
    700: "#52524f",
    800: "#3a3a38",
    900: "#232322",
    950: "#161615",
    1000: "#000000",
}


@dataclass(frozen=True)
class Palette:
    """Semantic colour roles. Only greys, by design."""

    # Surfaces
    app: str = NEUTRAL[50]            # window chrome behind everything
    surface: str = NEUTRAL[0]         # panels, cards, sidebars
    surface_sunken: str = NEUTRAL[100]  # wells, inactive tabs, tracks
    surface_raised: str = NEUTRAL[0]
    stage: str = NEUTRAL[0]           # the artwork canvas itself
    overlay: str = NEUTRAL[950]       # inverted chips and badges

    # Lines
    divider: str = NEUTRAL[150]       # low-contrast hairlines between regions
    border: str = NEUTRAL[200]        # resting control borders
    border_strong: str = NEUTRAL[300]  # hovered / emphasised borders
    focus: str = NEUTRAL[900]         # keyboard focus ring

    # Text
    text: str = NEUTRAL[900]          # high-contrast body and headings
    text_secondary: str = NEUTRAL[600]
    text_muted: str = NEUTRAL[500]
    text_disabled: str = NEUTRAL[400]
    text_inverse: str = NEUTRAL[0]

    # Emphasis ("accent" without hue)
    accent: str = NEUTRAL[900]
    accent_hover: str = NEUTRAL[800]
    accent_pressed: str = NEUTRAL[1000]
    accent_disabled: str = NEUTRAL[200]

    # Quiet interactive fills
    hover: str = NEUTRAL[100]
    pressed: str = NEUTRAL[150]
    selected: str = NEUTRAL[150]
    disabled_fill: str = NEUTRAL[100]

    # Artwork stage marks
    axis: str = NEUTRAL[300]
    axis_label: str = NEUTRAL[500]
    link: str = NEUTRAL[900]
    mark: str = NEUTRAL[900]


COLOR = Palette()


# ---------------------------------------------------------------------------
# Typography: sans-serif only, resolved at runtime with graceful fallbacks.
# ---------------------------------------------------------------------------

SANS_STACK: tuple[str, ...] = (
    "Inter",
    "Inter Display",
    "Segoe UI Variable Text",
    "Segoe UI",
    "SF Pro Text",
    "Helvetica Neue",
    "Noto Sans",
    "DejaVu Sans",
    "Liberation Sans",
    "Arial",
    "Helvetica",
)

# Only used for dense numeric readouts; a tabular sans is preferred when the
# platform has one, so the "mono" stack starts with sans faces that ship with
# tabular figures and only then falls back to a true monospace face.
NUMERIC_STACK: tuple[str, ...] = (
    "Inter",
    "Segoe UI Variable Text",
    "Segoe UI",
    "SF Pro Text",
    "DejaVu Sans",
    "Liberation Sans",
    "Arial",
    "DejaVu Sans Mono",
    "Consolas",
)


@dataclass(frozen=True)
class TypeSpec:
    """One step of the type scale. ``leading`` is a line height in pixels."""

    size: int
    weight: str = "normal"
    leading: int = 0
    slant: str = "roman"

    def line_height(self) -> int:
        return self.leading or int(round(self.size * 1.5))


# Sizes are Tk point sizes; the ramp is deliberately short so hierarchy stays
# legible: two display steps, one body step, two support steps.
TYPE: dict[str, TypeSpec] = {
    "display": TypeSpec(size=17, weight="bold", leading=24),
    "title": TypeSpec(size=13, weight="bold", leading=20),
    "heading": TypeSpec(size=11, weight="bold", leading=18),
    "body": TypeSpec(size=10, weight="normal", leading=17),
    "body_strong": TypeSpec(size=10, weight="bold", leading=17),
    "label": TypeSpec(size=9, weight="bold", leading=14),    # section eyebrows
    "caption": TypeSpec(size=9, weight="normal", leading=14),
    "micro": TypeSpec(size=8, weight="normal", leading=12),
    "numeric": TypeSpec(size=10, weight="normal", leading=16),
    "numeric_strong": TypeSpec(size=14, weight="bold", leading=20),
}

_NUMERIC_ROLES = {"numeric", "numeric_strong"}


# ---------------------------------------------------------------------------
# Spacing, radii, borders, metrics, motion.
# ---------------------------------------------------------------------------

# An 8px rhythm with 4px half-steps for dense control interiors.
SPACE: dict[str, int] = {
    "0": 0,
    "xxs": 2,
    "xs": 4,
    "sm": 8,
    "md": 12,
    "lg": 16,
    "xl": 24,
    "xxl": 32,
    "xxxl": 48,
}

RADIUS: dict[str, int] = {"none": 0, "sm": 4, "md": 8, "lg": 12, "pill": 999}

BORDER: dict[str, int] = {"hairline": 1, "thin": 1, "thick": 2}

METRIC: dict[str, int] = {
    "header_height": 64,
    "footer_height": 44,
    "sidebar_width": 380,
    "control_height": 32,
    "control_height_lg": 38,
    "slider_height": 28,
    "slider_track": 4,
    "slider_knob": 16,
    "progress_height": 6,
    "min_window_width": 1120,
    "min_window_height": 720,
    "default_window_width": 1480,
    "default_window_height": 880,
    "mean_preview_height": 208,
    "outputs_height": 300,
}

# Understated motion: short, few, and only used for state settling.
MOTION: dict[str, int] = {"instant": 0, "fast": 90, "base": 160, "slow": 240, "poll": 1000}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_resolved_families: dict[str, str] = {}
_fonts: dict[str, tkfont.Font] = {}


def _available_families() -> set[str]:
    try:
        return {name.lower() for name in tkfont.families()}
    except Exception:  # pragma: no cover - no Tk root available
        return set()


def resolve_family(stack: tuple[str, ...] = SANS_STACK, cache_key: str = "sans") -> str:
    """Return the first family in ``stack`` the platform actually has.

    Falls back to the family behind ``TkDefaultFont`` and finally to the
    literal name ``TkDefaultFont`` so callers always receive something Tk can
    render. Requires a Tk root to exist for the lookup to be meaningful.
    """
    cached = _resolved_families.get(cache_key)
    if cached:
        return cached
    available = _available_families()
    chosen = ""
    for family in stack:
        if family.lower() in available:
            chosen = family
            break
    if not chosen:
        try:
            chosen = tkfont.nametofont("TkDefaultFont").actual("family")
        except Exception:  # pragma: no cover - no Tk root available
            chosen = ""
    chosen = chosen or "TkDefaultFont"
    _resolved_families[cache_key] = chosen
    return chosen


def resolve_numeric_family() -> str:
    """Family for numeric readouts (a tabular sans when one is available)."""
    return resolve_family(NUMERIC_STACK, cache_key="numeric")


def font(role: str = "body") -> tkfont.Font:
    """Return (and memoise) the ``tkinter.font.Font`` for a type-scale role."""
    if role in _fonts:
        return _fonts[role]
    spec = TYPE.get(role, TYPE["body"])
    family = resolve_numeric_family() if role in _NUMERIC_ROLES else resolve_family()
    created = tkfont.Font(family=family, size=spec.size, weight=spec.weight, slant=spec.slant)
    _fonts[role] = created
    return created


def font_tuple(role: str = "body") -> tuple:
    """A plain ``(family, size, weight)`` tuple for widgets that prefer one."""
    spec = TYPE.get(role, TYPE["body"])
    family = resolve_numeric_family() if role in _NUMERIC_ROLES else resolve_family()
    if spec.weight == "normal" and spec.slant == "roman":
        return (family, spec.size)
    return (family, spec.size, spec.weight)


def reset_font_cache() -> None:
    """Forget resolved families/fonts (used by tests and after theme swaps)."""
    _resolved_families.clear()
    _fonts.clear()


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(channel)))) for channel in rgb)


def mix(first: str, second: str, amount: float) -> str:
    """Blend two hex colours; ``amount`` 0 returns ``first``, 1 returns ``second``."""
    amount = max(0.0, min(1.0, float(amount)))
    a, b = hex_to_rgb(first), hex_to_rgb(second)
    return rgb_to_hex(tuple(a[i] + (b[i] - a[i]) * amount for i in range(3)))


def rounded_points(x0: float, y0: float, x1: float, y1: float, radius: float) -> list[float]:
    """Polygon points approximating a rounded rectangle.

    Tk has no native rounded corners, so the shape is drawn honestly as a
    smoothed polygon whose corner points are duplicated to keep the straight
    edges straight.
    """
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    radius = max(0.0, min(float(radius), (x1 - x0) / 2, (y1 - y0) / 2))
    return [
        x0 + radius, y0,
        x1 - radius, y0, x1 - radius, y0, x1, y0,
        x1, y0 + radius, x1, y1 - radius, x1, y1,
        x1 - radius, y1, x1 - radius, y1, x0 + radius, y1,
        x0, y1, x0, y1 - radius, x0, y0 + radius,
        x0, y0, x0 + radius, y0, x0 + radius, y0,
    ]


def rounded_rect(canvas: tk.Canvas, x0, y0, x1, y1, radius=RADIUS["md"], **kwargs):
    """Draw a rounded rectangle on ``canvas`` and return the item id."""
    kwargs.setdefault("smooth", True)
    kwargs.setdefault("splinesteps", 24)
    return canvas.create_polygon(*rounded_points(x0, y0, x1, y1, radius), **kwargs)


def setup_styles(root: tk.Misc) -> ttk.Style:
    """Configure every ttk style the studio uses. Safe to call once per root."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:  # pragma: no cover - exotic Tk builds
        pass

    body = font("body")
    body_strong = font("body_strong")
    label = font("label")
    caption = font("caption")

    # --- containers ------------------------------------------------------
    style.configure("App.TFrame", background=COLOR.app)
    style.configure("Surface.TFrame", background=COLOR.surface)
    style.configure("Sunken.TFrame", background=COLOR.surface_sunken)

    for name, fg, typeface in (
        ("Display.TLabel", COLOR.text, font("display")),
        ("Title.TLabel", COLOR.text, font("title")),
        ("Heading.TLabel", COLOR.text, font("heading")),
        ("Body.TLabel", COLOR.text, body),
        ("BodyStrong.TLabel", COLOR.text, body_strong),
        ("Secondary.TLabel", COLOR.text_secondary, body),
        ("Muted.TLabel", COLOR.text_muted, caption),
        ("Eyebrow.TLabel", COLOR.text_muted, label),
        ("Numeric.TLabel", COLOR.text, font("numeric")),
    ):
        style.configure(name, background=COLOR.surface, foreground=fg, font=typeface)

    # --- buttons ---------------------------------------------------------
    common = dict(borderwidth=BORDER["thin"], focusthickness=1, relief="flat",
                  anchor="center", font=body_strong)

    style.configure("Primary.TButton", background=COLOR.accent, foreground=COLOR.text_inverse,
                    bordercolor=COLOR.accent, lightcolor=COLOR.accent, darkcolor=COLOR.accent,
                    focuscolor=COLOR.surface, padding=(SPACE["lg"], SPACE["sm"] + 2), **common)
    style.map(
        "Primary.TButton",
        background=[("disabled", COLOR.accent_disabled), ("pressed", COLOR.accent_pressed),
                    ("active", COLOR.accent_hover)],
        bordercolor=[("disabled", COLOR.accent_disabled), ("pressed", COLOR.accent_pressed),
                     ("active", COLOR.accent_hover)],
        lightcolor=[("disabled", COLOR.accent_disabled), ("pressed", COLOR.accent_pressed),
                    ("active", COLOR.accent_hover)],
        darkcolor=[("disabled", COLOR.accent_disabled), ("pressed", COLOR.accent_pressed),
                   ("active", COLOR.accent_hover)],
        foreground=[("disabled", COLOR.text_disabled)],
    )

    # clam paints a button's edge with light/dark/bordercolor, so an outlined
    # control needs all three set to the border tone rather than the fill.
    style.configure("Secondary.TButton", background=COLOR.surface, foreground=COLOR.text,
                    bordercolor=COLOR.border_strong, lightcolor=COLOR.border_strong,
                    darkcolor=COLOR.border_strong,
                    focuscolor=COLOR.focus, padding=(SPACE["md"], SPACE["sm"]), **common)
    style.map(
        "Secondary.TButton",
        background=[("disabled", COLOR.surface), ("pressed", COLOR.pressed), ("active", COLOR.hover)],
        lightcolor=[("disabled", COLOR.divider), ("pressed", COLOR.accent),
                    ("active", COLOR.border_strong)],
        darkcolor=[("disabled", COLOR.divider), ("pressed", COLOR.accent),
                   ("active", COLOR.border_strong)],
        bordercolor=[("disabled", COLOR.divider), ("pressed", COLOR.accent),
                     ("active", COLOR.border_strong)],
        foreground=[("disabled", COLOR.text_disabled)],
    )

    style.configure("Quiet.TButton", background=COLOR.surface, foreground=COLOR.text_secondary,
                    bordercolor=COLOR.surface, lightcolor=COLOR.surface, darkcolor=COLOR.surface,
                    focuscolor=COLOR.focus, padding=(SPACE["sm"], SPACE["xs"] + 1),
                    borderwidth=BORDER["thin"], focusthickness=1, relief="flat",
                    anchor="center", font=body)
    style.map(
        "Quiet.TButton",
        background=[("disabled", COLOR.surface), ("pressed", COLOR.pressed), ("active", COLOR.hover)],
        lightcolor=[("pressed", COLOR.border_strong), ("active", COLOR.hover)],
        darkcolor=[("pressed", COLOR.border_strong), ("active", COLOR.hover)],
        bordercolor=[("pressed", COLOR.border_strong), ("active", COLOR.hover)],
        foreground=[("disabled", COLOR.text_disabled), ("active", COLOR.text)],
    )

    # Segmented control: identical geometry, different value contrast.
    style.configure("Segment.TButton", background=COLOR.surface, foreground=COLOR.text_secondary,
                    bordercolor=COLOR.border_strong, lightcolor=COLOR.border_strong,
                    darkcolor=COLOR.border_strong,
                    focuscolor=COLOR.focus, padding=(SPACE["md"], SPACE["xs"] + 2), **common)
    style.map(
        "Segment.TButton",
        background=[("pressed", COLOR.pressed), ("active", COLOR.hover)],
        lightcolor=[("pressed", COLOR.accent), ("active", COLOR.border_strong)],
        darkcolor=[("pressed", COLOR.accent), ("active", COLOR.border_strong)],
        bordercolor=[("pressed", COLOR.accent), ("active", COLOR.border_strong)],
        foreground=[("disabled", COLOR.text_disabled), ("active", COLOR.text)],
    )
    style.configure("SegmentActive.TButton", background=COLOR.accent, foreground=COLOR.text_inverse,
                    bordercolor=COLOR.accent, lightcolor=COLOR.accent, darkcolor=COLOR.accent,
                    focuscolor=COLOR.surface, padding=(SPACE["md"], SPACE["xs"] + 2), **common)
    style.map(
        "SegmentActive.TButton",
        background=[("pressed", COLOR.accent_pressed), ("active", COLOR.accent_hover)],
        lightcolor=[("pressed", COLOR.accent_pressed), ("active", COLOR.accent_hover)],
        darkcolor=[("pressed", COLOR.accent_pressed), ("active", COLOR.accent_hover)],
        bordercolor=[("pressed", COLOR.accent_pressed), ("active", COLOR.accent_hover)],
    )

    # --- fields ----------------------------------------------------------
    style.configure("Search.TEntry", fieldbackground=COLOR.surface, background=COLOR.surface,
                    foreground=COLOR.text, insertcolor=COLOR.text, bordercolor=COLOR.border,
                    lightcolor=COLOR.border, darkcolor=COLOR.border, borderwidth=BORDER["thin"],
                    relief="flat", padding=(SPACE["sm"], SPACE["sm"] - 1), font=body)
    style.map(
        "Search.TEntry",
        bordercolor=[("focus", COLOR.focus), ("hover", COLOR.border_strong)],
        lightcolor=[("focus", COLOR.focus), ("hover", COLOR.border_strong)],
        darkcolor=[("focus", COLOR.focus), ("hover", COLOR.border_strong)],
        foreground=[("disabled", COLOR.text_disabled)],
    )

    style.configure("Studio.TSpinbox", fieldbackground=COLOR.surface, background=COLOR.surface,
                    foreground=COLOR.text, bordercolor=COLOR.border, lightcolor=COLOR.border,
                    darkcolor=COLOR.border, arrowcolor=COLOR.text_secondary,
                    arrowsize=11, borderwidth=BORDER["thin"], relief="flat",
                    padding=(SPACE["sm"], SPACE["xs"] + 1), font=font("numeric_strong"))
    style.map(
        "Studio.TSpinbox",
        bordercolor=[("focus", COLOR.focus), ("hover", COLOR.border_strong)],
        lightcolor=[("focus", COLOR.focus), ("hover", COLOR.border_strong)],
        darkcolor=[("focus", COLOR.focus), ("hover", COLOR.border_strong)],
        arrowcolor=[("disabled", COLOR.text_disabled), ("active", COLOR.text)],
        foreground=[("disabled", COLOR.text_disabled)],
    )

    # --- notebook --------------------------------------------------------
    style.configure("Studio.TNotebook", background=COLOR.surface, bordercolor=COLOR.divider,
                    lightcolor=COLOR.surface, darkcolor=COLOR.surface, borderwidth=1,
                    tabmargins=(0, 0, 0, 0))
    style.configure("Studio.TNotebook.Tab", background=COLOR.surface, foreground=COLOR.text_muted,
                    bordercolor=COLOR.divider, lightcolor=COLOR.surface, darkcolor=COLOR.surface,
                    focuscolor=COLOR.focus, padding=(SPACE["md"], SPACE["sm"]), borderwidth=0,
                    font=label)
    style.map(
        "Studio.TNotebook.Tab",
        background=[("selected", COLOR.surface), ("active", COLOR.hover)],
        foreground=[("selected", COLOR.text), ("active", COLOR.text_secondary)],
    )

    # --- scrollbars ------------------------------------------------------
    style.configure("Studio.Vertical.TScrollbar", background=COLOR.border,
                    troughcolor=COLOR.surface, bordercolor=COLOR.surface,
                    arrowcolor=COLOR.text_muted, lightcolor=COLOR.surface,
                    darkcolor=COLOR.surface, borderwidth=0, relief="flat", arrowsize=12, width=10)
    style.map("Studio.Vertical.TScrollbar",
              background=[("pressed", COLOR.text_muted), ("active", COLOR.border_strong)])
    style.configure("Studio.Horizontal.TScrollbar", background=COLOR.border,
                    troughcolor=COLOR.surface, bordercolor=COLOR.surface,
                    arrowcolor=COLOR.text_muted, lightcolor=COLOR.surface,
                    darkcolor=COLOR.surface, borderwidth=0, relief="flat", arrowsize=12, width=10)
    style.map("Studio.Horizontal.TScrollbar",
              background=[("pressed", COLOR.text_muted), ("active", COLOR.border_strong)])

    style.configure("Studio.Horizontal.TProgressbar", troughcolor=COLOR.surface_sunken,
                    background=COLOR.accent, bordercolor=COLOR.surface_sunken,
                    lightcolor=COLOR.accent, darkcolor=COLOR.accent, borderwidth=0, thickness=6)

    style.configure("Divider.TSeparator", background=COLOR.divider)
    return style


def apply_root(root: tk.Misc) -> ttk.Style:
    """Resolve fonts, set Tk's named fonts and configure ttk styles."""
    resolve_family()
    resolve_numeric_family()
    for named, role in (("TkDefaultFont", "body"), ("TkTextFont", "body"),
                        ("TkMenuFont", "body"), ("TkHeadingFont", "heading")):
        try:
            spec = TYPE[role]
            tkfont.nametofont(named, root=root).configure(
                family=resolve_family(), size=spec.size, weight=spec.weight)
        except Exception:  # pragma: no cover - platform dependent
            pass
    try:
        root.option_add("*Font", font("body"))
    except Exception:  # pragma: no cover
        pass
    return setup_styles(root)


__all__ = [
    "NEUTRAL", "Palette", "COLOR", "SANS_STACK", "NUMERIC_STACK", "TypeSpec", "TYPE",
    "SPACE", "RADIUS", "BORDER", "METRIC", "MOTION", "resolve_family",
    "resolve_numeric_family", "font", "font_tuple", "reset_font_cache", "hex_to_rgb",
    "rgb_to_hex", "mix", "rounded_points", "rounded_rect", "setup_styles", "apply_root",
]
