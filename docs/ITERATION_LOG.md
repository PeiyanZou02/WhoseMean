# Iteration log

A running record of each iteration, with the commit hashes needed to roll
back. Newest iteration first.

Rollback quick reference:

| Goal | Command |
| --- | --- |
| Look at an old version without changing anything | `git checkout <hash>` (then `git checkout iteration1` to come back) |
| Undo one specific commit, keep everything after it | `git revert <hash>` |
| Move the whole branch back to a point, discarding what came after | `git reset --hard <hash>` |
| Abandon the iteration entirely | `git checkout main` — `main` is untouched |

Note: annotated tags could not be pushed from the working session (the
remote rejects tag refs with HTTP 403), so commit hashes below are the
authoritative anchors. `a870ad8` is also the tip of `main`.

---

## Iteration 1.1 — dark, Apple-influenced interface

Goal: keep the monochrome constraint but move the studio onto a black
ground and borrow Apple's HIG proportions.

| Commit | Change | Status |
| --- | --- | --- |
| `cfc04b0` | dark theme across `theme.py`, `app.py`, `data.py` | complete |

What changed: Apple's system greys at the dark end of the ramp, elevation
by lightening rather than by shadow, white reserved as the single
strongest emphasis, San Francisco leading both font stacks, HIG-named
type roles with more generous leading, macOS control radii, and sentence
case in place of letterspaced capitals throughout.

Three defects fixed in the same pass: axis labels were drawn under the
thumbnails and are now drawn last on their own chips; the default window
was wider than a 1440px laptop screen and is now clamped to the display;
and thumbnails letterboxed onto a near-white pad that ringed every
non-square work on a black stage — `square()` now takes an explicit pad
colour and only the atlas passes black, so recorded feature coordinates
are untouched.

Verified: `py_compile`; the palette re-checked mechanically (16 distinct
hex values, all chroma <= 12); the headless smoke harness run under
`xvfb-run` with a 1,000-work synthetic collection, exercising every
training state, both viewers, the sidebar scroll and the minimum window
size without exceptions; screenshots captured and inspected.

Still unverified: everything requiring torch or CUDA, and font resolution
on macOS and Windows (observed on Linux only, where the stack falls
through to DejaVu Sans — a Mac will pick up SF Pro and look closer to the
intended design than these screenshots do).

Known cosmetic inconsistency left in place: the artwork and comparison
viewers still letterbox onto the light feature pad, so a work opened at
512px keeps its pale surround. That is deliberate — those windows show
what the model is actually fed.

---

## Iteration 1 — branch `iteration1`

Goal: deepen the conceptual framing with HCI literature, replace the
from-scratch pix2pix backend with a current generative approach, and
rebuild the interface in a monochrome sans-serif design system.

| Commit | Change | Status |
| --- | --- | --- |
| `a870ad8` | v1.0 baseline — original pix2pix desktop studio (= `main`) | stable |
| `c236c8d` | `docs/ITERATION1_CODE_AUDIT.md` — engineering audit of v1.0 | complete |
| `90892bb` | `docs/ITERATION1_RESEARCH.md`, `docs/ITERATION1_MODEL_LANDSCAPE.md` | snapshot |
| `5500f65` | `src/whose_mean/theme.py` (new) + `src/whose_mean/app.py` rewrite | complete |
| `f5f2f7b` | `docs/ITERATION1_PLAN.md` + this log | snapshot |
| `eb74cd0` | revisions to research and model survey | snapshot |
| `HEAD` | final research deliverable + this log brought up to date | complete |

### Note on `5500f65`

Its message calls the UI work a mid-edit checkpoint. That was accurate at
the moment of commit but the snapshot in fact captured the finished
refactor — no further source changes followed. The history was left
as-is rather than rewritten, so an already-pushed commit stays valid for
anyone who has fetched it. Treat `5500f65` as the complete UI change.

### What landed

**Engineering audit** (`ITERATION1_CODE_AUDIT.md`, 657 lines). Findings
are tagged by how they were established: executed, calculated, or
read-only. The headline finding is that the network is a grayscale-to-RGB
colouriser trained with `GAN x1 + L1 x30 + edge x10` on individual works;
nothing in the objective optimises for a mean, and the true arithmetic
mean is already computed exactly and saved without training. Backlog is
41 items across P0/P1/P2, P0 estimated at ~8 person-days.

**UI redesign** (`theme.py` + `app.py`). A design-token module — one
13-step neutral ramp, semantic role palette, 10-role type scale, 8px
spacing rhythm, radii, metrics, motion — consumed throughout the
interface. Constraints verified mechanically: every hex literal in both
files has chroma <= 12, i.e. the palette carries no hue at all; the font
stacks are sans-only and resolved at runtime against
`tkinter.font.families()` with fallback through `TkDefaultFont`.

### Verification status

- `python -m py_compile src/whose_mean/*.py` passes.
- The UI was exercised headlessly under `xvfb-run` against a synthesised
  1,000-work dataset with a stubbed `torch`: construction, scene draw,
  keyword search and MST, mode switching, both zooms, the slider, all
  training states, empty states, the three secondary windows, sidebar
  scrolling, and a resize to the minimum window size.
- **Not verified**: anything needing torch or CUDA — real training, epoch
  frames, generated atlas, export of a real run. Those paths were only
  driven with faked status payloads. Font resolution was observed on
  Linux only. Stage rendering was judged with noise tiles, not real NGA
  thumbnails.

### Research findings

`ITERATION1_RESEARCH.md` and `ITERATION1_MODEL_LANDSCAPE.md` reach the
same diagnosis as the audit from a different direction: the central claim
is never computed. The network is a colouriser (luma to RGB); the
published output is `G(0.35 * luma(pixel_mean) + 0.65 * luma(medoid))`.
Nothing in the pipeline learns a likeness. Also noted: the feature space
is photometric rather than semantic (`saturation` and `mean_weight` are
computed and stored but never read), keyword search is substring matching
plus three hard-coded synonym sets, the "best" checkpoint is selected on
training L1 with no validation split, and the `STRUCTURE ANCHOR` label
disagrees with what it computes.

The recommendation is not to replace the pixel mean but to put it in a
series: show five means of the same selection at once, uncaptioned and
unranked — pixel mean, medoid, embedding centroid, LoRA-fine-tuned
likeness, and the collection inverted into a single token. Daston and
Galison's epistemic virtues map onto medoid / centroid / fine-tune, which
makes the citation operable rather than decorative. Galton's 1878
composite portraiture is added as the historical anchor for the pixel
mean. `model.py` and `training.py` are kept and relabelled "2017 mode"
rather than deleted.

Technical recommendation: SigLIP 2 So400m embeddings cached to disk as
the feature layer (runs on CPU, replaces the photometric axes, the
keyword alias table and the medoid in one move), then SDXL with
IP-Adapter, ControlNet and LCM/Turbo as the generative backend — chosen
for latency and ecosystem coexistence on one 12-24 GB card, not for
fidelity. Fallbacks are documented for low VRAM and CPU-only. Hosted APIs
are argued against as the primary path on conceptual grounds.

### Citation caveats

Section 11 of the research document lists eight items the agent could not
verify, including a journal venue, several incomplete author lists, and
one conference year. Figures in the model landscape are tagged verified /
secondary-source / estimate. No GPU was available, so the current build's
epoch wall-clock time is unmeasured; benchmarking it is the first item of
Phase 0. **Check the flagged citations before using any of this in a
submission.**
