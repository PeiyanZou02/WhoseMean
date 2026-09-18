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
