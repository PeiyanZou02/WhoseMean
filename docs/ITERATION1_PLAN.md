# Whose Mean? — Iteration 1 Implementation Plan

Peiyan Zou · ADV 9672 · branch `iteration1` · September 2026

Companion documents: `ITERATION1_RESEARCH.md` (literature and conceptual argument), `ITERATION1_MODEL_LANDSCAPE.md` (model survey and recommendation).

> **Note on concurrency.** `src/whose_mean/app.py` is being edited in parallel by another agent. Everything below that touches `app.py` is written as a *specification to be merged*, not as a patch to apply blind. The new modules in §3 are designed so that the UI depends on them, never the reverse — `app.py` should end up importing `means`, `embeddings`, `provenance` and `session`, and nothing in those modules should import Tkinter.

---

## 1. The thesis of this iteration, in one paragraph

The current build claims that a model constructs a statistical likeness from a curated set, but computes an arithmetic pixel average and colourises it. Iteration 1 makes the claim true and makes it plural: **five different averages of the same selection, computed in five different ways, shown without a verdict** — pixel mean, medoid, embedding centroid decoded through a pretrained diffusion model, a LoRA fine-tuned on the selection, and the selection inverted into a single word. Alongside them, a permanent panel answering the title literally: *whose*. The technical work is in service of one interaction change: **editing the dataset must produce visible consequence in under a second**, because that is the only way a viewer can actually feel what inclusion and exclusion do.

---

## 2. Phased milestones

Effort is in **focused working days** for one person already familiar with the codebase. Multiply by 1.5–2 for calendar time.

| Phase | Deliverable | Effort | Blocks |
|---|---|---|---|
| **P0** Ground truth | Benchmarks, pinned deps, hardware decision, `DATASET.md` | **1 d** | everything |
| **P1** Feature layer | `embeddings.py`, `projection.py`; semantic space replaces photometric axes; semantic search replaces the alias table | **4 d** | P2, P3 |
| **P2** Whose? | `provenance.py` + the provenance panel + attribution ("which works pulled this mean") | **3 d** | — |
| **P3** Means abstraction | `means.py` with M1/M2 + retrieval-M3; mean-strategy selector in the UI; **this is where the piece becomes the piece** | **3 d** | P4 |
| **P4** Diffusion backend | `backends/diffusion.py`: SDXL + IP-Adapter + ControlNet + Turbo/LCM; generative M3 | **6 d** | P5 |
| **P5** Slow means | LoRA (M4) and textual inversion (M5) as background jobs; job queue refactor of `training.py` | **5 d** | — |
| **P6** Comparison & exhibition | Run history, branching, side-by-side, export, kiosk mode, wall text | **3 d** | P7 |
| **P7** Study & write-up | Pilot + 12–16 participants, analysis, paper/pictorial draft | **8 d** | — |
| | **Total** | **33 d** | |

**Order matters.** P1→P3 delivers a coherent, defensible artwork *with no diffusion model at all*. If the schedule collapses, stopping after P3 still yields something honest and exhibitable. P4/P5 are the upside, not the foundation. Do not invert this.

---

## 3. New modules

All new modules are pure Python + NumPy + PyTorch with **no Tkinter imports**, so they are testable headlessly and reusable if the UI is ever ported off Tk.

### 3.1 `src/whose_mean/embeddings.py`

```python
EMBED_SPACES = {
    "siglip2":  EmbedSpec(hf_id="google/siglip2-so400m-patch14-384", dim=1152, modal="joint"),
    "clip-h":   EmbedSpec(hf_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", dim=1024, modal="joint"),
    "dinov2-l": EmbedSpec(hf_id="facebook/dinov2-large", dim=1024, modal="image"),
}

class EmbeddingStore:
    """Immutable, L2-normalised embeddings for the whole collection, cached on disk."""
    space: str
    ids: np.ndarray          # int64  (n,)      object ids, canonical order
    matrix: np.ndarray       # float32 (n, d)   L2-normalised
    meta: dict               # model id, revision, preprocessing, created_at, sha

    def index_of(self, ids: Sequence[int]) -> np.ndarray: ...
    def centroid(self, ids, weights=None) -> np.ndarray:      # renormalised
    def medoid(self, ids) -> int:                             # argmin sum of cosine distance
    def knn(self, vec, k=12, within=None) -> list[tuple[int, float]]
    def contributions(self, ids, k=20) -> list[tuple[int, float]]
        """Works ranked by ||centroid(ids) - centroid(ids \ {i})||, i.e. leave-one-out pull."""
    def clusters(self, ids, k=3) -> dict[int, list[int]]      # k-means, for 'sub-means'

def ensure_embeddings(space: str = "siglip2", device=None,
                      progress: Callable[[int, int], None] | None = None) -> EmbeddingStore
    """Load data/nga/embeddings_{space}.npz, or compute and cache it."""
```

Cache format: `.npz` with `ids`, `matrix`, plus a JSON sidecar recording model id, revision, preprocessing (crop mode, resolution) and a hash of `records.jsonl`, so a stale cache is detected rather than silently used.

**Preprocessing decision:** embed a *centre crop*, not the letterboxed square. The `(238, 236, 230)` pad is a preprocessing artefact and should not be in the feature vector. Keep `square()` for display only.

`contributions()` is small but load-bearing: it is the operational answer to "whose mean?" and the direct application of Kulesza et al.'s explanatory debugging.

### 3.2 `src/whose_mean/projection.py`

```python
def layout_3d(store, ids=None, method="umap", seed=42) -> dict[int, tuple[float,float,float]]
AXIS_MODES = ["semantic", "photometric", "chronological"]
```

- `semantic` — UMAP (or PCA if `umap-learn` is unwanted as a dependency) of the embedding matrix to 3-D, scaled to [−1, 1]. Axes become *unlabelled*, which is correct: a semantic space has no honest axis names. Say so in the UI: `AXES · LEARNED SIMILARITY · NOT INTERPRETABLE`.
- `photometric` — the existing warmth / luminance / edge axes, preserved as a mode. Do not delete them; switching between the two is a legible demonstration of Offert & Bell's perceptual bias.
- `chronological` — x = parsed year, y = luminance, z = classification band. Cheap, and immediately makes the collection's temporal skew visible.

Keep layouts deterministic (`seed=42`) and cache them, so the space does not reshuffle between launches.

### 3.3 `src/whose_mean/means.py`

```python
@dataclass(frozen=True)
class Selection:
    included: tuple[int, ...]
    weights: dict[int, float] | None      # finally uses the dead 'mean_weight' field
    space: str
    def key(self) -> str: ...             # stable hash, for caching

@dataclass
class MeanResult:
    strategy: str                # "pixel" | "medoid" | "centroid" | "lora" | "token"
    image: PIL.Image.Image | None
    neighbours: list[int]        # works nearest the mean, for the always-on preview
    contributors: list[tuple[int, float]]
    elapsed_s: float
    provenance: dict             # model ids, steps, seed, conditioning scale, licence
    caption: None                # deliberately never set — see Gaver et al. 2003

class MeanStrategy(Protocol):
    name: str
    latency_class: Literal["instant", "fast", "slow"]
    requires_gpu: bool
    def compute(self, sel: Selection, ctx: Context) -> MeanResult: ...
    def preview(self, sel: Selection, ctx: Context) -> MeanResult: ...   # must be < 50 ms
```

Implementations: `PixelMean`, `Medoid`, `CentroidRetrieval` (no generation — the *k* nearest real works to the centroid), `IPAdapterCentroid`, `LoRALikeness`, `TokenInversion`, and `Pix2PixHistorical` wrapping the existing code path.

**`PixelMean` must be incremental.** Maintain a running `float64` sum and a per-pixel coverage count; removal is `sum -= I_i`, which is O(1) per edit and makes M1 genuinely instant after the first pass. Also fixes the letterbox problem: divide by the coverage mask rather than by N.

**Every strategy must implement `preview()`.** The UI always calls `preview()` synchronously on an edit and dispatches `compute()` to a worker. That single rule is what buys the interaction budget in §5.

### 3.4 `src/whose_mean/provenance.py`

```python
@dataclass
class SelectionProfile:
    n: int
    by_classification: Counter
    by_century: Counter                 # parsed from beginyear / displaydate
    by_attribution: Counter             # top artists + share
    unattributed: int
    by_creditline: Counter              # donor concentration — the real 'whose'
    top_donor_share: float
    median_year: int | None
    filters_applied: list[FilterStep]   # the upstream chain, with counts

def profile(ids: Sequence[int], records) -> SelectionProfile
def chain() -> list[FilterStep]
    # eligible -> accessioned -> open-access primary -> classification filter
    # -> sample(seed=42, limit) -> user removals
```

`collect()` in `data.py` already prints `eligible`, `requested`, `saved`, `failed` and `seed`. Persist that dict to `data/nga/collection_manifest.json` so `chain()` can read real numbers instead of recomputing them. Donor concentration via `creditline` is the highest-value, lowest-effort metric here: *"38 % of this mean was given by one family"* is exactly the sentence the piece is looking for.

### 3.5 `src/whose_mean/session.py`

```python
@dataclass
class Run:
    run_id: str
    parent_id: str | None        # branching, à la Picbreeder
    selection: Selection
    results: dict[str, MeanResult]
    profile: SelectionProfile
    created_at: datetime
    note: str

class SessionLog:
    def append(self, run: Run) -> None
    def branch_from(self, run_id: str) -> Selection
    def compare(self, a: str, b: str) -> ComparisonView
    def to_jsonl(self, path) -> None        # the study's event log
```

### 3.6 `src/whose_mean/backends/diffusion.py`

Lazy-loaded (import cost must not be paid at startup), with a hard capability probe.

```python
@dataclass
class Capabilities:
    device: str; vram_gb: float
    can_sdxl: bool; can_ipadapter: bool; can_controlnet: bool
    can_train_lora: bool; distilled: str | None       # "turbo" | "lcm" | None

def probe() -> Capabilities
class DiffusionBackend:
    def load(self, caps: Capabilities) -> None
    def from_image_embedding(self, emb, *, structure=None, steps=None, seed=42) -> Image
    def with_lora(self, path: Path) -> ContextManager
    def train_lora(self, ids, steps, on_progress) -> Path
    def invert_token(self, ids, steps, on_progress) -> Path
```

The capability probe drives graceful degradation (see the hardware table in `ITERATION1_MODEL_LANDSCAPE.md` §5): the app must launch and be fully usable on a machine with no GPU, with the generative strategies greyed out and a one-line explanation — not a `RuntimeError`, which is what `training.py:131` does today.

---

## 4. Changes to existing files

### `data.py`

| Change | Why | Effort |
|---|---|---|
| Persist the `collect()` summary dict to `data/nga/collection_manifest.json` | Feeds `provenance.chain()` | 15 min |
| Add `square_crop(path, size)` (centre crop, no pad) alongside `square()` | Embeddings and the pixel mean must not include the letterbox pad | 30 min |
| Add `beginyear`, `endyear`, `attributioninverted`, `visualbrowsertimespan` to the record dict if present in `objects.csv` | Provenance panel needs dates **[verify these column names against the live CSV]** | 1 h |
| Optionally join `constituents.csv` for artist nationality / life dates | Enables "whose" by geography **[schema unverified — see `ITERATION1_RESEARCH.md` §11]** | 2 h |
| Make `prepare_interactive` emit `aspect`, `source_w`, `source_h` per work | Lets the UI show how much of the pixel mean is padding | 20 min |
| Add `whose-mean-data embed` subcommand delegating to `embeddings.ensure_embeddings` | One-command setup | 30 min |

Keep the seed-42 sampling exactly as is. It is reproducible and honestly documented; changing it would invalidate the existing work.

### `model.py`

**No changes.** Keep pix2pix intact and reachable as a labelled "2017 mode". It is now content — the historical contrast that makes the new backends legible. Deleting it would be the single worst decision available.

### `training.py`

Refactor from "one training function" to "a job queue", preserving the existing `current.json` protocol so any UI polling code keeps working.

```python
@dataclass
class Job:
    job_id: str
    kind: Literal["pixel_mean", "embed", "sample", "lora", "invert", "pix2pix"]
    selection: Selection
    params: dict
    state: Literal["queued","running","complete","error","stopped"]
    progress: float
    message: str
    result_paths: dict[str, str]

def submit(job: Job) -> str
def status(job_id: str | None = None) -> dict     # back-compatible shape when job_id is None
def cancel(job_id: str) -> bool
```

Also:
- Replace the hard CUDA `RuntimeError` with a capability probe and a degraded path.
- Replace "best by training L1" with either a held-out split or — better — drop "best" entirely. On a 1000-image artwork there is no meaningful validation, and calling a memorisation minimum "best" is a small dishonesty in a project about honesty.
- Move `matching_ids()` and `CONCEPT_ALIASES` **out** of `training.py` into a new `search.py`, and reimplement as: SigLIP text-embedding similarity with an adjustable threshold, plus exact metadata field matching. Delete the `tree`/`dog`/`sardine` alias table.
- Rewrite `_write()`'s 100-attempt `PermissionError` retry loop as a documented Windows-specific helper (it is fine, but it deserves a comment explaining why it exists).

### `app.py` (specification — coordinate with the concurrent editor)

| Area | Change | Justification |
|---|---|---|
| Right panel header | Replace "PIX2PIX TRAINING" with a **MEAN** strategy selector: five radio-style buttons (`PIXEL · MEDOID · CENTROID · LEARNED · TOKEN`) plus a greyed `2017` for pix2pix | Daston & Galison's three virtues made operable; Weisz et al.'s *generative variability* |
| Mean panel | Show the active mean **plus** a strip of the 6 works nearest the current centroid, updated on every click | Sub-second feedback (Amershi et al.); retrieval preview is 10 ms |
| New panel: **WHOSE?** | `SelectionProfile` rendered compactly: n, century histogram, top 3 attributions with %, unattributed count, top donor with %, and the filter chain with counts | Crawford & Paglen; Denton et al.; this panel *is* the title |
| Replace the MODEL I/O tab | Live **"WHY THIS MEAN"**: top-20 contributor thumbnails from `contributions()`, and a delta view against the previous mean | Kulesza et al., explanatory debugging |
| Keyword field | Relabel from "CONCEPT KEYWORDS" to **FILTER**, and separate it clearly from any future prompt field. It filters the dataset; it does not condition the model | The current label implies text conditioning that does not exist |
| Search behaviour | Additive highlight + a "focus" toggle, instead of dimming 97 % of the collection to 30 % opacity by default | Whitelaw's generous interfaces; Dörk et al.'s explorability |
| Structure anchor | Re-implement honestly: label it **STRUCTURE · <medoid title>** and drive a ControlNet canny/depth map from the medoid, with 0 % = free generation, 100 % = rigid structural constraint | The current slider's label and its computation disagree |
| New: weight brush | Shift-drag to set a continuous per-work weight in [0, 2] instead of binary include/exclude | Activates the dead `mean_weight` field; widens the model object space (design-space survey, §4 of the research doc) |
| New: A/B strip | A horizontal filmstrip of previous runs; click to compare side by side; right-click to branch | Picbreeder's branching model; the piece's argument is comparative |
| New: interpolate | Save two selections as A and B; a slider morphs between `centroid(A)` and `centroid(B)` | Latent-space exploration (Sun et al.); the strongest single demonstration of the thesis |
| Axis mode toggle | `SEMANTIC / PHOTOMETRIC / CHRONOLOGICAL` | Offert & Bell's perceptual bias, made switchable |
| Captions | **None.** The five means are labelled by *operation*, never ranked or explained as "better" | Gaver et al., ambiguity as a resource |
| `smoke_test()` | Read the expected work count and atlas size from `works.json` instead of hard-coding `1000` and `[1280, 3200]` | Currently breaks for any `--limit != 1000` |
| Kiosk mode | `--exhibition`: fullscreen, auto-reset after 3 min idle, no file dialogs, event log to JSONL | Needed for P7 deployment |

---

## 5. Interaction budget (non-negotiable)

| Event | Budget | Mechanism |
|---|---|---|
| Hover / rotate / zoom | 16 ms | unchanged |
| Toggle one work | **< 100 ms** to updated pixel mean + medoid + nearest-neighbour strip + WHOSE? panel | incremental sum; NumPy on a (1000, 1152) matrix |
| Generative mean refresh | **< 2 s** | SDXL-Turbo / LCM, 1–4 steps, cached IP-Adapter conditioning |
| Full-quality render | < 15 s | 25–30 steps, user-invoked |
| Learned likeness (LoRA) | minutes–hours, **visibly** | background job, live sample every N steps, cancellable |

If the toggle budget cannot be met, the iteration has failed regardless of image quality. Measure it; do not assume it.

---

## 6. Evaluation strategy

### 6.1 Framing

This is **research through design** (Zimmerman et al., CHI 2007): the contribution is the artefact and the argument it embodies, not a controlled comparison against a baseline. Evaluating it as a productivity tool would be a category error — nobody needs a mean image.

### 6.2 Venue strategy

| Venue | Track | Fit | Recommendation |
|---|---|---|---|
| **DIS** | **Pictorials** | Very high — visual-first, argument-through-artefact, values annotated design rationale | **Primary target.** The five-means comparison is inherently pictorial |
| **Creativity & Cognition (C&C)** | **Artworks / Papers** | Very high — explicitly hosts critical computational art with a research claim | **Co-primary.** Submit the artwork, with the study as the paper |
| **CHI** | alt.chi | High — provocation-friendly | Ambitious stretch; alt.chi is a better fit than the Papers track |
| **CHI** | Interactivity | High | Good companion submission if the kiosk mode lands |
| **SIGGRAPH / SIGGRAPH Asia** | Art Papers | Medium–high | Good if the generative fidelity of P4/P5 is strong |
| **ISEA, Ars Electronica** | — | High | Exhibition reach; no peer-reviewed paper |
| **CSCW / IUI / NeurIPS** | — | Low | Wrong shape. Do not chase these |

### 6.3 Study design

**Part A — lab sessions, n = 14** (7 with art-historical or museum background, 7 without). 75 minutes, think-aloud, semi-structured.

1. Free exploration, 10 min, no task. *(Measures whether the generous overview affords wandering — Whitelaw, Dörk.)*
2. Directed task: "make a mean you would be willing to hang in the Gallery's entrance, and tell me why."
3. Counter-task: "now make one you think is dishonest."
4. Comparison probe: all five means for the same selection, side by side. "Which of these is the average? Can more than one be?"
5. Semi-structured interview on provenance, exclusion, and authorship.

**Part B — exhibition deployment, 2 weeks**, kiosk mode, JSONL event log (anonymous, no images of visitors, notice posted), plus a physical guestbook.

### 6.4 Measures

| Measure | Instrument | What it tests |
|---|---|---|
| **Articulated dataset awareness** (primary) | Pre/post free-text: "What does this collection over- and under-represent?" Coded by two raters for specificity and correctness; report agreement | Sengers et al.'s reflective design — the outcome that actually matters |
| **Interpretive plurality** (primary) | Count of distinct readings of the same mean across participants; thematic analysis | Gaver et al. — success is divergence, not consensus |
| Creativity support | **CSI** (Cherry & Latulipe 2014), six factors | Comparable to other CST work. **Caveat: report all six factors separately.** Exploration and Immersion are the load-bearing ones; "Results Worth Effort" will read oddly for a work whose point is that the result is contestable, and that should be discussed, not averaged away |
| Mental-model accuracy | 8-item post-task quiz: what does the strategy selector change? what does the structure slider do? whose data is this? | Kulesza et al. measured a 52 % understanding gain from explanation; test whether the WHY panel delivers anything similar |
| Behavioural | From the event log: works excluded, strategies switched per session, runs branched, interpolations used, time to first exclusion, revisits | Whether the fast loop is actually used as a loop |
| Latency check | Instrumented timings of the §5 budget | Engineering acceptance |

### 6.5 What would falsify the design

Be explicit about this, or the evaluation is theatre.

- Participants treat the generative means as "the good ones" and the pixel mean as broken → the framing failed; the labelling and ordering need rework.
- Nobody switches strategies → the selector is not legible as an argument; it has become a settings menu.
- Post-task dataset-awareness text is no more specific than pre-task → the WHOSE? panel is decoration.
- The removal gesture is used < 3 times per session → the loop is still too slow, or exclusion does not feel consequential.

### 6.6 Ethics

No personal data. The event log records interaction events and selections, not identities. Physical exhibition needs posted notice of logging. One content caution: Galton's lineage runs through composite *portraits of people*, and the NGA set contains portraits of identifiable historical persons; the piece should not silently produce composite faces without framing that history (see Lydon et al. 2024 in the research doc). If a portraits-only subset is offered, frame it explicitly.

---

## 7. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Target GPU is weaker than assumed** | Medium | High | Capability probe + graceful degradation from day one. P1–P3 run on CPU. Establish the actual exhibition machine in P0 |
| **IP-Adapter centroid produces generic "museum painting" regardless of selection** — the prior swamps the data | **High** | **High** | This is the most likely technical failure. Mitigate by: high conditioning scale; negative conditioning against the *global* collection centroid (`c_sel − λ·c_all`, which renders what makes the selection *distinct*); and, critically, by **always showing M1/M2 alongside**, so the failure is itself visible and on-topic rather than fatal |
| **LoRA on 1000 heterogeneous works learns nothing coherent** | Medium | Medium | Expected and interesting. Offer k-means sub-means (train on a cluster). Fix a step budget, not epochs. If it fails, that failure is a finding about what "the average of a thousand things" can mean |
| **Textual inversion does not converge for 1000 images** | High | Low | It is a stretch goal. Officially demonstrated on 3–5 images. Fall back to inverting the *medoid neighbourhood* (the 20 nearest works to the centroid) |
| **Merge conflicts with the concurrent `app.py` work** | High | Medium | Keep all new logic in new modules with no Tk dependency; hand the UI editor §3 and §4 as an interface contract |
| **Scope explosion from five means × three spaces × three axis modes** | High | High | Ship M1/M2/M3-retrieval first (P3) and treat P4/P5 as optional. Apply the cut list in §8 early and without sentiment |
| **Dependency weight** (diffusers, transformers, peft, umap-learn, accelerate) makes setup fragile | Medium | Medium | Two extras: `pip install -e .` for the CPU/embedding build, `pip install -e .[generate]` for diffusion. Pin exact versions. The base install must never require CUDA |
| **Licence drift** (a model relicensed mid-project) | Low | Medium | Record model id + revision + licence in every `MeanResult.provenance`. Prefer Apache-2.0 components. Re-check FLUX.2 klein 4B at the start of P4 |
| **The piece becomes a tech demo** | Medium | **High** | The captioning rule (§4) and the "no verdict" rule are structural defences. If a reviewer can tell which mean you think is right, you have over-designed |

---

## 8. What to cut if time runs short

In order. Cut from the bottom up and stop when the schedule fits.

1. **Textual inversion (M5).** Sharpest idea, least certain to work, easiest to describe in the paper as future work.
2. **Interpolation between two selections.** Beautiful, not load-bearing.
3. **The third embedding space (DINOv2/v3) and the axis-mode toggle.** Keep semantic only.
4. **The weight brush.** Binary include/exclude is a defensible position, and arguably a sharper one.
5. **LoRA (M4).** Painful to lose — it is the *trained judgment* term — but if it cannot be made stable, a single pre-baked LoRA on the full 1000 shown as a fixed reference preserves most of the argument at a fraction of the cost.
6. **The diffusion backend entirely (P4).** M1 + M2 + centroid-retrieval-M3 is a complete, honest, exhibitable work. It loses the "what a pretrained prior adds" argument, which is a real loss — but a smaller loss than shipping something slow and broken.

**Never cut, in any scenario:** the WHOSE? provenance panel, the contributor attribution, the sub-second edit loop, the retained pixel mean, and the no-captions rule. Those five are the work.

---

## 9. Immediate next actions (P0, one day)

1. Benchmark one epoch of the current pix2pix loop on the actual target GPU. Record seconds/epoch. Every latency claim in these documents depends on that one number.
2. Name the target hardware, including the exhibition machine.
3. Pin `requirements.txt` to exact versions; split out a `[generate]` extra.
4. Verify the NGA CSV column names for `beginyear` / `endyear` and inspect `constituents.csv` for nationality and life dates.
5. Write `docs/DATASET.md` following the *Collections as ML Data* checklist (Lee et al. 2025): the four upstream filters, the seed, the counts, the CC0 licence, the known exclusions.
6. Confirm the outstanding citations listed in `ITERATION1_RESEARCH.md` §11 before anything is submitted anywhere.
