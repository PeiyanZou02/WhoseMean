# Whose Mean? — Iteration 1 Engineering Audit

| | |
|---|---|
| **Repository** | `/home/user/WhoseMean` |
| **Branch / commit** | `iteration1` @ `a870ad8` ("Refine final desktop project structure") |
| **Scope** | `src/whose_mean/{app,data,model,training,__main__,__init__}.py`, `README.md`, `pyproject.toml`, `requirements.txt`, `OPEN_DESKTOP_STUDIO.cmd` |
| **Size** | 1,462 LOC across 6 modules (`app.py` 886, `training.py` 242, `data.py` 239, `model.py` 87) |
| **Audit date** | 2026-09-18 |

> **Provenance of line numbers.** This audit was performed against the committed tree at `a870ad8`, and every `file:line` reference below refers to **that commit**, not to the working tree.
>
> Other agents were editing `src/` concurrently. At the time each file was read, all tracked sources were byte-identical to `a870ad8` (verified by `md5sum` + `git status`). By the end of the audit an untracked `src/whose_mean/theme.py` had appeared and `src/whose_mean/app.py` had been modified. Neither change is covered here. To reconcile a finding against the working tree, use `git show a870ad8:<path>` or `git diff a870ad8 -- <path>`.

## Verification method

`torch`, `numpy`, `Pillow` and `tkinter` are **not installed** in the audit environment, and there is no display. Every finding is therefore tagged:

- **[V]** — verified by execution in this environment.
- **[C]** — verified by calculation from the source (arithmetic reproduced in a standalone script).
- **[R]** — established by reading the source only; not executed.

What I did execute:

| Check | Result |
|---|---|
| `python3 -m py_compile src/whose_mean/*.py` | **Pass** (Python 3.11.15) — no syntax errors. **[V]** |
| Dependency probe (`numpy`, `PIL`, `torch`, `tkinter`) | All absent — no runtime import or UI test possible. **[V]** |
| Re-implemented `training.matching_ids` on fixture works | Reproduced two real defects (see C-1, C-2). **[V]** |
| Re-implemented `Studio.build_keyword_edges` (MST) on 1,000 random points | **445 ms** per call, quadratic scaling (100→4.4 ms, 250→27.9 ms, 500→111 ms). **[V]** |
| Parameter/byte arithmetic for `Generator512` / `Discriminator512` | 16,751,555 / 693,985 params ≈ 67.0 MB / 2.8 MB fp32. **[C]** |
| `data.square()` letterbox-padding fraction for realistic NGA aspect ratios | 33.3 % pad for 3:2, 69.8 % for 10:3. **[C]** |
| Thumbnail atlas geometry (`20 × 64` × `50 × 64`) | 1280 × 3200 — matches the `smoke_test` assertion at `app.py:856`. **[C]** |
| Live fetch of NGA `objects.csv` / `published_images.csv` headers | Confirms column names, `sequence` is an integer string, `assistivetext` is machine-generated alt text, no licence column. **[V]** |
| `tkinter.Misc.unbind` implementation | Inspected CPython **3.12** (funcid-selective, fixed). CPython 3.11 not installed — see C-11. **[V/R]** |
| Repo scan for tests, lint, CI, type config | None exist. **[V]** |

---

## 1. Architecture map

### 1.1 Module responsibilities

| Module | LOC | Responsibility | Assessment |
|---|---|---|---|
| `data.py` | 239 | Path constants, NGA CSV download, candidate filtering, image download (`collect`), letterboxing (`square`), XYZ feature extraction (`prepare_interactive`), thumbnail atlas, CLI | Cohesive, the cleanest module. Mixes CLI, network, and numeric feature code, but at this size that is acceptable. |
| `model.py` | 87 | `Generator512` (6-scale U-Net), `Discriminator512` (PatchGAN), `edge_loss` | Clean, pure, fully type-annotated, no I/O. The best module in the repo. |
| `training.py` | 242 | Global training thread, on-disk status protocol, keyword matching, mean/anchor construction, training loop, result generation | **Tangled.** Holds process-global mutable state, does file I/O, ML, *and* text search (`matching_ids`, `training.py:31`) which has nothing to do with training. |
| `app.py` | 886 | All Tk widgets, 3-D projection, scene rendering, hit testing, keyword highlighting + MST, polling, epoch/result browsing, export, CLI entry | **Severely tangled.** `Studio` is a god object with ~40 instance attributes (`app.py:253-279`) owning UI, domain state, file I/O, image decoding, and export orchestration. |
| `__main__.py` / `__init__.py` | 5 / 3 | Entry point, version | Fine. |

### 1.2 Call graph (principal paths)

```
whose-mean-data collect      data.main:223
  └─ collect:98 ─ download_metadata:46 ─ download:33          (urllib, blocking)
                └─ candidates:51                              (two full CSV passes)
                └─ per-record urlopen + Image.verify:110-116

whose-mean-data visualize    data.main:223
  └─ prepare_interactive:158 ─ load_records:142
                             └─ square:149 × N                (all images in RAM)
                             └─ writes outputs/nga/works.json

whose-mean / OPEN_DESKTOP_STUDIO.cmd
  app.main:861 ─ tk.Tk() ─ Studio.__init__:246
      ├─ json.loads(works.json):253                 [UI thread, blocking]
      ├─ data.load_records():256                    [UI thread, blocking]
      ├─ _build_layout:306
      ├─ _load_atlas:428 ─ data.thumbnail_atlas:206 [UI thread, decodes N JPEGs]
      │     └─ 2N ImageTk.PhotoImage + 2N PIL tiles
      ├─ _bind_scene:445
      └─ root.after(100, poll_training):285

  poll_training:819  (every 1000 ms, UI thread)
      ├─ training.status:64          → reads current.json + globs frames/
      ├─ rebuild_epoch_view:711      → opens EVERY epoch PNG, builds board
      ├─ load_epoch:696              → opens + LANCZOS-resizes one PNG
      └─ rebuild_results_view:735    → decodes atlas JPEG, N crops/pastes

  GENERATE ─ prompt_and_train:658 ─ prompt_matches:631 ─ training.matching_ids:31
                                  ─ build_keyword_edges:639   (O(n²))
                                  ─ start_training:679 ─ training.start:80
                                        └─ threading.Thread(_run):102   [worker]

  training._run:124  (daemon worker thread)
      ├─ load_records / square × N        (mean accumulation, :139-145)
      ├─ _pair:112 × N × epochs           (JPEG decode + LANCZOS every sample)
      ├─ Generator512 / Discriminator512 forward+backward
      ├─ frame:169 per epoch, torch.save × 2 per epoch
      └─ generation pass :223-234 → generated/*.jpg + generated_atlas.jpg
```

### 1.3 Threading model

- **Exactly one background thread**, created at `training.py:102`, `daemon=True`, named `pix2pix-live-training`. There is no thread pool, no `DataLoader` worker, no async I/O.
- **Marshalling is entirely via the filesystem.** The worker never touches Tk objects and never calls `root.after` — it serialises a whole state dict to `outputs/nga/live/current.json` via `_write` (`training.py:50`), which writes to a PID+TID-suffixed temp file and `Path.replace()`s it (atomic on POSIX and on Windows since the `PermissionError` retry loop at `training.py:54-61`). The UI polls that file once a second (`app.py:819-848`). **This is the single best design decision in the codebase**: it means there is no shared mutable Python object between threads, so no data race on the status payload. **[R]**
- **Synchronisation primitives:** one `threading.Lock` (`_guard`, `training.py:22`) guarding only `start()`, and one `threading.Event` (`_stop`, `training.py:21`). `stop()` (`training.py:107`) reads the `_thread` global **without** the lock.
- **Everything else runs on the Tk main thread**, including all image decoding, all resizing, the export copy, and the whole scene render.

### 1.4 State ownership

| State | Owner | Shared? | Risk |
|---|---|---|---|
| `Studio.works`, `by_id`, `record_map` (`app.py:254-256`) | UI thread | No | Read-only after init. Fine. |
| `Studio.removed`, `keyword_matches`, `hovered`, `selected`, `rot_*`, `*_zoom` | UI thread | No | Fine, but scattered across one god object. |
| `training._thread`, `_stop`, `_guard` (`training.py:20-22`) | Module globals | Yes (UI ↔ worker) | Process-local only — see C-6. |
| `current.json` | Worker writes, UI reads | Yes (via disk) | Cross-*process* shared; stale files survive crashes — see C-6. |
| `_run`'s local `state` dict (`training.py:126`) | Worker only | No | Good. |
| `thumbnail_atlas` `lru_cache` (`data.py:206`) | Process-global | Potentially | Not thread-safe by contract, but only ever called from the UI thread. |

### 1.5 Where responsibilities are tangled

1. **`Studio` is a god object.** `app.py:245-848` mixes widget construction, a hand-rolled 3-D projector (`rotate`/`project`, `app.py:456-468`), hit testing, an MST builder, file-format knowledge of the run directory layout, PIL decoding, and `shutil` export. There is no view/model separation and no way to unit-test any of it without a display.
2. **Keyword search lives in `training.py`.** `matching_ids` (`training.py:31`) re-reads and re-parses `works.json` from disk on every call and belongs in `data.py` or a new `search.py`.
3. **The run-directory layout is duplicated in three places.** `app.py:693-698`, `app.py:736-743`, `app.py:786-800` and `training.py:125, 219-235` each independently encode `frames/epoch_%03d.png`, `generated/<id>.jpg`, `generated_atlas.jpg`, `records.json`. There is no single `RunLayout` abstraction.
4. **Atlas geometry is duplicated.** `columns = 20` at `data.py:211` versus the hard-coded `index % 20` / `index // 20` at `app.py:436`; `cols = 32` at `training.py:219` versus `source_columns = 32` at `app.py:744` and `index % 32` at `app.py:764`. Changing one silently renders the wrong thumbnails. **[R]**
5. **Two different luminance definitions coexist.** `data.py:163` uses a flat channel mean; `training.py:115` and `training.py:148` use Rec.601 (`.299/.587/.114`). The Y axis the user navigates is not the luminance the model is trained on.
6. **Coding style is bimodal.** `data.py`/`model.py` are PEP 8 with full type hints; `training.py` and roughly half of `app.py` are dense semicolon-chained one-liners with no hints (`training.py:183-197`, `app.py:52-74`, `app.py:639-653`). This looks like two different authors or two different iterations and materially hurts reviewability.

---

## 2. Correctness risks and concrete bugs

Ordered roughly by severity. Every item carries a `file:line`.

### C-1 — Any non-ASCII or non-alphanumeric keyword silently selects the *entire* collection **[V]**

`training.py:33` tokenises with `re.findall(r'[a-z0-9]+', ...)`. A query with no ASCII alphanumerics (Chinese, Cyrillic, `"!!!"`, `"—"`) yields zero tokens, and `training.py:34` then returns **every** object id.

Downstream, `prompt_and_train` (`app.py:658-677`) sees a non-empty `query`, gets `matches == all ids`, computes `run_removed = (all_ids - matches) | self.removed` → `self.removed`, and trains on the **whole collection** while the UI highlights everything as a "match". The user believes they are training a concept-filtered mean; they are not.

Reproduced:

```
matching_ids('中', works) -> {1, 2, 3, 4}   # all four fixtures
```

The `if not query_tokens: return all` branch is only correct for the *empty-prompt* case, which `app.py:635` and `app.py:659` already handle separately. It should return `set()`.

### C-2 — The pluralisation heuristic is one-directional and misses the obvious case **[V]**

`training.py:39`:

```python
aliases.update({token+'s', token[:-1] if token.endswith('s') else token})
```

For a plural query the stem is never added. Verified: `matching_ids('buses')` does **not** match the work titled *"A Bus in the City"*; `matching_ids('bus')` does. It also generates junk (`'glass'` → `{'glass', 'glasss', 'glas'}`) and mangles legitimate words ending in `s` (`'paris'` → also tries `'pari'`). For an artwork whose central interaction is keyword curation, this is a user-visible correctness defect.

### C-3 — `matching_ids` never searches the artwork descriptions **[V]**

`training.py:43` searches only `title`, `artist`, `classification`, `medium` from `works.json`. The NGA `assistivetext` field (confirmed present in `published_images.csv` — it is a rich, machine-generated visual description) *is* stored in `records.jsonl` (`data.py:90`) and *is* displayed in the artwork panel (`app.py:227`), but is never indexed. Worse, `data.py:174-194` does not copy `assistivetext` into `works.json` at all, so it is not even reachable from the search path. The result: searching `"tree"` finds works with "tree" in the *title*, not works that *depict* trees — directly undercutting the "CONCEPT KEYWORDS" framing at `app.py:340`.

### C-4 — `STOP` is ignored during the two longest phases **[R]**

`_stop` is only checked at `training.py:178`, `:182` and `:203`.

- The mean-accumulation loop (`training.py:140-144`) decodes and letterboxes **every** image at 512 px with no `_stop` check. For 1,000 works that is minutes of unstoppable work.
- The result-generation loop (`training.py:223-233`) runs a full 512 px forward pass, JPEG encode and atlas paste for **every** record with no `_stop` check.

Pressing STOP during either phase sets the event, `stop()` returns `True`, the UI reports success, and nothing happens. Additionally `_stop.is_set()` remains `True` when the phase ends, so the run then aborts at `training.py:203` after wasting the entire generation pass.

### C-5 — Unhandled exceptions escape the worker thread's `try` **[R]**

`training.py:125-126` run **before** the `try:` at `:127`:

```python
run=LIVE/run_id;frames=run/'frames';frames.mkdir(parents=True,exist_ok=True)
state=json.loads((LIVE/'current.json').read_text(encoding='utf-8'))
```

If `mkdir` fails (permissions, full disk, path length) or `current.json` is missing/corrupt/mid-replace, the exception propagates out of `_run`, the thread dies, and **no error state is ever written**. `current.json` keeps saying `"queued"` forever and the UI's GENERATE button stays disabled (`app.py:822-824`). Because `OPEN_DESKTOP_STUDIO.cmd:9` launches via `pythonw.exe` there is no console, so `threading.excepthook`'s traceback goes to a non-existent stderr and is lost entirely. **The failure is completely invisible to the user.**

Note also that the `except Exception` handler at `training.py:240-242` writes `state`, which at that point may still be the *initial* queued dict — so an error occurring after several epochs discards the accumulated `history`.

### C-6 — Stale on-disk state permanently bricks the GENERATE button **[R]**

The thread is `daemon=True` (`training.py:102`). Closing the window during training kills the process with `current.json` still reading `"state": "training"`. On the next launch:

- `training.status()` (`training.py:64-77`) reads that file and returns `state == "training"`.
- `poll_training` (`app.py:822-824`) computes `active = True` and disables GENERATE.
- `training.stop()` (`training.py:107-109`) returns `False` because the in-process `_thread` global is `None`, so the user gets "No training run is active in this desktop process" (`app.py:691`) and **has no way to recover from the UI**. They must manually delete `outputs/nga/live/current.json`.

The same happens after any hard crash, OOM kill, or the C-5 silent thread death. There is no PID/heartbeat in the status payload (`training.py:96-100`) to let a fresh process detect that the writer is gone.

### C-7 — PIL file handles are leaked on every image load in the UI **[R]**

`data.py:149-155` correctly uses `with Image.open(...)`. `app.py` never does:

| Location | Code |
|---|---|
| `app.py:702` | `Image.open(path).convert("RGB").resize(...)` |
| `app.py:722` | `Image.open(path).convert("RGB").resize(...)` |
| `app.py:740` | `Image.open(path).convert("RGB")` |
| `app.py:762` | `Image.open(path).convert('RGB')` |
| `app.py:817` | `Image.open(path).convert('RGB')` |

`Image.open` is lazy and keeps the underlying file object until the `Image` is closed or garbage-collected. `.convert()` returns a *new* image and drops the last strong reference to the original, so CPython refcounting usually closes it promptly — but this is not guaranteed, and under `rebuild_epoch_view` (`app.py:722`) the pattern is executed once per epoch tile per poll. On Windows an un-closed handle blocks deletion of the run directory. Combined with `app.py:801` (`glob` over the export destination) and the export copy, this is a latent `ResourceWarning`/`PermissionError` source.

### C-8 — `export_results` can raise an unhandled exception mid-copy **[R]**

`app.py:787`:

```python
shutil.copy2(run/'frames'/f'epoch_{latest_epoch:03d}.png', destination/'pix2pix_mean.png')
```

There is **no existence check** on this specific file (unlike every other file in the loop at `app.py:788-792`). `frames` comes from `training.status()`, which derived it from a directory glob (`training.py:74`) that may be stale or may have observed a partially written PNG. If the file is absent the `FileNotFoundError` propagates out of a Tk command callback: Tk prints a traceback to stderr (invisible under `pythonw`), the just-created destination directory (`app.py:785`) is left as an empty orphan, and the user sees nothing at all.

The entire export — `shutil.copytree` of `frames/` **and** `generated/` (up to 1,000 JPEGs, potentially >1 GB) at `app.py:793-796` — also runs synchronously on the UI thread with the window frozen and no progress indication.

### C-9 — `set_output_zoom` silently disables epoch auto-follow **[R]**

`app.py:614-618` calls `load_epoch(self.current_epoch)`, and `load_epoch` unconditionally sets `self.follow_latest = False` (`app.py:704`). So merely zooming the mean preview with the mouse wheel or the `+`/`−` buttons stops the panel from advancing to new epochs — a silent, undiscoverable mode change. `load_epoch` conflates "render this epoch" with "the user chose this epoch".

### C-10 — `rebuild_results_view` can render a truncated result grid **[R]**

`app.py:744`:

```python
count = int(state.get("generated", state.get("artworks", 0)))
```

`has_generated_atlas` becomes true the instant `generated_atlas.jpg` is written (`training.py:234`), but `state['generated']` is only refreshed every 25 items (`training.py:231-233`) and is not set to `len(records)` until `training.py:236`. There is a real window in which `poll_training` (`app.py:843`) sees the atlas but a stale count, renders fewer than all tiles, and — because of the `last_result_run` guard at `app.py:843-845` — **never rebuilds the view again for that run**. Tiles are permanently missing. Also `self.result_ids` (`app.py:743`) is read from `records.json`, which is written *after* the atlas (`training.py:235`), so it can be empty, disabling all result clicks for the run.

### C-11 — `AttachedToplevel` may destroy all `<Configure>` bindings on the root window (Python ≤ 3.11) **[V for 3.12 / R for 3.11]**

`app.py:129` binds with `add='+'`; `app.py:150` unbinds with a funcid:

```python
self.owner.unbind('<Configure>', self._owner_binding)
```

I inspected CPython **3.12**'s `tkinter.Misc._unbind` and confirmed it is funcid-selective (it filters the binding script and only removes the matching handler) — so on 3.12+ this is correct. **[V]**

On CPython **3.11** — which `pyproject.toml:10` declares as the minimum and `README.md:39` recommends — `Misc.unbind(sequence, funcid)` historically called `bind(sequence, '')`, wiping *every* handler for that sequence. If that is the case on the user's interpreter, closing one `ArtworkViewer`/`ResultViewer`/`ComparisonViewer` breaks window-following for all the others. I could not execute this check (no 3.11 tkinter installed), so treat it as a version-dependent risk to confirm, not a certainty. **[R]**

Independently: `ComparisonViewer` is constructed at `app.py:767` with **no reference retained and no destroy-on-reopen logic**, unlike `detail_window` (`app.py:585-588`). Every result click spawns another window, each holding two 512 px PIL images plus up to two 2048×2048 `PhotoImage`s (`app.py:108-113`). They accumulate until manually closed.

### C-12 — Startup has no error handling for missing or corrupt data **[R]**

`app.py:253` (`json.loads(WORKS_PATH.read_text(...))`) and `app.py:256` (`load_records()`) run in `Studio.__init__` with no `try`. If `works.json` is missing, `FileNotFoundError` escapes `main()`; if `records.jsonl` is missing, `data.py:145` raises `SystemExit` — from inside a GUI, after `tk.Tk()` has already been created. Under `pythonw.exe` the user sees a window flash and vanish with no message. The obvious failure mode ("I ran the app before running `collect`") produces the worst possible diagnostics.

### C-13 — `nearest()` ignores depth **[R]**

`app.py:536-542` picks the 2-D-nearest projected point within 14 px, disregarding `z`. Because `draw_scene` painter-sorts by `z` (`app.py:491`), a work that is visually *behind* and occluded can be selected in preference to the one the user can actually see, if it is a pixel or two closer in screen space. In REMOVE mode this means silently excluding the wrong artwork.

### C-14 — `_stop` / `_thread` read without the lock **[R]**

`training.py:108` reads `_thread` outside `_guard`, while `training.py:82-103` mutates it under the lock. In CPython this is benign (attribute reads are atomic), but it is an unsynchronised access that a future refactor will make real. Similarly `stop()` can set `_stop` on a thread that is about to exit, leaving the event set; the next `start()` clears it at `training.py:95`, so no bug today.

### C-15 — Non-atomic artwork downloads survive `Ctrl-C` **[R]**

`data.py:112-113` streams straight into the final `target` path — unlike `download()` (`data.py:38-42`), which correctly uses a `.part` file plus `replace()`. The `except Exception` at `data.py:123` unlinks the partial file, but **`KeyboardInterrupt` is a `BaseException`** and is not caught. A `Ctrl-C` during a download leaves a truncated `NNNN.jpg`, and the next run's `if not target.exists()` guard (`data.py:109`) accepts it without re-downloading and, critically, **without running `verify()`** — which is only executed on the freshly-downloaded path (`data.py:114-116`). The corrupt file then poisons `prepare_interactive`, the atlas, and training.

### C-16 — `prepare_interactive` has no per-image error handling **[R]**

`data.py:160` builds the whole stack in one comprehension. A single missing or corrupt file raises and kills the entire `visualize` command with no indication of which record failed and no partial progress. The same is true of `thumbnail_atlas` (`data.py:214-217`), except there a missing *record* is skipped but a corrupt *file* still raises — during UI startup (`app.py:429`), with no handler.

### C-17 — CUDA / GPU memory **[R]**

- Hard requirement at `training.py:131`; no CPU fallback, not even a slow one. See H-4 for why this collides with the documented Windows install.
- No `torch.cuda.empty_cache()`, no explicit `del` of `g`/`d`/optimizers. In practice the worker thread frame is released when `_run` returns, so the allocator caches are reclaimable — I judge this **not** to be a leak. **[R]**
- `lg.item()` / `ld.item()` at `training.py:197` force a device synchronisation on **every** sample, serialising the pipeline. Minor but measurable.
- No OOM handling: a CUDA OOM lands in the generic `except Exception` (`training.py:240`) and surfaces as a raw exception string in a Tk label.

### C-18 — `smoke_test` hard-codes the sample size **[C]**

`app.py:856` asserts `works == 1000` and `atlas == [1280, 3200]`. `collect --limit` is user-configurable (`data.py:227`) and network failures reduce the count (`data.py:104-127`), so the only "test" in the repo fails for any run that is not exactly 1,000 successful downloads. The geometry itself is correct for N=1000 (verified by calculation), but the assertion is a constant where it should be derived.

### C-19 — Blocking calls on the UI thread (summary) **[R]**

| Location | Blocking work |
|---|---|
| `app.py:253-256` | JSON parse of `works.json` + 1,000-line JSONL parse |
| `app.py:429` → `data.py:206-220` | Decode + LANCZOS-resize **1,000 JPEGs**, JPEG-encode a 1280×3200 atlas |
| `app.py:435-442` | 1,000 crops, 1,000 `Image.blend`, **2,000 `ImageTk.PhotoImage`** allocations |
| `app.py:634-637` | `works.json` re-read + re-parse + O(n²) MST **on every key release** |
| `app.py:696-709` | PNG decode + LANCZOS resize per epoch change and per zoom tick |
| `app.py:711-724` | Decode of **every** epoch PNG, every time the frame set changes |
| `app.py:735-749` | Atlas JPEG decode + up to 1,000 crop/paste ops |
| `app.py:769-804` | `shutil.copytree` of the entire run directory |
| `app.py:108-113` | LANCZOS resize up to 3072×3072 on every mouse-wheel tick |
| `training.py:74` | Directory glob, once per second, from `poll_training` |

### C-20 — Minor / dead code **[R]**

- `self.epoch_refs` (`app.py:275`) and `self.detail_photo` (`app.py:278`) are assigned and never used.
- `connected` in `build_keyword_edges` (`app.py:643`, `:649`) is built and never read.
- `WhiteSlider.set_from_pointer` (`app.py:65-67`) stores `round(value, 2)` but passes the **unrounded** `value` to the callback, so the label (`app.py:656`) can disagree with the value actually sent to `training.start`.
- `draw_work` computes `source_tile` (`app.py:517`) but the `is_match` branch at `app.py:526` uses `self.source_tiles[index]` directly — harmless today because `is_match` implies `not is_dim`, but fragile.
- `saturation` is computed (`data.py:169`) and stored (`data.py:191`) but never used anywhere.
- `urlopen` is called with a URL taken from a downloaded CSV (`data.py:110`). `urllib` honours `file://` and `ftp://`. Low risk given the source is a trusted GitHub repo, but the scheme should be allow-listed.
- The poll loop (`app.py:848`) is never cancelled; `--ui-smoke-test` calls `root.destroy()` (`app.py:880`) with an `after` pending.

---

## 3. Performance

### 3.1 Per-frame cost of the XYZ canvas

`draw_scene` (`app.py:478-511`) does a **full teardown and rebuild** of the canvas on every hover change, every drag delta, every zoom tick, and every `<Configure>`:

1. `self.scene.delete("all")` — destroys ~1,000–2,000 canvas items.
2. A Python loop over all 1,000 works computing `rotate` + `project` (`app.py:456-468`) — ~8 trig calls and ~20 float ops each, all in interpreted Python.
3. A dict comprehension for `positions` (`app.py:490`) and a `sorted()` over 1,000 tuples (`app.py:491`).
4. **Two full passes** over the sorted list (`app.py:492-505`) plus a linear `next(...)` scan for the hovered row (`app.py:509`).
5. `create_image` × 1,000 — each allocates a new Tk canvas item.
6. When a keyword filter is active, `create_rectangle` × |matches| **and a freshly allocated `ImageTk.PhotoImage` per match** (`app.py:526-527`) — a 64→28 px LANCZOS resize per matching work, **per frame**. A query like `"painting"` matches by classification and can select hundreds of works, so this is hundreds of LANCZOS resizes and PhotoImage allocations at mouse-move rate.

The hovered thumbnail is also re-rendered from scratch every frame (64→88 LANCZOS, `app.py:520`) even when only the rotation changed.

Tk's canvas is a retained-mode scene graph; `delete("all")` + 1,000 `create_image` is the most expensive possible way to use it. I could not measure this (no display), but from the item counts and the interpreted per-item work, a redraw in the **50–250 ms** range is the realistic expectation — i.e. 4–20 fps while orbiting, with visible hover latency. **[R]**

The fix is straightforward and well-supported: create the 1,000 canvas items **once** and use `Canvas.coords()` / `itemconfigure()` / `tag_raise()` to move them, plus `after_idle` coalescing for drag events.

### 3.2 Keyword interaction cost **[V]**

`update_keyword_highlights` is bound to `<KeyRelease>` (`app.py:347`) with **no debounce**. Each keystroke triggers:

- `training.matching_ids` (`training.py:31-47`) — full re-read and `json.loads` of `works.json` (~0.4–1 MB for 1,000 works) plus a regex tokenisation of every work's metadata. There is no index and no caching.
- `build_keyword_edges` (`app.py:639-653`) — a Prim's MST in pure Python. **Measured: 445 ms for 1,000 points**, scaling quadratically (100 → 4.4 ms, 250 → 27.9 ms, 500 → 111 ms).
- A full `draw_scene`.

So a broad query on a fast typist's keyboard queues ~0.5 s of work *per character*. This is the worst interactive latency in the application. Mitigations: debounce ~200 ms, cache the parsed `works.json` and an inverted token index, vectorise the MST with NumPy (or cap the edge count), and skip the MST above a match threshold.

### 3.3 Startup **[R/C]**

`thumbnail_atlas` (`data.py:206-220`) is memoised with `functools.lru_cache` **in memory only** — the atlas is **never persisted to disk**. Every application launch therefore:

1. Parses `works.json` and `records.jsonl`.
2. Opens **1,000 JPEGs** (each up to 768×768 as fetched by `data.py:110`), EXIF-transposes, `ImageOps.contain` with LANCZOS down to 64 px, and pastes them (`data.py:214-217`).
3. JPEG-encodes a 1280×3200 atlas with `optimize=True`.
4. Then `_load_atlas` (`app.py:428-443`) re-decodes that atlas, crops 1,000 tiles, runs 1,000 `Image.blend` calls, and allocates **2,000 `ImageTk.PhotoImage`** objects.

All of this is on the Tk main thread before the first frame is drawn, with no splash screen or progress. Expect **tens of seconds** of an unresponsive or unpainted window on a cold cache. Persisting `atlas.jpg` next to `works.json` during `visualize` eliminates step 2 entirely and is a ~10-line change.

Memory: 1,000 × 64×64 RGB tiles × 2 (normal + dimmed) ≈ 24 MB of PIL data, plus 2,000 16×16 `PhotoImage`s. Tolerable, but the dim tiles are also stored at full 64 px (`app.py:440`) when only the 16 px version is ever shown for dim works.

### 3.4 Training loop **[R/C]**

| Aspect | Current | Assessment |
|---|---|---|
| Batch size | **1** (`training.py:183`) | Catastrophic GPU utilisation at 512 px. The 16.7 M-param generator will be launch-bound, not compute-bound. Batch 4–8 is feasible in fp16 on 8–12 GB. |
| Data loading | `_pair` (`training.py:112-117`) calls `data.square` → **opens, EXIF-transposes, LANCZOS-resizes the 768 px source JPEG from disk for every sample of every epoch** | This is almost certainly the bottleneck. 1,000 decodes/epoch × 20 epochs = 20,000 redundant decodes. No `Dataset`, no `DataLoader`, no `num_workers`, no prefetch, no pinned memory, no caching. Pre-resizing once to a 512 px `.npy` memmap (~786 MB) or even a cached 512 px JPEG directory would give a large multiple. |
| Mean pass | `training.py:139-144` | Another **full** 1,000-image decode+resize before training even starts, single-threaded, not interruptible (C-4), and not cached between runs even when the exclusion set is unchanged. |
| Mixed precision | `torch.autocast('cuda', float16)` + two `GradScaler`s (`training.py:136`) | Correctly applied; `fake` is computed once (`:184`) and reused for both the D and G passes, which is a genuine saving. `bfloat16` would remove the need for scalers on Ampere+. |
| Sync points | `lg.item()`, `ld.item()`, `reconstruction.item()`, `edges.item()` every sample (`training.py:197`) | Four device syncs per sample. Accumulate on-device and `.item()` once per epoch. |
| Status writes | Every 10 samples (`training.py:198`) | 100 JSON serialisations of the **entire** state dict (including the growing `history`) per epoch, each a write + atomic replace. Should be time-based (e.g. ≥250 ms) and should not re-serialise history. |
| Checkpointing | **Two** `torch.save` of the full generator **per epoch** (`training.py:210-211`) | 67.0 MB each → **134 MB per epoch**, **2.68 GB for a 20-epoch run** (calculated). `model_best.pt` is also selected on *training* L1 (see M-3), so half of that write volume buys nothing. Save `latest` every N epochs and `best` only on improvement. |
| Retention | None | Every run writes a new `outputs/nga/live/run_*` directory containing ~1,000 JPEGs, up to 200 PNGs and 2.7 GB of checkpoints. Nothing is ever pruned. |
| Generation pass | `training.py:223-234` | 1,000 sequential batch-1 forward passes plus 1,000 JPEG encodes with `optimize=True`, all uninterruptible (C-4). Batch this. |
| Determinism | `torch.manual_seed(42)`, `cuda.manual_seed_all(42)` (`training.py:132`), per-epoch seeded permutation (`training.py:180`) | Good intent, but no `cudnn.deterministic`, no `use_deterministic_algorithms`, and `benchmark` is left at its default. Runs are not bit-reproducible. |

### 3.5 Epoch / result browser **[R]**

`rebuild_epoch_view` (`app.py:711-724`) is called from `poll_training` whenever the frame set changes — i.e. **once per epoch** — and each time re-opens and re-resizes **every** epoch PNG from the start of the run. Over an E-epoch run this is O(E²) image decodes on the UI thread: 20 epochs → 210 decodes; 200 epochs → 20,100 decodes. It should append one tile.

The result board (`app.py:745`) for 1,000 works is 320 × 12,800 px — a single ~12 MB `PhotoImage` (`app.py:47`). Workable but wasteful; a virtualised/windowed view would be better.

`ZoomPane.show` (`app.py:108-113`) LANCZOS-resizes the full source on **every** wheel tick, up to 512→3072 (≈ 38 MB `PhotoImage` at 6× in `ResultViewer`). Wheel events arrive in bursts, so zoom will feel unresponsive. Coalesce with `after_idle` and downsample before upscaling.

---

## 4. Data pipeline

### 4.1 Collection and reproducibility

`README.md:53` claims: *"The sample uses seed 42 and selects accessioned, open-access primary images... It is a reproducible subset."* **This claim does not hold.** **[V]**

`data.py:28-29` pins the CSVs to the **`main` branch HEAD** of `NationalGalleryOfArt/opendata`:

```
.../opendata/main/data/objects.csv
.../opendata/main/data/published_images.csv
```

I fetched both headers live and confirmed the data is actively maintained (records carry `modified` timestamps in **2026-04**). `random.Random(42).shuffle(pool)` (`data.py:101`) is deterministic *given a fixed pool*, but the pool is rebuilt from a moving upstream: any object added, deaccessioned, reclassified, or newly opened changes `len(pool)` and therefore permutes **the entire selection**, not just the tail. Two collectors running the same command a month apart get materially different 1,000-work samples.

Compounding factors:

- `download()` skips re-download if the local file exists and is `> 1_000_000` bytes (`data.py:35`). So the sample also depends on *when you first ran it*, and a truncated-but-large CSV is silently accepted forever.
- Failures are not backfilled: `chosen = pool[:limit]` (`data.py:102`) and any failed download simply reduces the sample (`data.py:123-127`). The final count is network-dependent — which is exactly what `app.py:856` then asserts to be 1,000.
- Nothing records the upstream commit SHA, the CSV checksums, or the resolved object-id list in a form that could be replayed.

**Minimum fix:** pin a specific upstream commit SHA in the URLs, record the SHA + CSV SHA-256 + the ordered selected id list in a `manifest.json`, and add a `--manifest` replay path. Backfill from the pool on failure so the count is exact.

### 4.2 Network error handling

`data.py:110-116` — single `urlopen` attempt, 90 s timeout, no retry, no exponential backoff, no handling of HTTP 429/5xx, no `Retry-After`, no connection pooling. The default `--delay 0.02` (`data.py:228`) means up to ~50 requests/second against `api.nga.gov`, which is aggressive for a public museum IIIF endpoint. `download()` (`data.py:33-43`) has the same single-shot behaviour for the two large CSVs — a dropped connection 90 MB into `objects.csv` loses the whole transfer.

Failures are recorded well (`failures.jsonl`, `data.py:133-136`) with exception type and truncated message, and `collect` aborts only below 8 usable images (`data.py:138`). But there is no `--retry-failures` mode, so recovering from a transient outage means re-running the whole command.

### 4.3 Feature computation — soundness

Features are computed at `data.py:158-173` on the output of `square()` (`data.py:149-155`), which **letterboxes onto a `(238, 236, 230)` off-white canvas**. This is the central methodological problem.

**[C]** Calculated pad contribution for realistic NGA aspect ratios (source images are fetched as `!768,768`, so they preserve the original aspect):

| Source | Contained size at 96 px | Pad fraction |
|---|---|---|
| 768 × 512 (3:2) | 96 × 64 | **33.3 %** |
| 512 × 768 (2:3) | 64 × 96 | **33.3 %** |
| 768 × 768 | 96 × 96 | 0 % |
| 1000 × 300 (panorama) | 96 × 29 | **69.8 %** |

The pad colour has luminance **0.920** and warmth (R−B) **+0.0314**. Consequences:

- **Y (mean luminance, `data.py:163`)** for a 3:2 work is `0.333 × 0.920 + 0.667 × actual`. A dark 3:2 painting and a mid-tone square painting can land at the same Y. **The Y axis substantially encodes aspect ratio.**
- **X (warmth, `data.py:166`)** is dragged toward `+0.0314` in proportion to the pad fraction.
- **Z (edge density, `data.py:168`)** is doubly corrupted: the pad introduces two hard artificial edges at the image/pad boundary (a large gradient spike), while simultaneously diluting the mean over a large flat region. The net direction depends on the work's own contrast, so the bias is not even monotonic.

Other issues:

- **Luminance definition is inconsistent with the rest of the system.** `data.py:163` uses a flat channel mean; `training.py:115` and `:148` use Rec.601 weights. The axis the user navigates is not the luminance the model sees.
- **No gamma linearisation.** Warmth and luminance are computed on gamma-encoded sRGB, so "mean luminance" is perceptual-ish rather than photometric. Defensible for an artwork, but undocumented.
- **Edge density is resolution-coupled.** Forward differences at a fixed 96 px (`data.py:230`) measure edge density *after* an arbitrary downscale; changing `--size` changes the Z axis for everyone. LANCZOS ringing also injects edges.
- **Normalisation is dataset-relative.** `data.py:173` z-scores over the sample, clips to ±2.5σ and divides by 2.5. Coordinates are therefore not comparable across collections, and adding one work moves every other work. Clipping also silently collapses outliers onto the cube faces.
- **`saturation` is computed and stored but unused** (`data.py:169`, `:191`) — a fourth axis that never made it into the UI.

**Recommended fix:** compute features on the *contained* region only (mask the pad), switch to Rec.709/601 luminance consistently, and record the feature-extraction parameters in `works.json` alongside the coordinates.

### 4.4 Corrupt / missing images

See C-15 and C-16. Summary: verification only runs on freshly downloaded files (`data.py:114-116`); pre-existing files are trusted unconditionally; `Image.verify()` is a weak check that does not decode pixel data; and both `prepare_interactive` (`data.py:160`) and `thumbnail_atlas` (`data.py:217`) will abort the whole operation on a single bad file, the latter during UI startup with no handler.

### 4.5 Licensing and provenance

**Captured well:** `objectid`, `title`, `attribution`, `displaydate`, `classification`, `medium`, `creditline`, `image_uuid`, `iiifurl`, `source` URL, `collected_at` timestamp, and `openaccess: True` (`data.py:76-94`). The UI surfaces title, artist, date, classification, medium, object id, credit line and description (`app.py:219-228`). That is a genuinely respectable provenance record for a student project.

**Gaps:**

1. **No licence identifier is recorded.** `openaccess: True` is a boolean flag, not a licence. NGA open-access images are CC0 / public domain, but nothing in the repo states this, and I confirmed neither CSV carries a licence column — so the licence must be asserted from the programme's terms, and it isn't. `data.py:92`.
2. **`works.json` drops `creditline`, `assistivetext` and `image_uuid`** (`data.py:174-194`). `works.json` is the only artefact the UI's search path reads, so provenance and search are split across two files that can drift.
3. **Exported derivatives carry no attribution.** `export_results` (`app.py:769-804`) copies `records.json`, which `training.py:235` populates with **only `id` and `title`** — no artist, no credit line, no source URL, no licence. So the exported `generated/*.jpg` (derivative works of specific artworks) and `pix2pix_mean.png` ship with essentially no provenance. For an artwork about *whose* mean this is a conceptual as well as a legal weakness.
4. **`assistivetext` is presented as "DESCRIPTION"** (`app.py:227`). I confirmed from the live CSV that this field is machine-generated alt text, not curatorial description. Labelling it as a description misrepresents its epistemic status — ironic given the project's Steyerl/Turkle framing.
5. **No `User-Agent` contact information.** `WhoseMean/1.0` (`data.py:39`, `:111`) carries no URL or email, contrary to normal practice for bulk museum-API access.

---

## 5. ML soundness

### 5.1 What the model actually learns

`_pair` (`training.py:112-117`) constructs each training pair as:

- **Target**: the 512 px letterboxed RGB artwork, in `[-1, 1]`.
- **Input**: the **Rec.601 grayscale of that same target**, replicated across three channels.

So the task is **grayscale → RGB colourisation of a single artwork**, conditioned on nothing else. The loss (`training.py:194`) is:

```
L_G = BCE(D(a, fake), 1) + 30·L1(fake, b) + 10·edge_loss(fake, b)
```

With the L1 term weighted 30× and the gradient term 10× against a GAN term of weight 1, this is **an L1+gradient regression with a decorative adversarial garnish**. The discriminator (`model.py:56-75`, ~694 k params, 4 downsamples → 32×32 patch map) contributes roughly 2–3 % of the generator's gradient magnitude at typical loss scales. Calling the result "pix2pix" is accurate as to architecture and inaccurate as to dynamics.

There is also **no dropout** in `Generator512` (`model.py:22-26`). In the original pix2pix, dropout at train *and* test time is the generator's only source of stochasticity; without it the generator is a **deterministic function**. It is not a generative model in any meaningful sense — it is a learned image-to-image filter.

### 5.2 Does the objective match the "mean image" claim?

**No.** This is the most important finding in the audit.

The "mean image" is produced at `training.py:169-175`: take `structured_source`, push it through the generator, save the result. But `structured_source` (`training.py:160`) is a **blend of two grayscale inputs**, and the generator was never trained on blended inputs — it was trained on the grayscale of *individual* artworks. Feeding it a blend is an out-of-distribution query. The output is "what this colouriser does to a blurry grey image", which is a plausible artistic gesture but is not a mean of anything the training optimised for.

Meanwhile, **the true arithmetic mean is already computed exactly, without any learning**, at `training.py:139-145`, and saved as `mean_target.png` (`training.py:150`). The pixel mean of the curated subset requires zero epochs of GAN training. Every epoch of training changes the *colourisation style* applied to that mean, not the mean itself.

Concretely, what the epoch sequence in `frames/` shows is **the convergence of a colourisation network**, not the convergence of a mean. The loss that actually decreases (`train_l1`, `training.py:206`) measures reconstruction of individual artworks and has no relationship to the mean image being displayed. The UI reinforces the confusion by showing `TRAIN L1` next to the mean preview (`app.py:833`).

This is fixable *conceptually* without abandoning the piece — e.g. state honestly that the piece trains a colouriser on the curated subset and then asks it to hallucinate colour for the subset's average luminance; or add a term that explicitly optimises `G(mean_gray) ≈ mean_rgb`; or show `mean_target.png` alongside the generated frame so the viewer can see the difference between the statistical mean and the model's mean. But the current README/UI framing overclaims.

### 5.3 What the structure anchor really does

`training.py:151-162`:

1. Take the XYZ coordinates of the *included* records from `works.json` (`:152-155`). Note these are the **globally** z-scored coordinates from `data.py:173`, so the "centroid" is computed in a normalisation frame defined by the *whole* collection, not the subset. The centroid itself is correctly computed over the subset (`:156`).
2. Pick the single record nearest that centroid (`:157`) — the "representative".
3. Blend: `structured_source = (1 - s)·mean_source + s·anchor_source` (`:160`), with `s = 0.65` by default (`app.py:349`).

Three observations:

- **Both operands are grayscale.** `mean_source` (`training.py:148-149`) and `anchor_source` (`training.py:159`, from `_pair`) are luminance replicated across RGB. The blend contains **no colour information at all**; all colour in the "mean image" is invented by the generator.
- **The default is not a mean.** At 65 % the input is dominated by **one single artwork's luminance**. The UI is honest about this (`app.py:406-408`: *"35% arithmetic mean + 65% feature-centroid representative"*), and `README.md:80` describes it correctly — but the panel header still reads `PIX2PIX MEAN` (`app.py:370`) and the stats block still says `INPUT 512×512×3 GRAY` (`app.py:337`) without mentioning the blend. A user who never opens the MODEL I/O tab will reasonably believe they are looking at a mean.
- **Blending in tanh space.** The blend happens on `[-1, 1]` tensors derived from gamma-encoded sRGB, so it is neither a photometric average nor a perceptual one. Functionally it is a cross-dissolve, which is fine artistically but should not be described as arithmetic.

The anchor selection is also brittle: `np.square(coordinates - center).sum(1).argmin()` (`:157`) has no tie-breaking and no robustness to the clipped/normalised coordinate space (C-4.3), so a single outlier-heavy subset can produce a counter-intuitive representative.

### 5.4 Overfitting and mode collapse at 1,000 samples

| Risk | Assessment |
|---|---|
| **Overfitting** | **High and unmeasured.** There is no train/validation/test split anywhere. `train_l1` (`training.py:206`) is a pure training-set metric, and `model_best.pt` is selected on it (`training.py:209-210`) — so "best" means "most overfit". With 16.7 M parameters, 1,000 samples, no augmentation (no flips, no crops, no colour jitter), and up to 200 epochs, the generator can memorise the palette of individual works. |
| **Mode collapse** | **Low.** This is a *conditional* deterministic regressor with a 30× L1 term and skip connections; there is no latent to collapse. The real failure mode is the opposite — **regression to the mean**, i.e. desaturated, brownish, over-smoothed output. Given the sepia-heavy NGA prints/drawings subset and the warm `(238,236,230)` pad, expect the colouriser to converge on "everything is beige", which will look like a profound artistic statement and is actually an L1 artefact. |
| **Architecture** | The U-Net bottleneck is **8×8×512** (`model.py:33`), not 1×1 as in the original pix2pix. With six skip connections (`model.py:48-53`), the network has a strong identity path and limited global colour reasoning. It will produce locally plausible, globally incoherent colour. `ConvTranspose2d(4, 2, 1)` throughout (`model.py:24`) is a known checkerboard-artefact generator; `Upsample + Conv` is the standard remedy. |
| **Discriminator** | PatchGAN with `InstanceNorm2d` and no affine/spectral norm; D lr 1e-4 vs G 2e-4 (`training.py:134-135`), one-sided label smoothing at 0.9 (`training.py:188`). Reasonable choices, but with the GAN term at weight 1 against 30+10 it barely matters. |
| **`edge_loss`** | `model.py:78-87` is a sound first-difference gradient loss. Note it is computed on the **padded** image, so 10× weight is partly spent teaching the model to reproduce the letterbox border sharply. |
| **Seeding** | `torch.manual_seed(42)` + per-epoch permutation seed (`training.py:132`, `:180`) is good, but no `cudnn.deterministic`, so results are not reproducible run-to-run. |

### 5.5 Recommended ML changes (conceptually minimal)

1. Hold out 10 % of the subset; report validation L1; select checkpoints on it.
2. Add horizontal-flip augmentation at minimum.
3. Mask the letterbox pad out of `L1` and `edge_loss` so the model is not rewarded for painting the background.
4. Either add an explicit mean-reconstruction term, or reframe the claim (see 5.2).
5. Display `mean_target.png` next to the generated frame in the UI — it costs nothing and makes the conceptual point of the piece *legible* rather than obscured.

---

## 6. Engineering hygiene

### 6.1 Tests, typing, tooling **[V]**

| Item | Status |
|---|---|
| Test suite | **None.** No `tests/`, no `test_*.py`, no `pytest` dependency. |
| "Tests" that exist | `app.smoke_test` (`app.py:851-858`) with a hard-coded `1000`/`[1280,3200]` assertion, and `--ui-smoke-test` (`app.py:871-881`) which needs a display, a full dataset, and 1,000 downloaded images. Neither is runnable in CI. |
| Type annotations | `data.py` 9/9 defs, `model.py` 7/7 defs, **`app.py` 0/69**, **`training.py` 0/9**. |
| Type checker config | None (no mypy/pyright section in `pyproject.toml`). |
| Linter / formatter | None (no ruff/flake8/black config; `.gitignore` mentions `.ruff_cache/` and `.pytest_cache/`, so the intent existed). |
| CI | None (no `.github/`). |
| Logging | **No `logging` import anywhere.** Diagnostics are `print()` in `data.py`, a Tk label in `app.py:826`, a JSON field in `training.py:241`, and uncaught tracebacks to a stderr that `pythonw.exe` discards. |
| Docstrings | Module-level only; almost no function docstrings outside `model.py`. |

### 6.2 Packaging

- `pyproject.toml` is minimal but correct: setuptools backend, `src` layout, two console scripts (`pyproject.toml:18-20`).
- **`requirements.txt` duplicates `[project.dependencies]`** with the same three unpinned constraints — two sources of truth, no lockfile, no hashes.
- **No optional/dev extras** (`[project.optional-dependencies]`) for test/lint tooling.
- **`whose-mean` is registered as a `console_script`** (`pyproject.toml:19`), so on Windows it creates `whose-mean.exe`, which opens a console window — yet `README.md:66` tells the user to run it and `OPEN_DESKTOP_STUDIO.cmd:9` deliberately uses `pythonw.exe` to avoid exactly that. For a GUI app the entry point should be a `gui_scripts` entry.
- No `LICENSE` file in the repository, despite the project redistributing derivatives of museum images and documenting an open-data dependency.
- No package data / no `py.typed`.

### 6.3 Cross-platform assumptions (project is Windows-centric)

| Issue | Location | Effect on macOS / Linux |
|---|---|---|
| **Mouse wheel bindings** | `app.py:34`, `:105`, `:375`, `:454` | `<MouseWheel>` **does not fire on X11/Linux**, which delivers `<Button-4>`/`<Button-5>`. **All scroll interaction — orbit zoom, output zoom, epoch list scroll, viewer zoom — is dead on Linux.** |
| **`event.delta` magnitude** | `app.py:38`, `:171`, `:194`, `:238`, `:603`, `:607` | On Windows `delta` is ±120; **on macOS it is ±1**. `-int(event.delta/120)` → `0` (no scroll), and `math.exp(±1/120 × 0.12)` ≈ 1.001 (imperceptible zoom). **Scrolling is effectively broken on macOS too.** |
| **Fonts** | `Consolas` at `app.py:96`, `163`, `186`, `210`, `217`, `230`, and ~20 more | Windows-only font. Tk silently substitutes, breaking the deliberately monospaced layout (the info panel at `app.py:229-232` and the MODEL I/O block at `app.py:395-411` assume fixed width). Should be a font-family fallback list. |
| **Launcher** | `OPEN_DESKTOP_STUDIO.cmd` | Windows batch only. No `.sh`/`.command` equivalent. |
| **Documentation** | `README.md:39-69` | PowerShell only; `.\.venv\Scripts\...` paths do not exist on POSIX (`.venv/bin/`). No POSIX instructions at all. |
| **`PermissionError` retry loop** | `training.py:54-61` | A 100×20 ms retry around `Path.replace()` — a Windows-only workaround (POSIX `rename` over an open file succeeds). Harmless elsewhere, but undocumented as Windows-specific. |
| **CUDA-only hard requirement** | `training.py:131` | Excludes Apple Silicon (`mps`) entirely, and any CPU-only machine, with no degraded mode. |
| **`sys.stdout.encoding` dance** | `data.py:120-121` | A cp1252-console workaround. Correct, but signals the Windows assumption. |

### 6.4 Configuration and hard-coded constants

There is **no configuration module**. Constants are scattered and duplicated:

| Constant | Occurrences |
|---|---|
| Colours `#000000`/`#f2f2ee`/`#858781`/`#343632` | `app.py:20-23` (good — the only centralised group) |
| Image size `512` | `app.py:109`, `:587`, `:760`, `data.py` (implicit), `training.py:113`, `:139`, `:146` |
| Thumbnail tile `64` | `data.py:211`, `app.py:436`, `:744`, `:754-755`, `training.py:219` |
| Atlas columns `20` | `data.py:211` vs literal `20` at `app.py:436` |
| Result atlas columns `32` | `training.py:219` vs `app.py:744`, `:764` |
| Display grid columns `5` | `app.py:744`, `:756` |
| Pad colour `(238, 236, 230)` | `data.py:153`, `:213` |
| Loss weights `30`, `10` | `training.py:194` |
| Learning rates `2e-4`, `1e-4` | `training.py:134-135` |
| Seed `42` | `data.py:101`, `training.py:132`, `:180`, `data.py:137` |
| Epoch bounds `1..200` | `app.py:332`, `app.py:681`, `training.py:91` (triplicated) |
| Minimum artworks `8` | `training.py:88`, `data.py:138` |
| Poll interval `1000 ms` | `app.py:848` |

None of these are settable without editing source. A single `config.py` (or a dataclass loaded from TOML) would remove most of the duplication-induced fragility.

### 6.5 Reproducibility summary

| Layer | Reproducible? |
|---|---|
| Dataset selection | **No** — upstream CSVs pinned to `main` HEAD (`data.py:28-29`); see §4.1. |
| Download completeness | **No** — silent partial samples on network failure (`data.py:123-127`). |
| Feature extraction | Yes, given identical inputs and `--size` — but coordinates are dataset-relative (`data.py:173`). |
| Training data order | Yes — per-epoch seeded permutation (`training.py:180`). |
| Model initialisation | Yes — `torch.manual_seed(42)` (`training.py:132`). |
| Numerical results | **No** — no `cudnn.deterministic`, no `use_deterministic_algorithms`, fp16 autocast with dynamic loss scaling. |
| Dependency versions | **No** — `>=` constraints only, no lockfile. |
| Run provenance | Partial — `current.json` records `run_id`, `excluded`, `prompt`, `structure`, `epochs`, `device`; but **not** the package version, the git SHA, the dataset manifest, or the torch/CUDA versions. |

---

## 7. Prioritised remediation backlog

Effort key: **S** ≤ 2 h · **M** ≤ 1 day · **L** 2–4 days · **XL** > 1 week.

### P0 — Correctness and user-visible failure

| # | Change | Files (lines) | Finding | Effort |
|---|---|---|---|---|
| P0-1 | Return `set()` instead of all ids when a prompt tokenises to nothing, so non-ASCII/punctuation queries do not silently train on the whole collection. Keep the empty-prompt path explicit at the call sites. | `training.py:33-34`; callers `app.py:631`, `:635`, `:659-671` | C-1 | **S** |
| P0-2 | Make the worker thread fail loudly: move `run`/`frames`/`state` setup inside the `try` (or add an outer handler), always write a terminal `{"state":"error"}` including a traceback, and install a `threading.excepthook` that also writes it. | `training.py:124-127`, `:240-242` | C-5 | **S** |
| P0-3 | Make stale run state recoverable: write `pid` + a monotonic `heartbeat` into `current.json`; treat a run whose PID is dead or whose heartbeat is stale as `interrupted` so GENERATE re-enables. Add a "reset" affordance to the UI. | `training.py:64-77`, `:96-104`, `:196-202`; `app.py:819-848`, `:689-691` | C-6 | **M** |
| P0-4 | Check `_stop.is_set()` inside the mean-accumulation loop and the generation loop, and skip straight to the terminal state. | `training.py:140-144`, `:223-233` | C-4 | **S** |
| P0-5 | Guard every startup data load: catch missing/corrupt `works.json` and `records.jsonl` and show a `messagebox` telling the user to run `whose-mean-data collect`, instead of dying silently under `pythonw`. Replace `SystemExit` in `load_records` with a typed exception. | `app.py:246-256`, `:861-882`; `data.py:142-146` | C-12 | **S** |
| P0-6 | Fix the export path: existence-check the latest frame, wrap the whole export in `try/except` with a `messagebox`, remove the orphan destination on failure, and move the `copytree` off the UI thread with progress. | `app.py:769-804` | C-8 | **M** |
| P0-7 | Make the Windows install actually produce a CUDA build, or degrade gracefully. Document the `--index-url https://download.pytorch.org/whl/cu121` step, and add a CPU/`mps` fallback (slow but functional) instead of the hard `RuntimeError`. | `README.md:39-44`; `requirements.txt`; `pyproject.toml:15`; `training.py:130-131` | C-17, H-4 | **S** |
| P0-8 | Make scrolling work off Windows: bind `<Button-4>`/`<Button-5>` alongside `<MouseWheel>` and normalise `event.delta` (Windows ±120, macOS ±1, X11 button number) through one helper. | `app.py:34-38`, `:105`, `:170-171`, `:193-194`, `:237-238`, `:375`, `:454`, `:602-607` | §6.3 | **S** |

### P1 — Performance, data integrity, and misleading claims

| # | Change | Files (lines) | Finding | Effort |
|---|---|---|---|---|
| P1-1 | Stop rebuilding the canvas every frame. Create the 1,000 canvas items once; on interaction use `coords()`/`itemconfigure()`/`tag_raise()`. Pre-render the hover/select/match thumbnail sizes once into `PhotoImage` caches instead of per-frame LANCZOS. Coalesce drag/hover redraws with `after_idle`. | `app.py:478-535`, `:428-443`, `:551-556`, `:593-600` | §3.1 | **L** |
| P1-2 | Debounce keyword input (~200 ms), cache the parsed `works.json` plus an inverted token index, and vectorise or cap the MST. Measured 445 ms per keystroke at n=1000. | `app.py:347`, `:634-653`; `training.py:31-47` | §3.2, C-2 | **M** |
| P1-3 | Persist the thumbnail atlas to disk during `visualize` and load it at startup instead of re-decoding 1,000 JPEGs on every launch. Add a splash/progress while the UI builds. | `data.py:206-220`, `:158-203`; `app.py:428-443` | §3.3 | **M** |
| P1-4 | Add a real `Dataset`/`DataLoader` with a pre-resized 512 px cache (memmapped `.npy` or a cached JPEG dir), `num_workers>0`, pinned memory, and batch size ≥4. Reuse the cache for the mean pass. | `training.py:112-117`, `:139-145`, `:177-202`; new `dataset.py` | §3.4 | **L** |
| P1-5 | Fix feature extraction to ignore the letterbox pad (mask the contained region), and use one consistent luminance definition across `data.py` and `training.py`. Re-run `visualize`. | `data.py:149-173`; cf. `training.py:115`, `:148` | §4.3 | **M** |
| P1-6 | Make the NGA sample genuinely reproducible: pin an upstream commit SHA, record CSV SHA-256 + the ordered id list in `manifest.json`, backfill failed downloads from the pool, and add `--manifest` replay. Correct the README claim. | `data.py:28-29`, `:98-139`; `README.md:53` | §4.1 | **M** |
| P1-7 | Make image downloads atomic (`.part` + `replace`), catch `BaseException` for cleanup, and verify pre-existing files (or record a size/hash) rather than trusting `exists()`. | `data.py:107-127`, `:33-43` | C-15 | **S** |
| P1-8 | Cut checkpoint write volume: save `model_latest.pt` every N epochs, save `model_best.pt` only on improvement of a **validation** metric, and add a retention policy for `outputs/nga/live/run_*`. 2.68 GB per 20-epoch run today. | `training.py:209-211` | §3.4 | **S** |
| P1-9 | Append one tile to the epoch board instead of re-decoding every epoch PNG on every change (currently O(E²) decodes on the UI thread). | `app.py:711-724`, `:839` | §3.5 | **M** |
| P1-10 | Align the conceptual claim with the objective: show `mean_target.png` beside the generated frame, relabel the panel to distinguish the *arithmetic* mean from the *generated* mean, and state in the UI that the displayed result is a colourisation of a blended grayscale input. | `app.py:337`, `:370`, `:395-411`, `:833`; `README.md:5`, `:80` | §5.2, §5.3 | **S** |
| P1-11 | Add a validation split, report validation L1, select checkpoints on it, and add horizontal-flip augmentation. Mask the pad out of `L1`/`edge_loss`. | `training.py:129`, `:193-194`, `:204-213`; `model.py:78-87` | §5.4 | **M** |
| P1-12 | Index `assistivetext` (and `creditline`) in the keyword search, and carry them into `works.json` so "concept keywords" actually match depicted content. Fix the pluralisation heuristic (stem *and* inflect). | `data.py:174-194`; `training.py:37-43` | C-2, C-3 | **S** |
| P1-13 | Propagate full provenance into exports: put artist, credit line, source URL and an explicit licence string into `records.json`/`export_info.json`, and add a `LICENSE`/`ATTRIBUTION.md`. | `training.py:235`; `app.py:797-800`; `data.py:76-94`; repo root | §4.5 | **S** |
| P1-14 | Close PIL images deterministically (`with Image.open(...) as im:`) at all five UI load sites. | `app.py:702`, `:722`, `:740`, `:762`, `:817` | C-7 | **S** |
| P1-15 | Retry/backoff for all network calls (CSV + images), honour `Retry-After`, raise the default `--delay`, add a `--retry-failures` mode, and put contact info in the `User-Agent`. | `data.py:33-43`, `:106-127`, `:228` | §4.2 | **M** |

### P2 — Hygiene, structure, and polish

| # | Change | Files (lines) | Finding | Effort |
|---|---|---|---|---|
| P2-1 | Introduce a test suite. Start with the pure-logic units that need no display or GPU: `matching_ids`, `build_keyword_edges`, `candidates`, the feature maths, `_write` atomicity, `status()` state machine. Add `pytest` as a dev extra. | new `tests/`; `pyproject.toml` | §6.1 | **M** |
| P2-2 | Add `ruff` (lint + format) and `mypy`/`pyright` config; annotate `app.py` and `training.py` (currently 0/69 and 0/9 annotated defs). Normalise the semicolon one-liner style. | `pyproject.toml`; `app.py`, `training.py` | §6.1, §1.5 | **M** |
| P2-3 | Add a GitHub Actions workflow running lint + type check + the new tests on Linux and Windows. | new `.github/workflows/ci.yml` | §6.1 | **S** |
| P2-4 | Replace `print`/silent-traceback diagnostics with the `logging` module, writing to a rotating file under `outputs/` so `pythonw` failures are diagnosable. | all modules | §6.1 | **S** |
| P2-5 | Extract a `RunLayout` helper and a `config.py` for the ~15 duplicated constants (tile 64, atlas cols 20/32, grid cols 5, size 512, pad colour, epoch bounds, seeds, loss weights). | `app.py`, `training.py`, `data.py`; new `config.py` | §1.5, §6.4 | **M** |
| P2-6 | Decompose `Studio` (886 lines, ~40 attributes): split scene rendering, the run/status client, and export into separate classes so the non-Tk parts become testable. | `app.py:245-848` | §1.5 | **L** |
| P2-7 | Move `matching_ids`/`CONCEPT_ALIASES` out of `training.py` into `search.py` or `data.py`. | `training.py:24-47` | §1.5 | **S** |
| P2-8 | Fix `smoke_test` to derive expected geometry from the actual work count rather than asserting 1000/[1280,3200]. | `app.py:851-858` | C-18 | **S** |
| P2-9 | Separate "render epoch" from "user selected epoch" so zooming does not silently disable auto-follow. | `app.py:614-618`, `:696-709`, `:840-842` | C-9 | **S** |
| P2-10 | Rebuild the results view when the generated count changes, not only when `run_id` changes; write `records.json` before `generated_atlas.jpg`. | `app.py:735-751`, `:843-845`; `training.py:234-235` | C-10 | **S** |
| P2-11 | Track and reuse `ComparisonViewer`/`ResultViewer` windows the way `detail_window` is tracked, and destroy all children on root close. Confirm `unbind(sequence, funcid)` behaviour on the project's minimum Python (3.11) and work around it if needed. | `app.py:116-153`, `:767`, `:817` | C-11 | **S** |
| P2-12 | Prefer the front-most work in `nearest()` by breaking ties on `z`. | `app.py:536-542` | C-13 | **S** |
| P2-13 | Per-image error handling in `prepare_interactive` and `thumbnail_atlas` so one bad file does not abort the command or the UI startup. | `data.py:158-171`, `:206-220` | C-16 | **S** |
| P2-14 | Accumulate losses on-device and call `.item()` once per epoch; throttle status writes to time-based intervals and stop re-serialising `history` every write. | `training.py:196-202` | §3.4 | **S** |
| P2-15 | Replace `ConvTranspose2d` upsampling with `Upsample + Conv` to remove checkerboard artefacts; consider adding dropout so the generator is not purely deterministic. | `model.py:22-26`, `:34-39` | §5.4 | **S** |
| P2-16 | Add `[project.optional-dependencies]`, a lockfile, a `gui_scripts` entry point for `whose-mean`, a `LICENSE`, and POSIX setup/run instructions plus a shell launcher. Provide a font fallback chain instead of bare `Consolas`. | `pyproject.toml`; `requirements.txt`; `README.md`; new `run.sh`; `app.py` fonts | §6.2, §6.3 | **S** |
| P2-17 | Remove dead state and fix small inconsistencies: `epoch_refs`, `detail_photo`, `connected`, the unrounded slider callback, unused `saturation`. Allow-list URL schemes before `urlopen`. | `app.py:275`, `:278`, `:643`, `:649`, `:65-67`; `data.py:169`, `:110` | C-20 | **S** |
| P2-18 | Add `cudnn.deterministic`/`use_deterministic_algorithms` behind a `--deterministic` flag, and record package version, git SHA, torch/CUDA versions and the dataset manifest hash in each run's state. | `training.py:132`, `:96-100` | §6.5 | **S** |

---

## Summary

The project has a **sound skeleton**: a genuinely good thread/UI decoupling (filesystem-mediated, atomic writes, no shared mutable state), a clean and fully-typed model module, and a respectable provenance record from the NGA open data. The problems cluster in four places.

1. **The conceptual claim does not match the objective.** The network is a grayscale→RGB colouriser trained on individual artworks with a 30× L1 term; the "mean image" is that colouriser applied to a blend of two *grayscale* images, 65 % of which is a single artwork by default. The true arithmetic mean is computed exactly at `training.py:139-145` and requires no training at all.
2. **The feature space measures the wrong thing.** Every axis is contaminated by the letterbox pad — calculated at 33 % of pixels for a 3:2 work — so the Y axis substantially encodes aspect ratio, and the two luminance definitions in the codebase disagree.
3. **Failures are invisible.** A dead worker thread, a crashed process, or a missing dataset all produce either a permanently disabled GENERATE button or a window that vanishes without a message, because the app ships under `pythonw.exe` with no logging.
4. **Almost everything runs on the UI thread**, including 1,000 JPEG decodes at startup, a 445 ms MST per keystroke, a full canvas rebuild per mouse-move, O(E²) epoch-image decodes, and a multi-gigabyte export copy.

Plus the hygiene baseline: **no tests, no linting, no type checking, no CI, no logging**, and a hard Windows/CUDA dependency that the documented `pip install -e .` will not actually satisfy on Windows.

None of these are fatal to the piece. The P0 set is about eight person-days and turns a fragile demo into something that fails legibly; P1 is where the artwork's intellectual honesty and its interactive feel are recovered.
