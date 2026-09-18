# Whose Mean? — Iteration 1 Model Landscape

**Replacing pix2pix: generative backends, feature spaces, and what each one says an "average" is**
Peiyan Zou · ADV 9672 · branch `iteration1` · September 2026

---

## 0. Evidence quality

This survey mixes three kinds of claim. They are labelled throughout.

- **[V]** — verified this pass against a primary source (paper, model card, vendor announcement, official repo).
- **[S]** — from secondary sources (technical blogs, GPU-guide sites). Directionally reliable, numerically approximate.
- **[E]** — my own estimate, reasoned from architecture and parameter count. Not measured.

VRAM and speed numbers for local inference are hardware-, precision-, resolution- and runtime-dependent. Treat every number below as a planning figure, and benchmark before committing (see Phase 0 in `ITERATION1_PLAN.md`).

---

## 1. What we are actually replacing

The current backend is a from-scratch 512 px pix2pix **colouriser** (grayscale → RGB), batch size 1, no pretraining, trained for ~20 epochs on ≤ 1000 NGA images, with the "mean image" computed outside the network as an arithmetic pixel average. See `ITERATION1_RESEARCH.md` §1 for the full read.

So the replacement has to satisfy four requirements, in priority order:

1. **Sub-second feedback on a dataset edit.** The core gesture is "remove these works, see what changes". Interactive ML degrades badly above a second (Amershi et al. 2014). Today it is on the order of an hour **[E]** — and the bottleneck is not the network but the data path: 20 000 redundant JPEG decodes per default run, batch size 1, plus an uncached full-collection decode before every run (`ITERATION1_CODE_AUDIT.md` §3.4). **A faster model on the same data path will not fix this.** The I/O and caching work in §3.4 / §4.1 of that audit is a hard prerequisite for any claim made below.
2. **A learned prior.** The whole thesis is about statistical likeness. A model that has never seen anything but 1000 NGA images cannot produce one; a pretrained model conditioned on those 1000 images can.
3. **The "mean" must be a first-class computation, not a preprocessing artefact.** It should be something the model consumes (an embedding, a token, a set of weights), not a blurred JPEG fed to a colouriser.
4. **Runs on one consumer GPU, offline, reproducibly, at an exhibition.** No API key in a gallery, no internet dependency during a show.

---

## 2. Feature / embedding spaces (the cheap layer — do this first)

This layer costs nothing at runtime and is where most of the value is. Embed all 1000 works once, cache to disk, and every subsequent set operation is milliseconds of NumPy.

| Encoder | Sizes | Licence | VRAM (inference) | 1000-image embed time | Why it matters here |
|---|---|---|---|---|---|
| **SigLIP 2** (Tschannen et al. 2025) | ViT-B 86 M, L 303 M, So400m 400 M, g 1 B **[V]** | **Apache 2.0** **[V]** | ~1–3 GB for So400m fp16 **[E]** | seconds on GPU, ~1–3 min CPU **[E]** | Best open text↔image joint space; 109 languages **[V]**; gives free semantic search *and* a centroid that is meaningful across modalities |
| **CLIP ViT-L/14, ViT-H/14 (OpenCLIP)** | 428 M / 1 B | MIT / Apache-2.0 depending on checkpoint **[S]** | ~1–2 GB fp16 **[E]** | seconds **[E]** | The space IP-Adapter and unCLIP are *trained against* — required if you want to decode a centroid |
| **DINOv3** (Siméoni et al. 2025) | up to 7 B; smaller distilled backbones released **[V]** | **Meta commercial licence, registration + approval required — not Apache-2.0 like DINOv2** **[V]** | varies | — | Image-only, self-supervised, excellent dense/structural features; good *second* perceptual topology to contrast with SigLIP |
| **DINOv2** | ViT-S/B/L/g | Apache 2.0 **[V, by contrast in DINOv3 coverage]** | ~1–2 GB **[E]** | seconds **[E]** | The licence-clean fallback if DINOv3 registration is unacceptable for an exhibited artwork |

**Recommendation: SigLIP 2 So400m as the primary space, OpenCLIP ViT-H as the secondary (because IP-Adapter needs it), DINOv2 as an optional third "way of seeing".** Offering a switch between two or three spaces is not feature creep — it is the direct implementation of Offert & Bell's *perceptual bias*: the average moves when the perceptual topology changes, and the user can watch it move.

This layer alone replaces: the three photometric axes, the hard-coded synonym search, and the minimum-spanning-tree over warmth/luminance/edge. It also gives instant medoid, centroid, k-means sub-means, nearest-neighbour retrieval, and leave-one-out attribution — all in NumPy, all under 10 ms for N = 1000.

---

## 3. Generative backbones

### 3.1 Comparison

| Model | Params | Released | Licence | VRAM (local inference) | Speed @ 1024² | 1000-image LoRA cost | Verdict for this project |
|---|---|---|---|---|---|---|---|
| **SDXL 1.0** (Podell et al., ICLR 2024 **[V]**) | 2.6 B UNet + 2 text encoders | 2023 | CreativeML OpenRAIL++-M — permissive, includes use restrictions **[S]** | ~8 GB fp16 UNet + 1–2 GB VAE/TE **[S]** | ~2–5 s @ 25–30 steps; **~0.5 s with SDXL-Turbo at fp16** **[S]**; 1–4 steps with LCM-LoRA **[S]** | 1–3 h on 24 GB **[E, scaled from FLUX figures]** | **Primary recommendation.** Only backbone with a fully mature IP-Adapter + ControlNet + LoRA + LCM/Turbo stack, all runnable together on one 12–24 GB card |
| **SD 1.5** | 0.86 B UNet | 2022 | CreativeML OpenRAIL-M **[S]** | ~4–6 GB fp16 **[S]** | < 0.5 s with LCM **[S]** | 30–90 min on 12 GB **[E]** | **Low-VRAM fallback.** Ugly by 2026 standards, but the IP-Adapter ecosystem is deepest here and it will run on a gallery laptop |
| **SD 3.5 Large** | 8.1 B | 2024 | Stability AI **Community License** — free commercial under **$1 M** annual revenue **[V]** | ~18 GB fp16, ~11 GB fp8 **[V, vendor/NVIDIA figure]** | ~5–10 s **[E]** | 3–8 h on 24 GB **[E]** | Better text and prompt adherence than SDXL; thinner control-adapter ecosystem. Not worth the VRAM here |
| **SD 3.5 Medium** | 2.5 B | 2024 | same Community License **[V]** | ~9.9 GB excluding text encoders **[V]** | ~3–6 s **[E]** | 1–3 h **[E]** | Reasonable, but no advantage over SDXL for image-conditioned work |
| **FLUX.1 [dev]** | 12 B | 2024 | FLUX.1 **Non-Commercial** License **[V]** | ~20–24 GB fp16, ~12 GB fp8 **[S]** | ~5–15 s **[E]** | 1–3 h on 24 GB for 20–30 images at rank 16 **[V, vendor-adjacent blog]**; longer for 1000 | Excellent quality. Non-commercial licence is survivable for a student artwork but is a real constraint for exhibition/sale |
| **FLUX.1 Kontext [dev]** | 12 B rectified-flow transformer **[V]** | Jun 2025 | FLUX.1 Non-Commercial **[V]** | ~20–24 GB fp16 **[S]** | ~5–15 s **[E]** | as above | **Interesting for a specific feature:** instruction-based in-context editing ("make this mean warmer", "remove the gilt frames") without retraining. Keep as a stretch goal |
| **FLUX.2 [dev]** | **32 B** **[V]** | 25 Nov 2025 **[V]** | FLUX.2-dev **Non-Commercial** **[V]** | ~32 GB at **fp8** — A100 80 GB / H100 practical minimum **[S]** | — | infeasible locally | **Out of scope.** Cannot run on a desktop studio |
| **FLUX.2 [klein] 4B** | 4 B **[V]** | 15 Jan 2026 **[V]** | **Apache 2.0** **[V]** | ~13 GB **[S]** | fast (distilled) **[V, vendor claim]** | **[E]** likely 1–3 h on 24 GB | **Strongest licence-clean 2026 option.** Watch this one: if its IP-Adapter / control ecosystem matures during the project, it becomes the better primary |
| **FLUX.2 [klein] 9B** | 9 B **[V]** | Jan 2026 **[V]** | Non-commercial **[V]** | ~29 GB fp16 **[S]** | — | — | Wrong point on the curve for us |
| **Qwen-Image** | 20 B MMDiT + 8.3 B Qwen2.5-VL text encoder **[V]** | Aug 2025 | **Apache 2.0** **[V]** | ~40 GB+ fp16; GGUF quantisations exist for consumer cards **[S]** | slow locally **[E]** | expensive **[E]** | Best *licence* of the large models and outstanding text rendering — irrelevant to us. Too heavy |
| **Qwen-Image-Edit-2509** | same base | Sep 2025 **[V]** | Apache 2.0 **[V]** | as above | — | — | Multi-image editing and **native ControlNet-style conditions (keypoints, sketch)** **[V]**. A credible future editing path if VRAM allows |

### 3.2 Controllability and personalisation layers

| Technique | Paper | What it buys us | Cost |
|---|---|---|---|
| **IP-Adapter** | Ye et al. 2023, arXiv:2308.06721 **[V]** | Conditions a **frozen** diffusion model on an *image embedding* via decoupled cross-attention **[V]**. This is the key primitive: feed it the **centroid of the selection** and you have generated a mean from a set, with zero training | ~100 MB adapter; +~1 GB VRAM **[E]**; no training |
| **ControlNet** | Zhang, Rao & Agrawala, ICCV 2023 **[V]** | Spatial conditioning (canny, depth, scribble) via zero-convolutions; robust from < 50 k to > 1 M training images **[V]** | ~1.4 GB per adapter fp16 **[E]**; no training (use released adapters) |
| **T2I-Adapter** | — | Lighter alternative to ControlNet | smaller, weaker **[E]** |
| **LoRA** | Hu et al., ICLR 2022, arXiv:2106.09685 **[V]** | Low-rank trainable matrices injected into a frozen model; ~10 000× fewer trainable params and ~3× less GPU memory than full fine-tuning **[V]** | SDXL rank 16–32: a few tens of MB; **[S]** 24 GB is the practical minimum for comfortable FLUX.1 LoRA training at 512–1024 px with fp8 + gradient checkpointing; Kohya's fused backward pass brought peak VRAM to ~16 GB at 1024² |
| **DreamBooth** | Ruiz et al., CVPR 2023 **[V]** | Binds an identifier to a subject with class-prior-preservation loss **[V]** | Heavier than LoRA; usually combined with it |
| **Textual Inversion** | Gal et al., ICLR 2023 **[V]** | Learns a new pseudo-word in text-encoder embedding space from **3–5 images** **[V]**; the base model stays frozen | Minutes to ~1 h; ~a few KB of learned embedding **[E]** |

### 3.3 API options

| Service | Price per image | Verdict |
|---|---|---|
| **Gemini 3 Pro Image ("Nano Banana Pro")** | ~$0.134 at 1K–2K, ~$0.24 at 4K; ~$0.067 / $0.12 via Batch API **[S]** | Excellent quality, zero setup |
| **Gemini 3.1 Flash Image** | ~$0.045 @ 512 px up to ~$0.15 @ 4K **[S]** | Cheaper tier |
| **Imagen 4 Fast** | ~$0.02 **[S]** | Cheapest official Google option |
| **OpenAI GPT-Image** | varies by size/quality; not confirmed this pass | — |

**Recommendation: do not use an API as the primary backend.** Three reasons, in descending order of importance.

1. **Conceptual.** The piece's argument is that the output is a function of *this* dataset. A closed model trained on the open web reverses that: the dataset becomes a nudge on a prior that swamps it. You cannot honestly ask "whose mean?" of a model whose training set is undisclosed — which is, itself, the point Crawford & Paglen make.
2. **Exhibition.** Network dependency, rate limits, and per-render cost in a room where visitors click GENERATE repeatedly.
3. **Cost of the interaction model.** The instant-preview loop implies hundreds of renders per session.

**But do add one API as a labelled comparison mode** — a "reference average" button that shows what a general-purpose model produces from the same text description of the collection. Placing that next to the four local means is a strong rhetorical move and costs ~$0.13 per press.

---

## 4. Reframing the mean: four computations, four claims

This is the conceptual core of the iteration. Each option is technically cheap; the value is in their disagreement.

### M1 — Pixel mean (retain, reframe)
`mean = Σ Iᵢ / N` in sRGB. **Claim: the average is optical superposition.** Direct descendant of Galton's composite photography. Cost: O(N) image reads, ~1–3 s for 1000 works at 512 px, cacheable and incrementally updatable (`mean_new = (N·mean − I_removed) / (N−1)` — **an exact O(1) update on removal**, which makes it instant after the first computation). **Keep this. Label it as the historical baseline.**

*Required fix:* eliminate the `(238, 236, 230)` letterbox pad. Either centre-crop to square, or accumulate a per-pixel coverage mask and divide by it, so the mean is not partly a picture of the preprocessing function.

### M2 — Medoid (retain, promote)
`argmin_i Σ_j d(eᵢ, e_j)` in embedding space. **Claim: the average is a real member who stands in for the rest.** Daston & Galison's *truth-to-nature*. Already half-implemented as the "structure anchor" — but computed over three photometric scalars, which makes it the work with the most median warmth, not the most representative work. Recompute in SigLIP space and it becomes meaningful. Cost: < 10 ms. **Promote it from a hidden slider term to a named mean.**

### M3 — Embedding centroid, decoded (new — primary)
`c = Σ eᵢ / N` (optionally L2-renormalised), then generate conditioned on `c` via IP-Adapter. **Claim: the average is a point in perceptual space that no work occupies — a synthetic exemplar.** This is *mechanical objectivity relocated to feature space*, and it is the most direct modern realisation of Steyerl's "statistical rendering". Precedent: unCLIP (Ramesh et al. 2022) establishes that a CLIP image embedding can be decoded to an image; IP-Adapter makes it cheap on a frozen model.

Cost: centroid is < 1 ms. Generation is ~0.5–5 s depending on distillation. **Two-stage feedback is the key UX trick:** on every edit, instantly show the *k* nearest real works to the new centroid (a retrieval-based mean, 10 ms, no GPU), then let the diffusion render land a second or two later.

Why this is the right primary: it is the only option where the mean is a **first-class conditioning signal** and where the feedback loop is fast enough for genuine interaction.

### M4 — Learned likeness: LoRA on the selection (new — the slow one)
Fine-tune a LoRA on the selected subset. **Claim: the average is a disposition to produce, not a picture.** Daston & Galison's *trained judgment*. This is the only option that literally re-enacts "training a model on a dataset", and therefore the only one that earns the pix2pix rhetoric the current build gestures at — with a pretrained prior underneath, so the result is legible instead of muddy.

Cost: the honest number is tens of minutes to a few hours on a 24 GB card for 1000 images. **[E, scaled from [S] figures of 1–3 h for 20–30 images at rank 16 on a 4090 — note that step count, not image count, dominates, so a 1000-image run at the same step budget costs the same; the difference is epochs-over-data, not wall clock.]** Practically: fix a step budget (e.g. 1500–3000 steps), not an epoch count.

**Keep the slowness visible.** This is the Turkle move: the expensive mean should look expensive.

### M5 — Textual inversion: the collection as one word (new — the sharp one)
Learn a single pseudo-token `<nga-selection>` for the current subset (Gal et al. 2023). **Claim: the average is a *name*.** The collection is compressed into a category that can then be composed: `"a portrait of a dog in the style of <nga-selection>"`. Conceptually this is the sharpest of the five, because it makes the collection do what Crawford & Paglen say training sets do — become a *taxonomy*, a label that renders some things intelligible and others invisible.

Cost: minutes; a few KB. Officially demonstrated on 3–5 images **[V]**, so behaviour on 1000 is a research question, not a settled recipe — but "what happens when you try to name a thousand things with one word" is a good question for an artwork to ask.

### Comparison

| | M1 Pixel mean | M2 Medoid | M3 Centroid + IP-Adapter | M4 LoRA | M5 Inverted token |
|---|---|---|---|---|---|
| What the average *is* | superposition | a real member | a point in feature space | a disposition | a name / category |
| Epistemic virtue (Daston & Galison) | mechanical objectivity (optical) | truth-to-nature | mechanical objectivity (statistical) | trained judgment | — (taxonomy; Crawford & Paglen) |
| Latency after an edit | **O(1) incremental**, instant | < 10 ms | 10 ms preview + ~0.5–5 s render | 20 min – 3 h | ~5–40 min |
| Needs a GPU | no | no | yes | yes | yes |
| Needs training | no | no | no | yes | yes |
| Depends on a prior trained elsewhere | no | no | **yes** | yes | yes |
| Honest about "whose"? | totally (it is only this data) | totally | partly — the prior contributes | partly | partly |
| Failure mode | brown blur | arbitrary-looking single work | prior swamps the set; generic "museum painting" | overfit; mode collapse onto a few works | token fails to converge; drift |

That last-but-one row is the argument the interface should stage. M1 and M2 are *entirely* the user's data and look impoverished. M3–M5 look like art and are substantially the prior's doing. **The more beautiful the mean, the less of it is yours.** That sentence is the piece.

---

## 5. Recommendation

### Primary stack

```
Feature layer   SigLIP 2 So400m (Apache 2.0)  + OpenCLIP ViT-H (for IP-Adapter compat)
                → embeddings cached to data/nga/embeddings.npz
Generative      SDXL 1.0 base, fp16
                + IP-Adapter Plus (ViT-H image encoder)   → M3, centroid conditioning
                + ControlNet (canny or depth)             → honest "structure anchor"
                + LCM-LoRA or SDXL-Turbo                  → 1–4 step preview renders
                + PEFT LoRA training                      → M4, learned likeness
                + Textual Inversion                       → M5, collection-as-token
Historical mode Keep pix2pix (model.py, training.py) intact as a labelled "2017 mode"
Reference mode  Optional single API call (Gemini 3 Pro Image) as a labelled comparison
```

**Why SDXL over everything newer.** Not because it is the best model — it is not. Because it is the only backbone in 2026 where IP-Adapter, ControlNet, LoRA training, and 1–4-step distillation all exist, all work together, and all fit on one 12–24 GB consumer card simultaneously. The project's binding constraint is *interaction latency under a curation loop*, not image quality. SDXL-Turbo at ~0.5 s for 1024² on a 4090 **[S]** is what makes the loop feel like an instrument. FLUX.2 [dev] at 32 B and ~32 GB fp8 **[S]** would make it feel like a render farm.

**Fallbacks, in order.**
1. **< 10 GB VRAM or a gallery laptop:** SD 1.5 + IP-Adapter + LCM. Lower fidelity, sub-second, deepest adapter ecosystem.
2. **No usable GPU at all:** ship M1 + M2 + retrieval-based M3 only. This is a fully coherent artwork — it just does not generate. Do not treat this as failure; ~70 % of the conceptual payload survives, because the pixel mean, the medoid, the provenance panel and the attribution ranking all run on CPU.
3. **Licence-clean requirement (sale, commission, commercial exhibition):** FLUX.2 [klein] 4B, Apache 2.0, ~13 GB **[V licence, S VRAM]**. Re-evaluate at the start of Phase 3 — its adapter ecosystem is the open question.
4. **Quality ceiling needed for a specific output (print, poster):** one-off render via FLUX.1 Kontext [dev] or an API, clearly labelled as such.

### Hardware reality check

| Available GPU | What runs | Loop latency | What to cut |
|---|---|---|---|
| **RTX 4090 / 5090, 24–32 GB** | Everything: SDXL + IP-Adapter + ControlNet + Turbo, LoRA training in-session | ~0.5–2 s render; LoRA 20–60 min | nothing |
| **RTX 4070 Ti / 3090, 12–24 GB** | SDXL fp16 + IP-Adapter + one ControlNet; LoRA training at 512–768 px | ~1–4 s; LoRA 1–3 h | drop simultaneous ControlNet + Turbo; train LoRA overnight |
| **8–12 GB** | SDXL fp8/fp16 with sequential CPU offload, or SD 1.5 full stack | SDXL ~5–15 s; SD1.5 < 1 s | drop M4 in-session; pre-bake LoRAs offline |
| **6–8 GB** | SD 1.5 + IP-Adapter + LCM only | < 1 s | M4 and M5 pre-baked only |
| **CPU only / Apple Silicon** | Embeddings (slow but fine), M1, M2, retrieval-M3 | instant after warm-up | all diffusion |

**Action required:** the README says "a CUDA-capable NVIDIA GPU" without naming one. Establish the actual target hardware — including the exhibition machine, which is often *not* the development machine — before Phase 3. This single unknown determines half the plan.

### Licence summary for an exhibited artwork

| Component | Licence | Exhibition-safe? |
|---|---|---|
| NGA images + metadata | **CC0** **[V]** | Yes, unconditionally |
| SigLIP 2 | Apache 2.0 **[V]** | Yes |
| DINOv2 | Apache 2.0 **[V]** | Yes |
| DINOv3 | Meta commercial licence, registration + approval **[V]** | Read it first; avoid if the work may be sold |
| SDXL | CreativeML OpenRAIL++-M **[S]** | Yes, with use restrictions — read them |
| SD 3.5 | Stability Community License, free under $1 M revenue **[V]** | Yes for a student work |
| FLUX.1 / FLUX.1 Kontext / FLUX.2 [dev] / klein 9B | **Non-commercial** **[V]** | Non-commercial exhibition only; not for sale |
| FLUX.2 [klein] 4B | **Apache 2.0** **[V]** | Yes |
| Qwen-Image / Qwen-Image-Edit | Apache 2.0 **[V]** | Yes |

The CC0 source data is a genuine asset. Say so in the work.

---

## 6. What to say in the wall text

> Five averages of the same thousand pictures.
> One is a photograph of all of them at once.
> One is a single picture chosen to stand for the rest.
> One is a point in a machine's idea of similarity, rendered by a model that has seen a billion other pictures.
> One is a set of weights that learned a habit.
> One is a word.
> Only the first two are made entirely of this collection. They are also the ugliest.
