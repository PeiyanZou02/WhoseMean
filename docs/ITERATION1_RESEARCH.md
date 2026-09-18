# Whose Mean? — Iteration 1 Research

**Literature review, conceptual deepening, and annotated bibliography**
Peiyan Zou · ADV 9672 · branch `iteration1` · September 2026

---

## 0. How to read this document

Section 1 is a precise, code-level account of what the current build actually does — not what the README says it does. It is deliberately conceptual in emphasis; for the line-by-line engineering treatment (20 concrete defects, measured timings, remediation backlog) see the companion `ITERATION1_CODE_AUDIT.md`, produced in parallel. Where the two documents touch the same code, the audit's measured numbers take precedence over the estimates here, and I have adopted them below. Section 2 is the conceptual diagnosis that follows from that account. Sections 3–9 are the literature, organised by the design decision each body of work informs. Section 10 is the annotated bibliography with full citations. Section 11 flags what I could **not** verify.

Every citation in Section 10 was checked against a primary or authoritative secondary source during this research pass. Where a bibliographic detail (usually a journal name) could not be confirmed, it is marked `[UNVERIFIED VENUE]` rather than guessed.

---

## 1. What the system currently is

### 1.1 Data pipeline (`data.py`, 239 lines)

| Stage | Implementation | Notes |
|---|---|---|
| Source | `objects.csv` + `published_images.csv` pulled over HTTP from `NationalGalleryOfArt/opendata` on GitHub | No API key, no auth, fully reproducible |
| Object filter | `accessioned == "1"` AND `isvirtual != "1"` AND `classification ∈ {painting, drawing, print, photograph}` | Excludes sculpture, decorative arts, volumes, portfolios — a large, silent curatorial act |
| Image filter | `openaccess == "1"` AND `viewtype == "primary"`, lowest `sequence` wins | One image per object |
| Sample | `random.Random(42).shuffle(pool)`, take first `--limit` (1000) | Reproducible; documented honestly in the README |
| Download | IIIF `.../full/!768,768/0/default.jpg` | Longest side ≤ 768 px |
| Preprocess | `square()`: EXIF-transpose → `ImageOps.contain` to N×N → paste centred onto a solid `(238, 236, 230)` canvas | **Letterboxing with an off-white pad** |
| Features | 4 scalars per work: `mean(R−B)` (warmth), `mean(gray)` (luminance), mean abs. finite difference of gray in x and y (edge density), `mean(max−min channel)` (saturation) | Computed at 96 px |
| Normalisation | z-score, clip to ±2.5, divide by 2.5 → coordinates in [−1, 1] | |
| Output | `works.json` — `x = warmth`, `y = luminance`, `z = edge density`. `saturation` is stored but **never used as an axis**. `mean_weight = 1/N` is stored but **never read anywhere in the codebase** | |
| Atlas | 64 px tiles, 20 columns → 1280 × 3200 JPEG for 1000 works | Hard-coded in `app.smoke_test` |

### 1.2 Model (`model.py`, 87 lines)

`Generator512` is a six-scale U-Net (32→64→128→256→512→512 channels, InstanceNorm, LeakyReLU down / ReLU up, `tanh` output, skip concatenations, 8×8 bottleneck for a 512 px input). `Discriminator512` is a five-layer PatchGAN over the channel-wise concatenation of source and target (6 input channels). `edge_loss` is an L1 on horizontal and vertical finite differences. This is a faithful, compact pix2pix at 512 px — **16.7 M generator parameters, 67.0 MB per fp32 checkpoint** (calculated in `ITERATION1_CODE_AUDIT.md` §3.4) — with no pretraining, no attention, and no text conditioning of any kind.

### 1.3 Training and the "mean" (`training.py`, 242 lines)

This is where the conceptual claim and the computation diverge most sharply.

**The paired task is colourisation.** `_pair()` builds the source as Rec. 601 luma of the target, replicated across three channels. So the network learns `grayscale → RGB` on the included subset. It never sees a "mean" during training.

**The "mean image" is a pixel average.** `mean_rgb = Σ square(img, 512) / N` over the included records — an unweighted arithmetic mean in 8-bit sRGB over letterboxed 512 × 512 canvases. Saved as `mean_target.png`.

**The "structure anchor" is a grayscale blend.**
```
anchor      = argmin_i ||xyz_i − mean(xyz)||²        # medoid in the 3-D photometric space
structured  = (1 − s) · luma(pixel_mean) + s · luma(anchor_image)      # s = 0.65 by default
```
So at the default setting the network's input is 65 % one specific artwork's luminance and 35 % the collection's average luminance. The published "PIX2PIX MEAN" is then `G(structured)` — **a colourisation network, trained on the subset, applied to a blended grayscale composite**. Nothing in the pipeline learns an average.

**Training regime.** Batch size 1, Adam (G: 2e-4, D: 1e-4, β₁ = 0.5), fp16 autocast + GradScaler, one-sided label smoothing (0.9), loss `L_G = BCE_adv + 30·L1 + 10·edge`. Seeds fixed at 42. `RuntimeError` if CUDA is unavailable. Best checkpoint is selected by **training** L1 — there is no validation split, so "best" rewards memorisation. Default 20 epochs × 1000 works = 20 000 generator steps.

**State protocol.** A single `outputs/nga/live/current.json` written atomically (temp file + `replace`, with a 100-attempt retry for Windows `PermissionError`), polled by the UI at 1 Hz.

### 1.4 Interaction (`app.py`, 886 lines)

A dark, Consolas-typeset Tkinter studio. Left: a hand-rolled 3-D scatter of 1000 64 px tiles with a Y-then-X rotation, a `3/(3.8 − z)` perspective divide, painter's-algorithm depth sort, 16 px idle / 28 px matched / 42 px selected / 88 px hovered thumbnails. Right: epoch spinner, keyword field, structure slider, GENERATE/STOP, progress line, mean preview, and a notebook with EPOCH OUTPUTS / ALL RESULTS / MODEL I/O.

Modes: `INSPECT` opens a metadata window; `REMOVE` toggles exclusion. Keyword search tokenises `title + artist + classification + medium`, applies naive `+s`/`−s` pluralisation and **three hard-coded alias sets** (`tree`, `dog`, `sardine`), then draws a minimum spanning tree over matched works using squared XYZ distance. On GENERATE with a non-empty query, the *complement* of the match set is excluded for that run, unioned with manual removals.

### 1.5 Concrete technical limitations

1. **The central claim is not computed.** The "mean image" is an arithmetic pixel mean; the network is a colouriser applied to it. The piece asserts that a model constructs a statistical likeness, but the code never fits a likeness.
2. **Letterbox contamination.** Because aspect ratios vary and padding is a constant `(238, 236, 230)`, a large, fixed fraction of every "mean" is literally the background colour of the preprocessing function. The pixel mean measures the collection's aspect-ratio distribution as much as its imagery.
3. **The feature space is photometric, not semantic.** Three low-level statistics cannot support the question the title asks. "Whose mean?" is a question about people, periods, donors, and geographies; warmth/luminance/edge answer none of them.
4. **Search is string matching with three hand-written synonym sets.** The presence of `'sardine': {'sardine','sardines','fish','fishes'}` in the source is itself evidence: the concept vocabulary is whatever the author needed for a demo.
5. **GAN-from-scratch on ≤ 1000 images cannot learn a prior.** 20 k steps at batch 1 on a small dataset yields a weak, blurry colouriser. Fidelity ceilings are structural, not tunable.
6. **The interaction loop is minutes-to-hours, not seconds.** At 512 px with batch size 1 the GPU is launch-bound rather than compute-bound, and the real bottleneck is not the network at all: `_pair()` re-opens, EXIF-transposes and LANCZOS-resizes the 768 px source JPEG **for every sample of every epoch** — 20 000 redundant decodes for a default run — with no `Dataset`, no `DataLoader`, no prefetch and no cache. A further full 1000-image decode precedes training just to compute the pixel mean, and it is not cached between runs even when the exclusion set is unchanged. (See `ITERATION1_CODE_AUDIT.md` §3.4.) The core gesture of the piece — exclude works, see what changes — is therefore effectively non-interactive. This is the single most serious problem, and it is an HCI problem, not an ML one.
6b. **The UI is slow in its own right, independently of training.** The audit measured the keyword minimum-spanning-tree at **445 ms for 1000 points**, scaling quadratically, recomputed on every keystroke with no debounce alongside a full re-read and re-parse of `works.json`. `draw_scene` tears down and rebuilds ~1000 Tk canvas items on every hover, drag delta and zoom tick, re-allocating a LANCZOS-resized `PhotoImage` per matched work per frame. The thumbnail atlas is memoised in memory only and never persisted, so every cold launch decodes 1000 JPEGs on the Tk main thread before the first frame. `rebuild_epoch_view` re-decodes every epoch PNG on every epoch, i.e. O(E²). None of this needs a new model to fix, and all of it must be fixed before latency claims about the new backends mean anything.
7. **Hard CUDA requirement + Windows-first packaging** (`OPEN_DESKTOP_STUDIO.cmd`, `.venv\Scripts\*.exe`) makes exhibition and peer review fragile.
8. **No comparison.** Runs cannot be placed side by side in the UI. The argument of the piece is comparative, but the interface is not.
9. **No aggregate provenance.** A viewer can inspect one object's credit line but cannot see the composition of the set they are averaging — the piece withholds exactly the information its thesis is about.
10. **Dead affordances.** `mean_weight` and `saturation` are computed and stored but unused; the concept field is labelled "CONCEPT KEYWORDS" although the model has no text conditioning whatsoever; `smoke_test()` hard-codes `1000` works and a `[1280, 3200]` atlas, so any `--limit` other than 1000 fails the test.
11. **Mislabelled control.** The "STRUCTURE ANCHOR" slider reads as *average ↔ structure*, but computes *mean luminance ↔ one artwork's luminance*. Users are being told a different story from the one the code tells.

---

## 2. Conceptual diagnosis

The project is not wrong; it is *under-implemented relative to its own thesis*, and that gap is now the most productive thing about it.

Steyerl's "mean image" is a claim about **statistical rendering**: images indexed to the probable rather than the real. The current build gives us the *arithmetic* mean, which is the least statistical of all averages — it has no model, no prior, no learned structure. It is Galton's 1870s composite photograph, re-implemented in NumPy.

That is not a flaw to hide. It is **Exhibit A**. The strongest move available in iteration 1 is not to replace the pixel mean but to *put it in a series* with three other averages the machine can now compute, and let the viewer see that "the average" is not one thing:

| Average | Computation | What it claims an average *is* | Historical analogue |
|---|---|---|---|
| **Pixel mean** | `Σ Iᵢ / N` in sRGB | Optical superposition; the average is a *photograph of everything at once* | Galton's composite portraiture (1878–1883) |
| **Medoid** | `argmin_i Σ_j d(eᵢ, e_j)` | The average is a *real member* who stands in for the rest | Truth-to-nature; the "characteristic specimen" |
| **Embedding centroid → image** | `c = Σ eᵢ / N`, then decode/condition on `c` | The average is a *point in perceptual space* that no work occupies; a synthetic exemplar | Mechanical objectivity, now in feature space |
| **Learned likeness (LoRA / inverted token)** | fine-tune weights on the subset | The average is a *disposition to produce*, not a picture | Trained judgment; the atlas-maker's practised eye |

Daston and Galison's three epistemic virtues — **truth-to-nature, mechanical objectivity, trained judgment** — map onto the medoid, the centroid, and the fine-tune with almost embarrassing directness. Making that mapping literal and switchable in the UI turns a reading response into an argument that the interface itself performs. This is the single most important design recommendation in this document.

The second reframing concerns **where the politics live**. The current build lets the user remove works, which stages exclusion as an *individual* act. But the sample was already constrained four times before the user arrived: by what the NGA accessioned, by what it digitised, by what it released as open access, and by `FLAT_CLASSIFICATIONS`. Crawford & Paglen, Denton et al., and Birhane & Prabhu all argue that the consequential choices happen upstream of the model, in the dataset's constitution. The piece should surface those four filters as *first-class, visible, and contestable* — the user's removals are the fifth filter in a chain, not the first.

The third reframing concerns **latency as meaning**. Turkle's "discontent" with simulation is about the seductive completeness of the simulated object. A mean that appears instantly reads as a fact. A mean that takes forty minutes of visible GPU labour reads as a construction. Iteration 1 should keep *both*: an instant embedding-space preview (the mean as arithmetic, cheap, provisional) and a slow, watchable fine-tune (the mean as labour, expensive, committed). The contrast between the two is content, not overhead.

---

## 3. Human–AI co-creation and mixed-initiative creative interfaces

The relevant framing is that *both* parties can initiate. In the current build only the human initiates (GENERATE) and the machine's only channel back is a progress bar. Lin et al.'s design-space study of mixed-initiative co-creativity found that users want richer, bidirectional channels for communicating creative intent, not just a prompt box. Weisz et al.'s CHI 2024 design principles for generative AI applications formalise this into actionable guidance — most relevant here: *design for generative variability* (show multiple candidate means, not one), *design for imperfection*, and *design for mental models* (tell the user what the system actually did).

Amershi et al.'s account of interactive machine learning is the strongest argument for the latency redesign: the value of iML comes from rapid, incremental, user-driven cycles, and it degrades sharply when each cycle costs minutes. A one-hour retrain is not interactive machine learning; it is batch training with a GUI in front of it.

**Design consequences.** (a) Every dataset edit must produce *some* feedback in well under a second — the embedding-centroid mean provides this. (b) The system should propose, not just respond: surface "these 12 works are pulling the mean hardest", "this subset splits into three clusters — average them separately?". (c) Show several means at once (generative variability) rather than a single authoritative output.

## 4. Latent space exploration and navigation UIs

Picbreeder established the canonical interaction: a population of candidates, human selection as the fitness signal, and branching from other people's results — creativity as navigation of a space nobody authored. Recent work (Sun et al., *Browsing the Latent Space*, C&C 2023) applies the same logic to modern generative models with interpolation-based design exploration. A 2026 survey of 51 systems across CHI/VIS/UIST/NeurIPS/ICML/ICLR decomposes generative image interfaces into a **UI space**, a **model object space**, and the **mapping** between them, which is a useful audit lens: in Whose Mean? the model object space is currently just "the training subset", and the mapping from UI to model is a single binary include/exclude per work.

**Design consequences.** Widen the model object space from *{subset}* to *{subset, per-work weight, embedding centroid, interpolation path, conditioning strength}*, and give each a UI handle. Notably, `mean_weight` already exists in `works.json` and is unused — continuous weighting is the cheapest way to turn a binary curation act into a navigable space. Interpolating between two means (e.g. "works before 1850" → "works after 1850") is a far stronger demonstration of the thesis than either endpoint.

## 5. Dataset curation, transparency, provenance and consent

Crawford & Paglen's central claim is that training sets and their taxonomies are *epistemic infrastructures*: they determine what a system can see and what stays invisible. Denton et al. propose genealogy as method — reading a dataset through the norms and values embedded in its construction, not just auditing its contents. Birhane & Prabhu's WACV census of ImageNet demonstrates the audit at scale and foregrounds consent and justice. Offert & Bell add a dimension these three do not cover: **perceptual bias**, the gap between a model's assumed way of seeing and its actual perceptual topology — bias that lives in the architecture, not only in the data.

For AI art specifically, Jiang et al.'s survey of 459 artists (AIES 2024) and Lovato et al.'s CHI 2025 work on consent, credit and compensation give citable empirical grounding for the provenance argument. Whose Mean? is in an unusually comfortable position here — NGA open-access images are released under CC0 and the object metadata is CC0 — and that comfort should be made explicit in the work rather than assumed, because it is precisely the *counter-example* that makes the general problem legible.

**Design consequences.** (a) A permanent "WHOSE?" panel showing the composition of the current selection: date distribution, classification, attribution concentration, credit-line/donor concentration, and the count of works with no attributed artist. (b) A visible, clickable "provenance chain" naming all four upstream filters and their cardinalities (`eligible → accessioned → open-access primary → classification-filtered → sampled(seed 42) → your removals`). The `collect()` function already prints `eligible` and `saved` counts; they should be persisted and displayed. (c) A CC0 badge with the actual licence text — as a claim about this dataset, and implicitly about the models that lack one.

## 6. Bias, representation and statistical averaging in generative models

Luccioni et al.'s *Stable Bias* (NeurIPS 2023) analysed over 96 000 generated images from DALL·E 2 and Stable Diffusion v1.4/v2 and found over-representation of whiteness and masculinity — and, importantly, developed a method for studying bias in *synthetic* outputs, where the subjects have no ground-truth identity. That methodological problem is exactly Whose Mean?'s problem: the mean image depicts nobody, so the question "whose?" cannot be answered by looking at it. It can only be answered by instrumenting the *relation* between the set and the output.

**Design consequences.** Implement leave-one-out and top-contributor attribution in embedding space: for the current centroid `c`, rank works by `⟨eᵢ, c⟩` or by `‖c − c_{\i}‖`, and let the user see "the 20 works most responsible for this mean". This is the operational answer to the title, and it is cheap.

## 7. Explainable/interpretable ML for non-experts; interactive machine teaching

Kulesza et al.'s Explanatory Debugging (IUI 2015) is the model to copy: the system explains each prediction, the user corrects it, and the correction is fed back — yielding a 52 % increase in participants' understanding of the learner. The principle is *explanation and correction on the same surface*. The current build has correction (REMOVE) with no explanation at all; the MODEL I/O tab is a static architecture dump, which is documentation, not explanation.

**Design consequences.** Replace the static MODEL I/O text with a live "why this image?" panel bound to the currently displayed mean: which works contributed most, what the aggregate metadata of those works is, and what changed since the previous mean. Every removal should show a *delta*, not just a new picture.

## 8. Cultural heritage, GLAM collections, and generous interfaces

Whitelaw's argument is that search is "ungenerous": it withholds the collection and demands a query. Generous interfaces instead reveal scale, structure and texture before any query is entered. Dörk et al.'s *information flaneur* gives the complementary user model — curious, creative, critical browsing, with **explorability** as the design principle. Whose Mean? is already partly generous (all 1000 works visible at once, no empty state) and partly not (the alias-based keyword filter is a query-first mechanism that dims 97 % of the collection). Lee et al.'s "Collections as ML Data" checklist (JASIST 2025) is the practical GLAM-side complement for documenting a collection that will be used to train something.

**Design consequences.** Keep the always-visible overview; replace string search with embedding-space semantic search *plus* a faceted metadata browser (date, classification, medium, donor), so that querying enriches the overview instead of replacing it. Adopt the Collections-as-ML-Data checklist as the structure of a `DATASET.md` shipped with the work.

## 9. Critical/reflective design and research-through-design

Sengers et al.'s reflective design asks designers to make unconscious cultural assumptions available for reflection, by the user as well as the designer. Gaver et al.'s *ambiguity as a resource* supplies the specific tactic: deliberate ambiguity of information, context and relationship invites interpretation and deepens engagement. Zimmerman et al. legitimise the artefact itself as the research contribution.

These three together settle two design arguments. First, **do not caption the mean**. Do not tell the viewer which average is correct; present the four side by side and let the disagreement be the content (Gaver). Second, **evaluate for plurality of interpretation, not task success** — the right outcome measure is whether participants leave able to articulate what the collection over- and under-represents, not whether they produced a good picture (Sengers; see the evaluation section of `ITERATION1_PLAN.md`).

---

## 10. Annotated bibliography

### 10.1 Theoretical lineage already cited by the project

**Steyerl, Hito (2023). "Mean Images." *New Left Review* 140/141, March–June 2023, pp. 82–97.**
https://newleftreview.org/issues/ii140/articles/hito-steyerl-mean-images
The source of the project's title and thesis: machine-learning systems produce a new kind of machinic representation indexed to the *probable* rather than the real — an uncanny composite rather than an abstraction. **Informs:** the decision to treat "the mean" as plural and contested rather than as a single output, and the argument in §2 that an arithmetic pixel mean is precisely *not* a statistical rendering and should therefore be staged as the historical baseline against three genuinely statistical alternatives.

**Daston, Lorraine & Galison, Peter (2007). *Objectivity*. New York: Zone Books, 502 pp.**
https://www.zonebooks.org/books/5-objectivity
Charts three successive epistemic virtues in scientific atlas-making: truth-to-nature, mechanical objectivity, and trained judgment. **Informs:** the central UI proposal — a mean-strategy selector whose three non-trivial options (medoid / centroid / fine-tuned likeness) instantiate the three virtues, making the book's argument operable rather than quoted.

**Turkle, Sherry (2009). *Simulation and Its Discontents*. Cambridge, MA: MIT Press. With essays by W. J. Clancey, S. Helmreich, Y. A. Loukissas and N. Myers. xiv + 217 pp.**
https://direct.mit.edu/books/book/3855/Simulation-and-Its-Discontents
On what is lost when practice becomes mediated by simulation, and on simulation's seductive appearance of completeness. **Informs:** the decision to preserve visible computational labour — keeping a slow, watchable fine-tune alongside the instant preview, so the mean is legible as something made rather than something found.

### 10.2 Critical AI, datasets and provenance

**Crawford, Kate & Paglen, Trevor (2021). "Excavating AI: the politics of images in machine learning training sets." *AI & Society* 36, 1105–1116.**
https://link.springer.com/article/10.1007/s00146-021-01162-8 · correction: https://link.springer.com/article/10.1007/s00146-021-01301-1
Training sets and their taxonomies are epistemic infrastructures; the automated interpretation of images is an inherently social and political project. **Informs:** the "WHOSE?" provenance panel and the visible four-stage filter chain — making the taxonomy (`FLAT_CLASSIFICATIONS`) an object of interaction rather than a constant in a source file.

**Denton, Emily; Hanna, Alex; Amironesei, Razvan; Smart, Andrew & Nicole, Hilary (2021). "On the genealogy of machine learning datasets: A critical history of ImageNet." *Big Data & Society* 8(2).**
https://journals.sagepub.com/doi/10.1177/20539517211035955
Proposes genealogy as method: read datasets as informational infrastructure, examining the norms, values and assumptions in their constitution rather than only their contents. **Informs:** the `DATASET.md` deliverable and the decision to display the sampling seed, the eligible/collected/failed counts, and the classification filter in the interface itself.

**Birhane, Abeba & Prabhu, Vinay Uday (2021). "Large image datasets: A pyrrhic win for computer vision?" *WACV 2021*, pp. 1537–1547.**
https://openaccess.thecvf.com/content/WACV2021/html/Birhane_Large_Image_Datasets_A_Pyrrhic_Win_for_Computer_Vision_WACV_2021_paper.html
A quantitative census of ImageNet-ILSVRC-2012 covering consent, NSFW content, and the semantics of class labels. **Informs:** the framing of the NGA CC0 sample as a deliberate counter-example — a set where consent and licensing are unusually clean — and the argument that this should be stated in the work, not assumed.

**Offert, Fabian & Bell, Peter (2021). "Perceptual bias and technical metapictures: critical machine vision as a humanities challenge." *AI & Society* 36, 1133–1144.**
https://link.springer.com/article/10.1007/s00146-020-01058-z
Introduces *perceptual bias*: the gap between a machine vision system's assumed way of seeing and its actual perceptual topology — bias in the architecture, not only the data. **Informs:** the requirement that the interface let users switch the *space* in which the mean is computed (pixel / CLIP / SigLIP / DINO), because "the average" changes with the perceptual topology, and that change is visible and teachable.

**Salvaggio, Eryk (2023). *Flowers Blooming Backward Into Noise*. Animated video essay, c. 20 min; text version at ArtsEverywhere.**
https://www.artseverywhere.ca/flowers-blooming-backward/ · https://www.cyberneticforests.com/news/flowers-blooming-backward-into-noise-2023
Reads AI-generated images as products of their datasets, explicitly tying diffusion outputs to the lineage of composite photography and statistical correlation. **Informs:** the Galton framing in §2 and the "read the image as a dataset artefact" stance of the attribution panel.

**Jiang, Harry H.; Brown, Lauren; Cheng, Jessica; Khan, Mehtab; Gupta, Abhishek; Workman, Deja; Hanna, Alex; Flowers, Johnathan & Gebru, Timnit (2023/2024). "Foregrounding Artist Opinions: A Survey Study on Transparency, Ownership, and Fairness in AI Generative Art." *AAAI/ACM Conference on AI, Ethics, and Society (AIES)*.**
https://ojs.aaai.org/index.php/AIES/article/view/31691
Survey of 459 artists on utility, threat, disclosure of training works, ownership of derivatives, and compensation. **Informs:** the decision to display licence and credit-line information per work and in aggregate, and to make the CC0 status of the sample an explicit, visible claim.

**Lovato, Juniper et al. (2025). "Governance of Generative AI in Creative Work: Consent, Credit, Compensation, and Beyond." *CHI 2025*.**
https://dl.acm.org/doi/10.1145/3706598.3713799 · preprint https://arxiv.org/abs/2501.11457
Empirical HCI treatment of the consent/credit/compensation triad in creative practice. **Informs:** the exhibition-mode wall text and the structure of the provenance panel.

### 10.3 Composite images and the prehistory of the mean

**Galton, Francis (1878). "Composite Portraits." *Journal of the Anthropological Institute of Great Britain and Ireland* 8, 132–144.**
https://galton.org/essays/1870-1879/galton-1878-Composite-Portraits-Extended.pdf · JSTOR: https://www.jstor.org/stable/2841021
The originating technique: multiple-exposure photographic superimposition registered on the eyes, intended to reveal a "type". Galton applied it to convicts, psychiatric patients, tuberculosis patients and Jewish schoolboys in service of eugenic theory; the composites persistently produced idealised faces that undercut his thesis. **Informs:** the historical caption for the pixel-mean mode. The arithmetic mean in Whose Mean? *is* Galton's method in software; naming that is the single highest-value piece of conceptual work available, and it costs nothing to implement.

**Stephens, Elizabeth (2013). "Francis Galton's Composite Portraits: The Productive Failure of a Scientific Experiment." `[UNVERIFIED VENUE]`**
https://www.academia.edu/6746153/
Argues the technique was a productive failure: it failed to substantiate Galton's theories and was little adopted, yet shaped later visual epistemology. **Informs:** the framing of the pixel mean as an instructive failure rather than a deprecated feature. *The journal of publication could not be confirmed in this pass — see §11.*

**Lydon, Jane et al. (2024). "'We are all alike': Composite Portraits, CONVICTS, and the Ethics of Representation." *Australian Historical Studies* 56(1).**
https://www.tandfonline.com/doi/full/10.1080/1031461X.2024.2328095
Contemporary scholarship on the ethics of applying composite portraiture to archival populations who cannot consent. **Informs:** the ethical caution in exhibiting any composite of depicted persons, and the recommendation to offer a "portraits only" subset with explicit framing rather than silently including it.

### 10.4 Human–AI co-creation, creativity support, and interactive ML

**Amershi, Saleema; Cakmak, Maya; Knox, William Bradley & Kulesza, Todd (2014). "Power to the People: The Role of Humans in Interactive Machine Learning." *AI Magazine* 35(4), 105–120.**
https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2513
Canonical statement of interactive ML: value comes from rapid, incremental, user-driven cycles, and systems that ignore the user's tempo fail. **Informs:** the hard latency budget in the plan (< 300 ms for any dataset edit to produce a visible change) and the decision that full retraining must become an optional, explicitly-invoked commitment rather than the only feedback path.

**Kulesza, Todd; Burnett, Margaret; Wong, Weng-Keen & Stumpf, Simone (2015). "Principles of Explanatory Debugging to Personalize Interactive Machine Learning." *IUI 2015*, 126–137.**
https://doi.org/10.1145/2678025.2701399
Two-way explanation: the system explains its predictions, the user corrects them on the same surface; measured a 52 % increase in users' understanding of the learner. **Informs:** replacing the static MODEL I/O tab with a live "why this mean?" attribution panel, and showing a delta after every removal.

**Weisz, Justin D. et al. (2024). "Design Principles for Generative AI Applications." *CHI 2024*.**
https://dl.acm.org/doi/10.1145/3613904.3642466
Six principles for generative AI UX, including designing for generative variability, imperfection, and accurate mental models. **Informs:** showing multiple means simultaneously rather than one authoritative output, and labelling each with the operation that produced it.

**Lin, Zhiyu; Agarwal, Rohan & Riedl, Mark (2023). "Beyond Prompts: Exploring the Design Space of Mixed-Initiative Co-Creativity Systems." *ICCC 2023*; arXiv:2305.07465.**
https://arxiv.org/abs/2305.07465
A design space of human↔AI intent-communication channels, with a 185-participant study of user preferences across configurations. **Informs:** widening the machine's channel back to the user from "progress bar" to "proposed subsets, contributor rankings, and cluster suggestions".

**Cherry, Erin & Latulipe, Celine (2014). "Quantifying the Creativity Support of Digital Tools through the Creativity Support Index." *ACM TOCHI* 21(4), Article 21.**
https://dl.acm.org/doi/10.1145/2617588
The CSI psychometric instrument across six factors (Exploration, Expressiveness, Immersion, Enjoyment, Results Worth Effort, Collaboration). **Informs:** the quantitative half of the evaluation plan — with the explicit caveat that Exploration and Immersion are the load-bearing factors for a reflective artefact and that "Results Worth Effort" will read oddly for a work whose point is that the result is contestable.

**Secretan, Jimmy; Beato, Nicholas; D'Ambrosio, David B.; Rodriguez, Adelein; Campbell, Adam & Stanley, Kenneth O. (2008). "Picbreeder: evolving pictures collaboratively online." *CHI 2008*, 1759–1768.**
https://doi.org/10.1145/1357054.1357328 · extended: *Evolutionary Computation* 19(3), 2011, https://dl.acm.org/doi/10.1162/EVCO_a_00030
Collaborative interactive evolution: human selection as fitness, with branching from others' results. **Informs:** the run-history and branching model — each mean should be a node the user can fork from, not a disposable render.

### 10.5 Latent space interfaces and generative image UIs

**Liu, Vivian & Chilton, Lydia B. (2022). "Design Guidelines for Prompt Engineering Text-to-Image Generative Models." *CHI 2022*, Article 384.**
https://doi.org/10.1145/3491102.3501825
5000+ generations, 51 subjects, 51 styles; establishes that separating subject and style keywords yields far more predictable output than discursive prose. **Informs:** the prompt affordance in the new generative backend — if a text channel is added, it should be a structured subject/style pair, not a free-text box, and it must be visibly distinct from the dataset-filter box (which the current build conflates).

**Liu, Vivian; Qiao, Han & Chilton, Lydia B. (2022). "Opal: Multimodal Image Generation for News Illustration." *UIST 2022*.**
https://doi.org/10.1145/3526113.3545621
A structured pipeline that derives visual concepts from source material rather than asking the user to invent prompts. **Informs:** deriving conditioning from the *collection* (centroid, exemplars, metadata-derived text) rather than from typed prompts — which is also the conceptually correct choice here.

**Sun, Yuqian et al. (2023). "Browsing the Latent Space: A New Approach to Interactive Design Exploration for Volumetric Generative Systems." *Creativity & Cognition 2023*.**
https://dl.acm.org/doi/10.1145/3591196.3596815
Latent-space visualisation and interpolation as a design-exploration interface. **Informs:** the interpolation feature — a slider between two user-defined subsets' centroids, which demonstrates the thesis better than any single mean.

**"A Design Space of Visual Interfaces for Generative Image Models" (2026). arXiv:2609.18065.**
https://arxiv.org/html/2609.18065
Survey of 51 systems from CHI/VIS/UIST/NeurIPS/ICML/ICLR; decomposes interfaces into UI space, model object space, and mapping space. **Informs:** the audit lens in §4 and the argument for widening the model object space beyond a binary include/exclude. *Venue not confirmed — appears to be a preprint; cite as arXiv unless a venue is verified.*

### 10.6 Cultural heritage interfaces

**Whitelaw, Mitchell (2015). "Generous Interfaces for Digital Cultural Collections." *Digital Humanities Quarterly* 9(1).**
https://dhq.digitalhumanities.org/vol/9/1/000205/000205.html
Search is ungenerous: it withholds the collection and demands a query. Argues for rich, browsable interfaces that reveal scale and complexity. **Informs:** retaining the always-visible 1000-work overview as the home state and refusing a query-first redesign; replacing the dimming behaviour of keyword search with additive highlighting.

**Dörk, Marian; Carpendale, Sheelagh & Williamson, Carey (2011). "The Information Flaneur: A Fresh Look at Information Seeking." *CHI 2011*, 1215–1224.**
https://doi.org/10.1145/1978942.1979124
Proposes the flaneur as a user model — curious, creative, critical — with *explorability* as the guiding design principle. **Informs:** the "wander" affordances: hover-to-enlarge (already present), plus proposed cluster tours and serendipitous neighbour jumps in embedding space.

**Lee, Benjamin Charles Germain et al. (2025). "The 'Collections as ML Data' checklist for machine learning and cultural heritage." *JASIST*.**
https://asistdl.onlinelibrary.wiley.com/doi/10.1002/asi.24765
A practical checklist for GLAM collections used as machine-learning data. **Informs:** the structure of the proposed `docs/DATASET.md`.

**National Gallery of Art (n.d.). *Free Images and Open Access* / *Open Data Program*.**
https://www.nga.gov/artworks/free-images-and-open-access · https://github.com/NationalGalleryOfArt/opendata
Images the Gallery believes to be in the public domain are released under CC0, as is the object metadata for 130 000+ artworks and artists. **Informs:** the licence badge and the CC0 counter-example argument in §5.

### 10.7 Reflective and critical design

**Sengers, Phoebe; Boehner, Kirsten; David, Shay & Kaye, Joseph 'Jofish' (2005). "Reflective Design." *Critical Computing 2005 (CC'05, Aarhus)*, 49–58.**
https://doi.org/10.1145/1094562.1094569
Combines analysis of technologies' unconscious cultural assumptions with the building and evaluation of devices that open alternatives to reflection. **Informs:** the evaluation criterion — success is participants' articulated reflection on the collection's composition, not task completion.

**Gaver, William W.; Beaver, Jacob & Benford, Steve (2003). "Ambiguity as a Resource for Design." *CHI 2003*, 233–240.**
https://doi.org/10.1145/642611.642653
Three classes of ambiguity (of information, of context, of relationship) as a deliberate design resource encouraging personal engagement. **Informs:** the decision not to caption or rank the four means, and to let their mutual disagreement stand as the work's argument.

**Zimmerman, John; Forlizzi, Jodi & Evenson, Shelley (2007). "Research through design as a method for interaction design research in HCI." *CHI 2007*, 493–502.**
https://doi.org/10.1145/1240624.1240704
Legitimises the designed artefact as the research contribution for under-constrained problems. **Informs:** the venue strategy (DIS Pictorials / C&C Artworks) and the decision to write the contribution as an artefact-plus-annotated-portfolio rather than as a controlled experiment.

### 10.8 Generative model foundations cited in the model landscape

Full technical treatment is in `ITERATION1_MODEL_LANDSCAPE.md`; these are the citations that also carry conceptual weight.

- **Ramesh, Aditya; Dhariwal, Prafulla; Nichol, Alex; Chu, Casey & Chen, Mark (2022). "Hierarchical Text-Conditional Image Generation with CLIP Latents." arXiv:2204.06125.** https://arxiv.org/abs/2204.06125 — The unCLIP architecture: a prior producing a CLIP *image embedding*, then a decoder conditioned on it. Establishes that a point in CLIP space can be decoded to an image, which is the technical basis for "the centroid of a collection, made visible". **Informs:** the embedding-centroid mean.
- **Ye, Hu; Zhang, Jun; Liu, Sibo; Han, Xiao & Yang, Wei (2023). "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models." arXiv:2308.06721.** https://ip-adapter.github.io/ — Decoupled cross-attention lets a frozen diffusion model be conditioned on an image embedding. **Informs:** the primary generative path — condition on the *centroid embedding* of the selection, which makes "the mean" a first-class conditioning signal rather than a preprocessing artefact.
- **Gal, Rinon et al. (2023). "An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion." *ICLR 2023*.** https://arxiv.org/abs/2208.01618 — Learns a new pseudo-word in the text-encoder embedding space from a handful of images. **Informs:** the "collection as a single token" mean, which is conceptually the sharpest of the four: the collection becomes a *name* that can be composed into sentences.
- **Ruiz, Nataniel et al. (2023). "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation." *CVPR 2023*, 22500–22510.** https://openaccess.thecvf.com/content/CVPR2023/html/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.html — Binds a unique identifier to a subject by fine-tuning, with a class-prior-preservation loss. **Informs:** the "learned likeness" mean and its risk profile (overfitting / language drift as the aesthetic signature of a small collection).
- **Zhang, Lvmin; Rao, Anyi & Agrawala, Maneesh (2023). "Adding Conditional Control to Text-to-Image Diffusion Models." *ICCV 2023*, 3836–3847.** https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html — Zero-convolution side networks add spatial control; robust with datasets from < 50 k to > 1 M. **Informs:** the honest reimplementation of the "structure anchor" as a real structural constraint (edge/depth map of the medoid) rather than a luminance blend.
- **Esser, Patrick et al. (2024). "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." *ICML 2024*, 12606–12633.** https://proceedings.mlr.press/v235/esser24a.html — The SD3 formulation; rectified flow plus a dual-stream MMDiT. **Informs:** the flow-matching column of the model comparison.
- **Podell, Dustin et al. (2024). "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis." *ICLR 2024*.** https://iclr.cc/virtual/2024/poster/18250 — **Informs:** the recommended primary backbone.
- **Tschannen, Michael et al. (2025). "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features." arXiv:2502.14786.** https://arxiv.org/abs/2502.14786 — Apache-2.0 open-weight encoders at ViT-B/L/So400m/g, 109 languages. **Informs:** the recommended embedding space for the semantic layout and the centroid.
- **Siméoni, Oriane et al. (2025). "DINOv3." arXiv:2508.10104.** https://arxiv.org/abs/2508.10104 — Self-supervised, label-free dense features; 1.7 B curated images, up to 7 B params. Released under a Meta commercial licence requiring registration, **not** Apache-2.0 (unlike DINOv2). **Informs:** the "second perceptual topology" offered as a switchable alternative to SigLIP, and a licensing caution.
- **Luccioni, Sasha; Akiki, Christopher; Mitchell, Margaret & Jernite, Yacine (2023). "Stable Bias: Evaluating Societal Representations in Diffusion Models." *NeurIPS 2023 Datasets & Benchmarks*.** https://neurips.cc/virtual/2023/poster/73455 · arXiv:2303.11408 — 96 000+ images from DALL·E 2 and SD v1.4/v2; over-representation of whiteness and masculinity; method for studying bias in synthetic subjects. **Informs:** the attribution panel in §6 and the argument that the mean must be instrumented, not merely displayed.

---

## 11. What I could not verify

Stated plainly, because the alternative is fabrication.

1. **Stephens (2013) journal of publication.** The article exists and is widely circulated on Academia.edu and ResearchGate with that exact title and author, but the hosting venue could not be confirmed from accessible sources in this pass (ResearchGate is blocked from this environment). Marked `[UNVERIFIED VENUE]`. Either confirm it before submission or cite the Australian Historical Studies (2024) article instead, whose full citation is confirmed.
2. **Lydon et al. (2024) full author list.** The article, journal, volume and DOI are confirmed; the complete author list was not read. Verify before citing.
3. **Weisz et al. (2024) full author list.** Title, venue and DOI confirmed; the author list beyond the first author was not read.
4. **Lovato et al. (2025) full author list.** Title, venue, DOI and arXiv ID confirmed; full author list not read.
5. **Jiang et al. (AIES) year.** The paper appears in AIES proceedings with both 2023 and 2024 associated in different indexes (conference year vs. proceedings year). Verify before citing.
6. **"A Design Space of Visual Interfaces for Generative Image Models" (arXiv:2609.18065)** — authors and any peer-reviewed venue were not confirmed; cite as a preprint.
7. **Current build's actual epoch wall-clock time.** Still unmeasured. The parallel engineering audit established the *causes* (batch 1, 20 000 redundant JPEG decodes per run, four device syncs per sample, 134 MB of checkpoints per epoch) and reproduced the UI-side costs on CPU, but no GPU was available to either pass. Benchmark one epoch on the real target GPU in Phase 0; the entire latency argument deserves a measured number rather than an inference from architecture.
8. **The NGA `constituents.csv` schema** (needed for artist nationality and life dates in the provenance panel) was not inspected. Confirm the available fields before designing that panel.
