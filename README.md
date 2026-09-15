# Whose Mean?

**Peiyan Zou · ADV 9672 · W3 Reading Response**

`Whose Mean?` is an interactive machine-learning tool that turns 1,000 open-access works from the National Gallery of Art into a navigable dataset and a continuously negotiated “mean image.” Users can search, inspect, exclude, and retrain the collection while watching pix2pix transform individual works and collective averages across epochs. Following Hito Steyerl, the project treats the mean image as a statistical likeness that carries the values and omissions of its source data. Daston's account of Harvard's Glass Flowers raises a related question: does a model faithfully reproduce an individual object, or construct a generalized type from many examples? Following Turkle, the project also questions the apparent completeness of simulation: every generated mean is shaped by what the dataset includes and excludes.

The English-only interface shows every sampled work as a camera-facing thumbnail in a zoomable XYZ feature space. It exposes background training progress, per-epoch mean outputs, and the generated output for every trained work.

For the independent Python application, double-click `OPEN_DESKTOP_STUDIO.cmd`. It reads the local collection and calls the PyTorch training module directly; it does not require the browser, HTTP, or the local web server. The optional browser version remains available at <http://127.0.0.1:8765/> through `OPEN_WHOSE_MEAN.cmd`.

Create the local environment before the first run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Museum images, generated outputs, and model weights are intentionally excluded from Git because they are large and reproducible with the commands below.

## Data

The 1,000-work sample was drawn with seed 42 from 40,219 eligible NGA records that have an open-access primary image and an exact classification of Painting, Drawing, Print, or Photograph. The local sample contains:

- 692 Prints
- 176 Drawings
- 69 Paintings
- 63 Photographs

NGA publishes the metadata as bulk CSV and the images through public IIIF URLs, so this workflow does not require an API key. The sample is a reproducible subset of eligible records, not the museum's entire collection.

To rebuild it:

```powershell
.\.venv\Scripts\python.exe nga_pipeline.py collect --limit 1000 --delay 0.02
.\.venv\Scripts\python.exe nga_pipeline.py visualize-data --size 96
```

## Live 512px training

The live model uses a full 512 × 512 grayscale rendering of each work as the paired input and the corresponding 512 × 512 RGB work as the target. This gives pix2pix a meaningful conditional task: reconstruct structure while learning collection color. Every non-excluded artwork is visited exactly once in every epoch. Training uses a six-scale U-Net generator, PatchGAN discriminator, adversarial loss, L1 reconstruction loss, and edge loss.

The desktop `CONCEPT KEYWORDS` field searches the title, artist, classification, and medium of the local museum records. It accepts English keywords, and multiple keywords use ANY matching. It filters the training set; it does not claim that the unmodified pix2pix network understands free text. `GENERATE` trains on every matching work.

Typing in the keyword field immediately highlights matching thumbnails in the XYZ space with a larger image and white outline. Nonmatching thumbnails are composited over black at 30% opacity. A minimum spanning tree links the matching works by their XYZ feature distance, so each line represents the shortest available relationship in warmth, luminance, and edge density. This preview does not start computation; `GENERATE` applies the same match set to a new training run.

Mean outputs use a visible `STRUCTURE ANCHOR` control. At the default 0.65 setting, the inference input combines 35% of the exact arithmetic mean with 65% of the work nearest the selected set's XYZ feature centroid. The pure arithmetic mean, representative artwork, mixed input, and generated output are all saved separately in the run folder. This retains an auditable statistical mean while giving pix2pix enough spatial structure to produce recognizable content.

The right panel starts and stops training in a Python background thread. It displays the current artwork count, progress, measured train L1, saved mean output for every completed epoch, and a five-column panel containing the generated result for every trained artwork. Training and generation run locally on CUDA; the verified machine reports `cuda` on an RTX 5080.

The verified one-epoch integration run trained all 1,000 works in 34.63 seconds and reached train L1 `0.08415`. It then generated all 1,000 result images. One epoch validates the full pipeline; the interface defaults to 20 epochs for a longer artwork run.

## Interaction

- Hold the right mouse button and drag anywhere in the collection space to orbit the XYZ model. The left button remains dedicated to inspecting or removing works.
- Scroll over the collection canvas to scale the entire coordinate space.
- Hover a thumbnail to enlarge it; thumbnails always face the camera.
- Select `REMOVE`, click any works, and choose `GENERATE`. The next model is initialized from scratch and every remaining image participates in each epoch.
- Scroll over the mean output, or use the `OUTPUT` buttons, to enlarge it.
- Click an epoch tile to inspect a saved training output.
- Scroll through `ALL GENERATED RESULTS` to inspect every pix2pix result from the run.
- Click any thumbnail in `ALL RESULTS` to open its 512px original and 512px generated image side by side. The comparison window supports synchronized wheel and button zoom from 40% to 400%.
- Click `OPEN LARGE` or the main mean image to open the current epoch at up to 600% zoom.
- Click a collection thumbnail in `INSPECT` mode to open its 512px original with museum metadata.
- Click `EXPORT RESULTS` to save the final mean, epoch sequence, generated works, inputs, and run metadata to a chosen folder.

The standalone desktop application provides these controls in Tkinter. Closing the web page or stopping `lab.py serve` does not stop desktop training. Keep the desktop window open while its own training run is active.

The axes use measured image features: X is warmth (red minus blue), Y is luminance, and Z is edge density. Their positions describe visual statistics; they are not neural-network neurons or causal links. When a keyword is active, a minimum spanning tree connects matching works by distance in this feature space; no artwork connections are drawn when the keyword field is empty.

The XYZ origin stays centered in the collection canvas. The current pix2pix mean is displayed in the right training sidebar so it no longer pushes the collection away from the visual center.

## Meaning of the mean

The mean output is the generator's response to the grayscale arithmetic mean of the currently included 512px works. Because unrelated subjects and compositions cancel one another, the mean can remain diffuse even when individual generated results are recognizable. More epochs improve the learned mapping but cannot turn a diverse pixel average into one historically representative scene. This tension is part of the work's relation to Hito Steyerl's “mean image”: the center exposes the distribution and omissions of the archive rather than claiming to be a neutral summary of art.

## Run from PowerShell

```powershell
.\.venv\Scripts\python.exe lab.py serve --port 8765
```

Training files are written to `outputs/nga/live/run_*`. `current.json` drives the progress interface, `frames/` stores mean outputs by epoch, and `generated_atlas.jpg` contains every per-work output. Older LoRA experiments remain as offline comparison files but are not loaded or shown by this interface.

References: [NGA Open Data](https://github.com/NationalGalleryOfArt/opendata), [pix2pix](https://phillipi.github.io/pix2pix/), and [Hito Steyerl, “Mean Images”](https://newleftreview.org/issues/ii140/articles/hito-steyerl-mean-images).
