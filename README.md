# Whose Mean?

**Peiyan Zou · ADV 9672 · W3 Reading Response**

Whose Mean? is an interactive machine-learning artwork that turns 1,000 open-access works from the National Gallery of Art into a navigable dataset and a continuously negotiated “mean image.” Users can search, inspect, exclude, and retrain the collection while watching pix2pix transform individual works and collective averages across epochs.

Following Hito Steyerl, the project treats the mean image as a statistical likeness that carries the values and omissions of its source data. Daston's account of Harvard's Glass Flowers asks whether a model reproduces an individual object or constructs a generalized type from many examples. Following Turkle, the project also questions the apparent completeness of simulation: every generated mean is shaped by what the dataset includes and excludes.

## Features

- Standalone English-language Python interface built with Tkinter
- 1,000 camera-facing artwork thumbnails in a rotatable and scalable XYZ space
- Measured axes for warmth, luminance, and edge density
- Metadata keyword search with highlighted matches and feature-space connections
- Local 512 × 512 pix2pix training with live progress and epoch outputs
- Inspect and remove modes for changing the training set
- Original/generated comparison windows with synchronized zoom
- Large mean-image viewer and complete result export

## Project structure

```text
WhoseMean/
├── src/whose_mean/
│   ├── app.py          # Tkinter interface and interaction
│   ├── data.py         # NGA download, preprocessing, and XYZ features
│   ├── model.py        # 512 px pix2pix generator and discriminator
│   └── training.py     # Background training and result generation
├── OPEN_DESKTOP_STUDIO.cmd
├── pyproject.toml
├── requirements.txt
└── README.md
```

Museum images, generated outputs, model weights, and the virtual environment are intentionally excluded from Git. They remain local and can be reconstructed from the NGA open-access source.

## Setup

Python 3.11 or newer, a CUDA-capable NVIDIA GPU, and Tkinter are required. In PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Download the reproducible NGA sample and calculate its interactive coordinates:

```powershell
.\.venv\Scripts\whose-mean-data.exe collect --limit 1000 --delay 0.02
.\.venv\Scripts\whose-mean-data.exe visualize --size 96
```

The sample uses seed 42 and selects accessioned, open-access primary images classified as Painting, Drawing, Print, or Photograph. It is a reproducible subset of eligible records, not the museum's complete collection. No API key is required.

## Run

Double-click:

```text
OPEN_DESKTOP_STUDIO.cmd
```

Or run the installed command:

```powershell
.\.venv\Scripts\whose-mean.exe
```

Training happens locally on the GPU. Closing a browser has no effect on the desktop application.

## Interaction

- Right-drag the collection to orbit the XYZ space.
- Scroll over the collection to scale it.
- Hover an artwork to enlarge it above the other thumbnails.
- Click an artwork in `INSPECT` mode to open its 512 px image and museum metadata.
- Enter one or more English keywords to highlight metadata matches. Unmatched works remain visible at 30% opacity.
- Matching works are connected by a minimum spanning tree based on their XYZ feature distance. No artwork connections appear when the keyword field is empty.
- Select `REMOVE`, click works, and press `GENERATE` to train again without them.
- Adjust `STRUCTURE ANCHOR` to mix the arithmetic pixel mean with the work nearest the selected set's feature centroid.
- Open epoch tiles, individual results, and the main mean image to inspect them at larger scales.
- Click `EXPORT RESULTS` to save the final mean, all epochs, all per-work outputs, inputs, and run metadata.

## Output

Each run is stored under `outputs/nga/live/run_*` and contains:

- `frames/` with one mean output per epoch
- `generated/` with one 512 × 512 result per included artwork
- `mean_target.png`, `structured_input.png`, and `representative.png`
- `generated_atlas.jpg` and `records.json`
- local generator checkpoints used by the running application

The export command also creates `pix2pix_mean.png` and `export_info.json` in the chosen destination.

## References

- [National Gallery of Art Open Data](https://github.com/NationalGalleryOfArt/opendata)
- [pix2pix](https://phillipi.github.io/pix2pix/)
- [Hito Steyerl, “Mean Images”](https://newleftreview.org/issues/ii140/articles/hito-steyerl-mean-images)
