"""Download and prepare the National Gallery of Art collection sample."""

from __future__ import annotations

import argparse
import csv
import functools
import io
import json
import random
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "nga"
SOURCE = DATA / "source"
IMAGES = DATA / "images"
OUT = PROJECT_ROOT / "outputs" / "nga"

OBJECTS_URL = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/objects.csv"
PUBLISHED_URL = "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/published_images.csv"
FLAT_CLASSIFICATIONS = {"painting", "drawing", "print", "photograph"}


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"Using existing {path.name} ({path.stat().st_size / 1_000_000:.1f} MB)", flush=True)
        return
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "WhoseMean/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, 1024 * 1024)
    temporary.replace(path)
    print(f"Downloaded {path.name} ({path.stat().st_size / 1_000_000:.1f} MB)", flush=True)


def download_metadata() -> None:
    download(OBJECTS_URL, SOURCE / "objects.csv")
    download(PUBLISHED_URL, SOURCE / "published_images.csv")


def candidates() -> list[dict]:
    objects_path = SOURCE / "objects.csv"
    images_path = SOURCE / "published_images.csv"
    if not objects_path.exists() or not images_path.exists():
        raise SystemExit("Run the collect command first.")

    selected = {}
    with objects_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            classification = row.get("classification", "").strip().casefold()
            if row.get("accessioned") == "1" and row.get("isvirtual") != "1" and classification in FLAT_CLASSIFICATIONS:
                selected[row["objectid"]] = row

    primary = {}
    with images_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            object_id = row.get("depictstmsobjectid")
            if object_id in selected and row.get("openaccess") == "1" and row.get("viewtype") == "primary":
                previous = primary.get(object_id)
                if previous is None or int(row.get("sequence") or 9999) < int(previous.get("sequence") or 9999):
                    primary[object_id] = row

    records = []
    for object_id, image in primary.items():
        artwork = selected[object_id]
        records.append(
            {
                "objectid": int(object_id),
                "title": artwork.get("title"),
                "displaydate": artwork.get("displaydate"),
                "attribution": artwork.get("attribution"),
                "classification": artwork.get("classification"),
                "subclassification": artwork.get("subclassification"),
                "medium": artwork.get("medium"),
                "creditline": artwork.get("creditline"),
                "image_uuid": image.get("uuid"),
                "iiifurl": image.get("iiifurl"),
                "width": int(image.get("width") or 0),
                "height": int(image.get("height") or 0),
                "assistivetext": image.get("assistivetext"),
                "source": f"https://www.nga.gov/artworks/{object_id}",
                "openaccess": True,
            }
        )
    return records


def collect(limit: int = 1000, delay: float = 0.02) -> None:
    download_metadata()
    pool = candidates()
    random.Random(42).shuffle(pool)
    chosen = pool[:limit]
    IMAGES.mkdir(parents=True, exist_ok=True)
    saved, failures = [], []

    for index, record in enumerate(chosen, 1):
        target = IMAGES / f"{record['objectid']}.jpg"
        try:
            if not target.exists():
                url = record["iiifurl"].rstrip("/") + "/full/!768,768/0/default.jpg"
                request = urllib.request.Request(url, headers={"User-Agent": "WhoseMean/1.0"})
                with urllib.request.urlopen(request, timeout=90) as response, target.open("wb") as output:
                    shutil.copyfileobj(response, output)
                with Image.open(target) as check:
                    check.verify()
                time.sleep(delay)
            record["localfile"] = target.name
            record["collected_at"] = datetime.now(timezone.utc).isoformat()
            saved.append(record)
            encoding = sys.stdout.encoding or "utf-8"
            title = str(record.get("title") or "Untitled").encode(encoding, errors="replace").decode(encoding)
            print(f"{index:04d}/{len(chosen)}  {record['classification']}  {title}", flush=True)
        except Exception as error:
            target.unlink(missing_ok=True)
            failures.append(
                {"objectid": record["objectid"], "reason": type(error).__name__, "message": str(error)[:300]}
            )

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "records.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in saved) + "\n", encoding="utf-8"
    )
    (DATA / "failures.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in failures) + ("\n" if failures else ""),
        encoding="utf-8",
    )
    print(json.dumps({"eligible": len(pool), "requested": len(chosen), "saved": len(saved), "failed": len(failures), "seed": 42}, indent=2))
    if len(saved) < 8:
        raise SystemExit("Too few usable images were collected.")


def load_records() -> list[dict]:
    path = DATA / "records.jsonl"
    if not path.exists():
        raise SystemExit("No NGA records found. Run the collect command first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def square(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (238, 236, 230))
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
        return np.asarray(canvas).copy()


def prepare_interactive(size: int = 96) -> None:
    records = load_records()
    pixels = np.stack([square(IMAGES / record["localfile"], size) for record in records])
    raw = []
    for image in pixels.astype(np.float32) / 255:
        gray = image.mean(2)
        raw.append(
            [
                float((image[:, :, 0] - image[:, :, 2]).mean()),
                float(gray.mean()),
                float((np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()) / 2),
                float((image.max(2) - image.min(2)).mean()),
            ]
        )
    raw_array = np.asarray(raw)
    normalized = np.clip((raw_array - raw_array.mean(0)) / (raw_array.std(0) + 1e-8), -2.5, 2.5) / 2.5
    works = []
    for record, values, coordinates in zip(records, raw_array, normalized):
        works.append(
            {
                "id": record["objectid"],
                "title": record.get("title"),
                "artist": record.get("attribution"),
                "date": record.get("displaydate"),
                "classification": record.get("classification"),
                "medium": record.get("medium"),
                "source": record.get("source"),
                "x": float(coordinates[0]),
                "y": float(coordinates[1]),
                "z": float(coordinates[2]),
                "warmth": float(values[0]),
                "light": float(values[1]),
                "edge": float(values[2]),
                "saturation": float(values[3]),
                "mean_weight": 1 / len(records),
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "works.json").write_text(
        json.dumps(
            {"count": len(works), "axes": {"x": "warmth (red minus blue)", "y": "mean luminance", "z": "edge density"}, "works": works},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"works": len(works), "output": str(OUT / "works.json")}))


@functools.lru_cache(maxsize=1)
def thumbnail_atlas() -> bytes:
    document = json.loads((OUT / "works.json").read_text(encoding="utf-8"))
    works = document["works"]
    records = {int(record["objectid"]): record for record in load_records()}
    tile, columns = 64, 20
    rows = (len(works) + columns - 1) // columns
    atlas = Image.new("RGB", (columns * tile, rows * tile), (238, 236, 230))
    for index, work in enumerate(works):
        record = records.get(int(work["id"]))
        if record:
            atlas.paste(Image.fromarray(square(IMAGES / record["localfile"], tile)), ((index % columns) * tile, (index // columns) * tile))
    buffer = io.BytesIO()
    atlas.save(buffer, format="JPEG", quality=82, optimize=True)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Whose Mean NGA dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect", help="Download a reproducible open-access sample.")
    collect_parser.add_argument("--limit", type=int, default=1000)
    collect_parser.add_argument("--delay", type=float, default=0.02)
    visualize_parser = subparsers.add_parser("visualize", help="Calculate XYZ coordinates for the interface.")
    visualize_parser.add_argument("--size", type=int, default=96)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.limit, args.delay)
    else:
        prepare_interactive(args.size)


if __name__ == "__main__":
    main()
