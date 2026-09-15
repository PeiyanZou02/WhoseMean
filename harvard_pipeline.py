"""Harvard Art Museums pilot: collect permitted images, compute means, train a compact cGAN."""
import argparse
import json
import os
import random
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

from lab import Generator, Discriminator

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'harvard'
IMAGES = DATA / 'images'
OUT = ROOT / 'outputs' / 'harvard'
API = 'https://api.harvardartmuseums.org/object'
DEFAULT_CLASSES = 'Paintings|Prints|Drawings|Photographs'
FIELDS = ','.join(['objectid', 'title', 'people', 'dated', 'classification', 'culture',
                   'primaryimageurl', 'imagepermissionlevel', 'url', 'copyright'])


def api_key():
    key = os.environ.get('HARVARD_API_KEY', '').strip()
    if not key:
        raise SystemExit('HARVARD_API_KEY is missing. Request one at https://www.harvardartmuseums.org/collections/api')
    return key


def request_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'WhoseMean research prototype/0.1'})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def collect(limit, classifications, delay):
    if limit is not None and limit < 1:
        raise SystemExit('--limit must be positive')
    IMAGES.mkdir(parents=True, exist_ok=True)
    records_path = DATA / 'records.jsonl'
    failures_path = DATA / 'failures.jsonl'
    key = api_key()
    saved, page, failures, seen = [], 1, [], set()
    target_count = limit if limit is not None else float('inf')
    while len(saved) < target_count:
        params = {'apikey': key, 'classification': classifications, 'hasimage': 1,
                  'size': 100, 'page': page,
                  'sort': 'random:42', 'fields': FIELDS}
        payload = request_json(API + '?' + urllib.parse.urlencode(params))
        records = payload.get('records', [])
        if not records:
            break
        for record in records:
            # Level 0 is the museum's unrestricted display tier. Missing values are excluded.
            if record.get('imagepermissionlevel') != 0 or not record.get('primaryimageurl'):
                failures.append({'objectid': record.get('objectid'), 'reason': 'permission_or_image_missing'})
                continue
            object_id = int(record['objectid'])
            if object_id in seen:
                continue
            seen.add(object_id)
            target = IMAGES / f'{object_id}.jpg'
            try:
                if not target.exists():
                    req = urllib.request.Request(record['primaryimageurl'], headers={'User-Agent': 'WhoseMean research prototype/0.1'})
                    with urllib.request.urlopen(req, timeout=60) as response:
                        raw = response.read()
                    from io import BytesIO
                    with Image.open(BytesIO(raw)) as source:
                        image = ImageOps.exif_transpose(source).convert('RGB')
                        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
                        image.save(target, 'JPEG', quality=90, optimize=True)
                record['localfile'] = target.name
                record['collected_at'] = datetime.now(timezone.utc).isoformat()
                saved.append(record)
                total_label = str(limit) if limit is not None else 'ALL'
                print(f"{len(saved):04d}/{total_label}  {record.get('classification')}  {record.get('title')}", flush=True)
            except Exception as error:
                failures.append({'objectid': object_id, 'reason': type(error).__name__, 'message': str(error)[:300]})
            if len(saved) >= target_count:
                break
        if page >= payload.get('info', {}).get('pages', page):
            break
        page += 1
        time.sleep(delay)
    records_path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in saved) + '\n', encoding='utf-8')
    failures_path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in failures) + ('\n' if failures else ''), encoding='utf-8')
    print(json.dumps({'saved': len(saved), 'failed_or_excluded': len(failures), 'records': str(records_path)}, indent=2))
    if len(saved) < 8:
        raise SystemExit('Too few usable images were collected.')


def load_records():
    path = DATA / 'records.jsonl'
    if not path.exists():
        raise SystemExit('No Harvard records found. Run collect first.')
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def square_image(path, size):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        image = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
        # Preserve the whole object. The pale field is explicit and becomes part of the mean.
        canvas = Image.new('RGB', (size, size), (238, 236, 230))
        canvas.paste(image, ((size-image.width)//2, (size-image.height)//2))
        return np.asarray(canvas).copy()


def build_array(size=64):
    records = load_records()
    arrays, used = [], []
    for record in records:
        path = IMAGES / record['localfile']
        if path.exists():
            try:
                arrays.append(square_image(path, size))
                used.append(record)
            except Exception:
                pass
    if len(arrays) < 8:
        raise SystemExit('Too few readable Harvard images.')
    return np.stack(arrays), used


def means(size=256):
    OUT.mkdir(parents=True, exist_ok=True)
    array, records = build_array(size)
    mean = array.astype(np.float32).mean(0).round().astype(np.uint8)
    median = np.median(array, axis=0).round().astype(np.uint8)
    Image.fromarray(mean).save(OUT / 'pixel_mean.png')
    Image.fromarray(median).save(OUT / 'pixel_median.png')
    counts = {}
    for record in records:
        name = record.get('classification') or 'Unknown'
        counts[name] = counts.get(name, 0) + 1
    report = {'dataset': 'Harvard Art Museums API records', 'artworks': len(records),
              'classifications': counts, 'mean_definition': 'per-pixel arithmetic mean after contain-fit on a 238/236/230 field',
              'median_definition': 'per-pixel channel median with identical preprocessing',
              'sample_seed': 42, 'resolution': size, 'generated_at': datetime.now(timezone.utc).isoformat(),
              'status': 'statistical_means_ready', 'model_trained': False}
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def train(epochs=30, size=64):
    if epochs < 1:
        raise SystemExit('--epochs must be positive')
    OUT.mkdir(parents=True, exist_ok=True)
    np_images, records = build_array(size)
    target = torch.from_numpy(np_images).permute(0, 3, 1, 2).float() / 127.5 - 1
    source = F.interpolate(F.interpolate(target, (8, 8), mode='area'), (size, size), mode='bilinear', align_corners=False)
    n = len(target)
    generator_seed = torch.Generator().manual_seed(42)
    order = torch.randperm(n, generator=generator_seed)
    validation_count = max(1, round(n * .1))
    validation, training = order[:validation_count], order[validation_count:]
    torch.manual_seed(42)
    g, d = Generator(), Discriminator()
    og = torch.optim.Adam(g.parameters(), lr=.0003, betas=(.5, .999))
    od = torch.optim.Adam(d.parameters(), lr=.0003, betas=(.5, .999))
    bce = nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, epochs + 1):
        losses = []
        shuffled = training[torch.randperm(len(training))]
        for ids in shuffled.split(8):
            a, b = source[ids], target[ids]
            fake = g(a)
            od.zero_grad()
            real_score, fake_score = d(a, b), d(a, fake.detach())
            ld = (bce(real_score, torch.ones_like(real_score)) + bce(fake_score, torch.zeros_like(fake_score))) / 2
            ld.backward(); od.step()
            for parameter in d.parameters(): parameter.requires_grad_(False)
            og.zero_grad()
            score = d(a, fake)
            lg = bce(score, torch.ones_like(score)) + 50 * F.l1_loss(fake, b)
            lg.backward(); og.step()
            for parameter in d.parameters(): parameter.requires_grad_(True)
            losses.append((lg.item(), ld.item()))
        with torch.inference_mode():
            val_l1 = F.l1_loss(g(source[validation]), target[validation]).item()
        row = {'epoch': epoch, 'val_l1': val_l1, 'g': float(np.mean(losses, axis=0)[0]), 'd': float(np.mean(losses, axis=0)[1])}
        history.append(row)
        if epoch % 5 == 0 or epoch == epochs:
            print(json.dumps(row), flush=True)
    torch.save(g.state_dict(), OUT / 'model.pt')
    (OUT / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    with torch.inference_mode():
        mean_source = source.mean(0, keepdim=True)
        generated = g(mean_source)
        direct = ((mean_source[0].permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
        result = ((generated[0].permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
    Image.fromarray(direct).resize((256,256), Image.Resampling.NEAREST).save(OUT / 'model_input_mean.png')
    Image.fromarray(result).resize((256,256), Image.Resampling.NEAREST).save(OUT / 'mean_painting.png')
    report_path = OUT / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    report.update({'artworks': n, 'model_trained': True, 'epochs': epochs,
                   'train_artworks': len(training), 'validation_artworks': len(validation),
                   'final_val_l1': history[-1]['val_l1'], 'status': 'mean_painting_ready',
                   'model_definition': 'compact pix2pix-style cGAN: degraded 8x8 input to 64x64 artwork'})
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    collect_parser = sub.add_parser('collect')
    scope = collect_parser.add_mutually_exclusive_group()
    scope.add_argument('--limit', type=int, default=200)
    scope.add_argument('--all', action='store_true', help='collect every matching API record; potentially very large')
    collect_parser.add_argument('--classifications', default=DEFAULT_CLASSES)
    collect_parser.add_argument('--delay', type=float, default=.15)
    mean_parser = sub.add_parser('mean')
    mean_parser.add_argument('--size', type=int, default=256)
    train_parser = sub.add_parser('train')
    train_parser.add_argument('--epochs', type=int, default=30)
    train_parser.add_argument('--size', type=int, default=64)
    args = parser.parse_args()
    if args.command == 'collect': collect(None if args.all else args.limit, args.classifications, args.delay)
    elif args.command == 'mean': means(args.size)
    else: train(args.epochs, args.size)
