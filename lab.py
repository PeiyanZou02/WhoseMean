"""Small, actual paired conditional GAN experiment; synthetic data only by default."""
import argparse
import base64
import functools
import io
import json
import random
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
NGA_DATA = ROOT / 'data' / 'nga'
torch.set_num_threads(4)
_model_512_lock = threading.Lock()
_model_512_cache = {'mtime': None, 'model': None, 'device': None}
_painting_model_cache = {'mtime': None, 'model': None, 'device': None}


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Conv2d(3, 16, 4, 2, 1)
        self.e2 = nn.Conv2d(16, 32, 4, 2, 1)
        self.e3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.d1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.d2 = nn.ConvTranspose2d(64, 16, 4, 2, 1)
        self.out = nn.ConvTranspose2d(32, 3, 4, 2, 1)

    def forward(self, x, channel=-1, capture=False):
        a = F.leaky_relu(self.e1(x), .2)
        b = F.leaky_relu(self.e2(a), .2)
        c = F.leaky_relu(self.e3(b), .2)
        if channel >= 0:
            mask = torch.ones_like(c)
            mask[:, channel] = 0
            c = c * mask
        d = F.relu(self.d1(c))
        e = F.relu(self.d2(torch.cat([d, b], 1)))
        y = torch.tanh(self.out(torch.cat([e, a], 1)))
        return (y, [a, b, c, d, e]) if capture else y


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(6, 16, 4, 2, 1), nn.LeakyReLU(.2),
                                 nn.Conv2d(16, 32, 4, 2, 1), nn.LeakyReLU(.2),
                                 nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, x, y):
        return self.net(torch.cat([x, y], 1))


@functools.lru_cache(maxsize=1)
def dataset():
    """96 invented compositions, balanced warm/cool groups; no museum images."""
    rng = random.Random(42)
    images = []
    for i in range(96):
        warm = i < 48
        im = Image.new('RGB', (64, 64), (223, 211, 185) if warm else (181, 204, 218))
        draw = ImageDraw.Draw(im)
        for _ in range(7):
            x, y = rng.randrange(50), rng.randrange(50)
            color = ((rng.randrange(130, 240), rng.randrange(40, 160), rng.randrange(20, 100))
                     if warm else (rng.randrange(20, 100), rng.randrange(70, 170), rng.randrange(130, 240)))
            box = (x, y, min(63, x + rng.randrange(8, 30)), min(63, y + rng.randrange(8, 30)))
            (draw.ellipse if rng.random() < .5 else draw.rectangle)(box, fill=color)
        images.append(np.asarray(im).copy())
    target = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 127.5 - 1
    source = F.interpolate(F.interpolate(target, (8, 8), mode='area'), (64, 64), mode='bilinear', align_corners=False)
    return source, target


def pil(t):
    a = ((t.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype('uint8')
    return Image.fromarray(a)


def encoded(im):
    b = io.BytesIO()
    im.save(b, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()


def collection_status():
    folder = OUT / 'nga'
    if not (folder / 'report.json').exists():
        folder = OUT / 'harvard'
    report_path = folder / 'report.json'
    if not report_path.exists():
        return {'ready': False, 'message': '尚未生成真实馆藏结果。NGA 数据无需 API key，可直接运行下载。'}
    report = json.loads(report_path.read_text(encoding='utf-8'))
    result = {'ready': True, 'report': report, 'images': {}}
    for name in ['pixel_mean', 'pixel_median', 'model_input_mean', 'mean_painting',
                 'model_input_mean_hd', 'mean_painting_hd', 'model_input_mean_512',
                 'mean_painting_512']:
        path = folder / f'{name}.png'
        if path.exists():
            with Image.open(path) as image:
                result['images'][name] = encoded(image.convert('RGB'))
    return result


@functools.lru_cache(maxsize=1)
def nga_records():
    path=NGA_DATA/'records.jsonl'
    if not path.exists(): return {}
    return {int(r['objectid']):r for r in (json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip())}


def work_canvas(record, size=512):
    path=NGA_DATA/'images'/record['localfile']
    if not path.exists(): raise ValueError('Work image is missing')
    with Image.open(path) as source:
        source=ImageOps.exif_transpose(source).convert('RGB')
        display=ImageOps.contain(source,(size,size),Image.Resampling.LANCZOS)
        canvas=Image.new('RGB',(size,size),(238,236,230))
        canvas.paste(display,((size-display.width)//2,(size-display.height)//2))
    return canvas


@functools.lru_cache(maxsize=1)
def thumbnail_atlas():
    works_path=OUT/'nga'/'works.json'
    if not works_path.exists(): raise ValueError('Interactive work data is not ready')
    works=json.loads(works_path.read_text(encoding='utf-8'))['works']
    tile=64; columns=20; rows=(len(works)+columns-1)//columns
    atlas=Image.new('RGB',(columns*tile,rows*tile),(238,236,230))
    source_records=nga_records()
    for index,work in enumerate(works):
        record=source_records.get(int(work['id']))
        if record: atlas.paste(work_canvas(record,tile),((index%columns)*tile,(index//columns)*tile))
    buffer=io.BytesIO(); atlas.save(buffer,format='JPEG',quality=82,optimize=True)
    return buffer.getvalue()


@functools.lru_cache(maxsize=1)
def nga_sum_512():
    records=nga_records()
    total=np.zeros((512,512,3),dtype=np.float64)
    for record in records.values():
        total += np.asarray(work_canvas(record),dtype=np.float64)
    return total


def work_detail(object_id):
    record=nga_records().get(object_id)
    if not record: raise ValueError('Work does not exist')
    canvas=work_canvas(record)
    count=max(2,len(nga_records()))
    mean=(nga_sum_512()/count).astype(np.float32)
    pixels=np.asarray(canvas).astype(np.float32); difference=np.abs(pixels-mean).mean(2)
    # Fixed scale: a 64-level channel difference maps to white. No per-image normalization.
    visual=(np.clip(difference/64,0,1)*255).astype(np.uint8)
    heat=Image.fromarray(visual).convert('RGB')
    return {'record':record,'image':encoded(canvas),'difference':encoded(heat),
            'count':count,'mean_weight':1/count,
            'leave_one_out_delta':float(difference.mean()/255/(count-1)),
            'difference_scale':'black=0, white=64+ mean RGB levels'}


def loaded_512_model():
    from nga_pipeline import Generator512
    folder=OUT/'nga'; path=folder/'model_512_best.pt'
    if not path.exists(): path=folder/'model_512.pt'
    if not path.exists(): raise ValueError('512 model is not ready')
    mtime=path.stat().st_mtime_ns
    with _model_512_lock:
        if _model_512_cache['mtime'] != mtime:
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model=Generator512().to(device).eval()
            model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
            _model_512_cache.update({'mtime':mtime,'model':model,'device':device})
        return _model_512_cache['model'],_model_512_cache['device']


def exclusion_result(object_ids):
    records=nga_records(); ids=list(dict.fromkeys(object_ids))
    if not ids: raise ValueError('Select at least one work')
    if len(ids)>50: raise ValueError('Select no more than 50 works')
    if any(i not in records for i in ids): raise ValueError('One or more works do not exist')
    count=len(records); total=nga_sum_512()
    removed=np.zeros_like(total)
    for object_id in ids:
        removed += np.asarray(work_canvas(records[object_id]),dtype=np.float64)
    baseline=total/count; counterfactual=(total-removed)/(count-len(ids))
    exact_difference=np.abs(counterfactual-baseline).mean(2)
    exact_visual=(np.clip(exact_difference/4,0,1)*255).astype(np.uint8)
    counter_image=Image.fromarray(np.clip(counterfactual,0,255).round().astype(np.uint8))

    def source_tensor(pixels):
        tensor=torch.from_numpy(pixels.astype(np.float32)).permute(2,0,1).unsqueeze(0)/127.5-1
        return F.interpolate(F.interpolate(tensor,(32,32),mode='area'),(512,512),mode='bilinear',align_corners=False)
    model,device=loaded_512_model()
    with _model_512_lock,torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
        base_generated=model(source_tensor(baseline).to(device)).float().cpu()
        counter_generated=model(source_tensor(counterfactual).to(device)).float().cpu()
    generated_pixels=((counter_generated[0].permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
    generated_difference=(counter_generated-base_generated).abs().mean(1)[0].numpy()*127.5
    generated_visual=(np.clip(generated_difference/16,0,1)*255).astype(np.uint8)
    return {'removed_count':len(ids),'remaining_count':count-len(ids),
            'exact_mean':encoded(counter_image),
            'generated':encoded(Image.fromarray(generated_pixels)),
            'exact_difference':encoded(Image.fromarray(exact_visual).convert('RGB')),
            'generated_difference':encoded(Image.fromarray(generated_visual).convert('RGB')),
            'exact_mean_delta':float(exact_difference.mean()/255),
            'generated_mean_delta':float(generated_difference.mean()/255),
            'difference_scales':{'exact':'white=4+ RGB levels','generated':'white=16+ RGB levels'}}


@functools.lru_cache(maxsize=1)
def painting_records():
    return {key:value for key,value in nga_records().items() if value.get('classification')=='Painting'}


@functools.lru_cache(maxsize=1)
def painting_sum_512():
    total=np.zeros((512,512,3),dtype=np.float64)
    for record in painting_records().values():
        total += np.asarray(work_canvas(record),dtype=np.float64)
    return total


def loaded_painting_model():
    from nga_pipeline import Generator512
    path=OUT/'nga'/'pix2pix_painting'/'model_best.pt'
    if not path.exists(): raise ValueError('Painting pix2pix model is not ready')
    mtime=path.stat().st_mtime_ns
    with _model_512_lock:
        if _painting_model_cache['mtime'] != mtime:
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model=Generator512().to(device).eval()
            model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
            _painting_model_cache.update({'mtime':mtime,'model':model,'device':device})
        return _painting_model_cache['model'],_painting_model_cache['device']


def exclude_paintings(object_ids):
    records=painting_records(); ids=list(dict.fromkeys(object_ids))
    if not ids: raise ValueError('Select at least one painting')
    if len(ids)>=len(records)-1: raise ValueError('Leave at least two paintings')
    if any(i not in records for i in ids): raise ValueError('Selection contains a non-painting work')
    count=len(records); total=painting_sum_512()
    removed=sum((np.asarray(work_canvas(records[i]),dtype=np.float64) for i in ids),
                start=np.zeros_like(total))
    baseline=total/count; counter=(total-removed)/(count-len(ids))
    def degraded(pixels):
        value=torch.from_numpy(pixels.astype(np.float32)).permute(2,0,1).unsqueeze(0)/127.5-1
        return F.interpolate(F.interpolate(value,(64,64),mode='area'),(512,512),mode='bilinear',align_corners=False)
    model,device=loaded_painting_model()
    with _model_512_lock,torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
        base=model(degraded(baseline).to(device)).float().cpu()
        result=model(degraded(counter).to(device)).float().cpu()
    pixels=((result[0].permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
    return {'removed_count':len(ids),'remaining_count':count-len(ids),'generated':encoded(Image.fromarray(pixels)),
            'exact_mean_delta':float(np.abs(counter-baseline).mean()/255),
            'model_delta':float((result-base).abs().mean().item()/2)}


def painting_training_status():
    folder=OUT/'nga'/'pix2pix_painting'
    report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    history=json.loads((folder/'history.json').read_text(encoding='utf-8'))
    epochs=sorted(int(path.stem.split('_')[-1]) for path in (folder/'frames').glob('epoch_*.png'))
    return {'report':report,'history':history,'epochs':epochs}


def training_512_status():
    folder=OUT/'nga'; history_path=folder/'history_512.json'
    report_path=folder/'report.json'
    report=json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    frame_dir=report.get('model_512_frame_dir','frames_512')
    frames=folder/frame_dir
    history=json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
    available=sorted(int(path.stem.split('_')[-1]) for path in frames.glob('epoch_*.png')) if frames.exists() else []
    return {'ready':bool(available),'history':history,'epochs':available,
            'best_epoch':report.get('model_512_best_epoch',available[-1] if available else 0),
            'frame_dir':frame_dir}


def train(epochs=40):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(42)
    x, y = dataset()
    # Each group contributes 40 training and 8 held-out images.
    ti = torch.tensor(list(range(40)) + list(range(48, 88)))
    vi = torch.tensor(list(range(40, 48)) + list(range(88, 96)))
    g, d = Generator(), Discriminator()
    og = torch.optim.Adam(g.parameters(), lr=.0005, betas=(.5, .999))
    od = torch.optim.Adam(d.parameters(), lr=.0005, betas=(.5, .999))
    bce = nn.BCEWithLogitsLoss()
    history = []
    with torch.no_grad():
        initial = F.l1_loss(g(x[vi]), y[vi]).item()
        input_error = F.l1_loss(x[vi], y[vi]).item()
    torch.save(g.state_dict(), OUT / 'epoch_000.pt')
    history.append({'epoch': 0, 'val_l1': initial, 'g': None, 'd': None})
    for epoch in range(1, epochs + 1):
        losses = []
        for ids in ti[torch.randperm(len(ti))].split(8):
            a, b = x[ids], y[ids]
            fake = g(a)
            od.zero_grad()
            real_score, fake_score = d(a, b), d(a, fake.detach())
            ld = (bce(real_score, torch.ones_like(real_score)) + bce(fake_score, torch.zeros_like(fake_score))) / 2
            ld.backward()
            od.step()
            for p in d.parameters():
                p.requires_grad_(False)
            og.zero_grad()
            score = d(a, fake)
            lg = bce(score, torch.ones_like(score)) + 50 * F.l1_loss(fake, b)
            lg.backward()
            og.step()
            for p in d.parameters():
                p.requires_grad_(True)
            losses.append((lg.item(), ld.item()))
        with torch.no_grad():
            val = F.l1_loss(g(x[vi]), y[vi]).item()
        row = {'epoch': epoch, 'val_l1': val, 'g': float(np.mean(losses, axis=0)[0]), 'd': float(np.mean(losses, axis=0)[1])}
        history.append(row)
        torch.save(g.state_dict(), OUT / f'epoch_{epoch:03d}.pt')
        (OUT / 'history.json').write_text(json.dumps(history), encoding='utf-8')
        if epoch % 5 == 0 or epoch == epochs:
            print(json.dumps(row), flush=True)
    with torch.no_grad():
        mean = x[ti].mean(0, keepdim=True)
        pred = g(mean)
        changed = g(mean, channel=0)
        delta = (pred - changed).abs().mean().item()
        assert torch.isfinite(pred).all() and delta > 0
        assert torch.equal(g(mean), g(mean)), 'Inference must be deterministic'
        canvas = Image.new('RGB', (64 * 4, 64))
        for i, t in enumerate([mean, pred, changed, (pred-changed).abs().clamp(0, 1)*2-1]):
            canvas.paste(pil(t), (i * 64, 0))
        canvas.resize((1024, 256)).save(OUT / 'experiment.png')
    report = {'dataset': '96 synthetic compositions, NOT Harvard artworks', 'train_images': 80,
              'validation_images': 16, 'seed': 42, 'epochs': epochs, 'resolution': 64,
              'torch': torch.__version__, 'device': 'cpu', 'initial_val_l1': initial,
              'final_val_l1': val, 'degraded_input_val_l1': input_error,
              'channel_0_ablation_mean_absolute_change': delta,
              'note': 'Compact pix2pix-style U-Net + conditional patch discriminator; proof of pipeline, not canonical architecture or artistic quality.'}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


@functools.lru_cache(maxsize=6)
def load_generator(epoch):
    path = OUT / f'epoch_{epoch:03d}.pt'
    if not path.exists():
        raise ValueError('Checkpoint does not exist')
    model = Generator().eval()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model


def infer(epoch, weight, channel):
    if not 0 <= weight <= 1 or not -1 <= channel < 64:
        raise ValueError('Invalid weight or channel')
    g = load_generator(epoch)
    x, _ = dataset()
    mean = weight * x[:40].mean(0, keepdim=True) + (1-weight) * x[48:88].mean(0, keepdim=True)
    with torch.inference_mode():
        baseline = g(mean)
        pred, features = g(mean, channel, True)
        diff = (pred-baseline).abs().mean(1)[0] / 2  # Fixed [0,1] scale in display RGB units.
        heat = np.zeros((64, 64, 3), dtype=np.uint8)
        heat[:, :, 0] = (diff.numpy().clip(0, 1) * 255).astype('uint8')
        maps = []
        for f in features:
            # Eight actual channels per layer, each normalized separately for display.
            grid = Image.new('RGB', (32*8, 32))
            for k in range(8):
                a = f[0, k].numpy()
                a = (a-a.min()) / (np.ptp(a)+1e-8)
                grid.paste(Image.fromarray((a*255).astype('uint8')).convert('RGB').resize((32,32)), (32*k, 0))
            maps.append({'shape': list(f.shape[1:]), 'image': encoded(grid)})
    return {'input': encoded(pil(mean)), 'baseline': encoded(pil(baseline)), 'output': encoded(pil(pred)),
            'difference': encoded(Image.fromarray(heat)), 'features': maps,
            'delta': float(diff.mean()), 'epoch': epoch, 'weight': weight, 'channel': channel}


def serve(port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlparse, parse_qs
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path == '/':
                    body, kind = (ROOT/'index.html').read_bytes(), 'text/html; charset=utf-8'
                elif parsed.path == '/api/history':
                    body, kind = (OUT/'history.json').read_bytes(), 'application/json'
                elif parsed.path == '/api/health':
                    body, kind = json.dumps({'status': 'ok', 'service': 'Whose Mean?',
                                             'thread': threading.current_thread().name}).encode(), 'application/json'
                elif parsed.path == '/api/harvard':
                    body, kind = json.dumps(collection_status(), ensure_ascii=False).encode(), 'application/json; charset=utf-8'
                elif parsed.path == '/api/collection':
                    body, kind = json.dumps(collection_status(), ensure_ascii=False).encode(), 'application/json; charset=utf-8'
                elif parsed.path == '/api/works':
                    path=OUT/'nga'/'works.json'
                    if not path.exists(): raise ValueError('Interactive work data is not ready')
                    body, kind = path.read_bytes(), 'application/json; charset=utf-8'
                elif parsed.path == '/api/thumb-atlas':
                    body, kind = thumbnail_atlas(), 'image/jpeg'
                elif parsed.path == '/api/coherent-mean':
                    q=parse_qs(parsed.query); seed=int(q.get('seed',['7'])[0])
                    if seed not in {7,42,123}: raise ValueError('Invalid coherent-mean seed')
                    path=OUT/'diffusion'/'painting'/f'coherent_mean_seed_{seed}.png'
                    if not path.exists(): raise ValueError('Coherent mean is not ready')
                    body, kind = path.read_bytes(), 'image/png'
                elif parsed.path == '/api/pix2pix-subset':
                    q=parse_qs(parsed.query); theme=q.get('theme',['all'])[0]
                    folders={'all':'pix2pix_painting','landscape':'pix2pix_painting_landscape'}
                    if theme not in folders: raise ValueError('Invalid pix2pix subset')
                    path=OUT/'nga'/folders[theme]/'mean_painting.png'
                    if not path.exists(): raise ValueError('Pix2pix subset result is not ready')
                    body, kind = path.read_bytes(), 'image/png'
                elif parsed.path == '/api/painting-training':
                    body, kind = json.dumps(painting_training_status()).encode(), 'application/json'
                elif parsed.path == '/api/live-training':
                    import live_training
                    body, kind = json.dumps(live_training.status()).encode(), 'application/json'
                elif parsed.path == '/api/live-frame':
                    import live_training
                    q=parse_qs(parsed.query); epoch=int(q.get('epoch',['0'])[0]); state=live_training.status()
                    path=OUT/'nga'/'live'/state.get('run_id','')/'frames'/f'epoch_{epoch:03d}.png'
                    if not path.exists(): raise ValueError('Live training frame does not exist')
                    body, kind = path.read_bytes(), 'image/png'
                elif parsed.path == '/api/live-generated-atlas':
                    import live_training
                    state=live_training.status();path=OUT/'nga'/'live'/state.get('run_id','')/'generated_atlas.jpg'
                    if not path.exists(): raise ValueError('Generated result atlas is not ready')
                    body, kind = path.read_bytes(), 'image/jpeg'
                elif parsed.path == '/api/painting-frame':
                    q=parse_qs(parsed.query); epoch=int(q.get('epoch',['50'])[0])
                    path=OUT/'nga'/'pix2pix_painting'/'frames'/f'epoch_{epoch:03d}.png'
                    if not path.exists(): raise ValueError('Painting training frame does not exist')
                    body, kind = path.read_bytes(), 'image/png'
                elif parsed.path == '/api/exclude-paintings':
                    q=parse_qs(parsed.query); raw=q.get('ids',[''])[0]
                    ids=[int(value) for value in raw.split(',') if value.strip()]
                    body, kind = json.dumps(exclude_paintings(ids)).encode(), 'application/json'
                elif parsed.path == '/api/work':
                    q=parse_qs(parsed.query); object_id=int(q.get('id',['0'])[0])
                    body, kind = json.dumps(work_detail(object_id),ensure_ascii=False).encode(), 'application/json; charset=utf-8'
                elif parsed.path == '/api/exclude':
                    q=parse_qs(parsed.query); raw=q.get('ids',[''])[0]
                    ids=[int(value) for value in raw.split(',') if value.strip()]
                    body, kind = json.dumps(exclusion_result(ids)).encode(), 'application/json'
                elif parsed.path == '/api/training-512':
                    body, kind = json.dumps(training_512_status()).encode(), 'application/json'
                elif parsed.path == '/api/image':
                    q=parse_qs(parsed.query); name=q.get('name',[''])[0]
                    allowed={'pixel_mean','pixel_median','mean_painting_512','model_input_mean_512'}
                    if name not in allowed: raise ValueError('Invalid image name')
                    path=OUT/'nga'/f'{name}.png'
                    if not path.exists(): raise ValueError('Image does not exist')
                    body, kind = path.read_bytes(), 'image/png'
                elif parsed.path == '/api/frame-512':
                    q=parse_qs(parsed.query); epoch=int(q.get('epoch',['0'])[0])
                    if epoch<0 or epoch>999: raise ValueError('Invalid epoch')
                    status=training_512_status()
                    path=OUT/'nga'/status['frame_dir']/f'epoch_{epoch:03d}.png'
                    if not path.exists(): raise ValueError('Training frame does not exist')
                    body, kind = path.read_bytes(), 'image/png'
                elif parsed.path == '/api/infer':
                    q = parse_qs(parsed.query)
                    result = infer(int(q.get('epoch', ['40'])[0]), float(q.get('weight', ['.5'])[0]), int(q.get('channel', ['-1'])[0]))
                    body, kind = json.dumps(result).encode(), 'application/json'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True
            except (ValueError, FileNotFoundError) as e:
                self.send_error(400, str(e))
            except (BrokenPipeError, ConnectionResetError):
                # A newer slider request superseded this response.
                pass
        def do_POST(self):
            from urllib.parse import urlparse
            parsed=urlparse(self.path)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length>200_000: raise ValueError('Request is too large')
                payload=json.loads(self.rfile.read(length) or b'{}')
                import live_training
                if parsed.path=='/api/train/start':
                    result=live_training.start(payload.get('excluded',[]),int(payload.get('epochs',20)),
                                               payload.get('prompt',''),float(payload.get('structure',.65)))
                elif parsed.path=='/api/train/stop':
                    result={'stopping':live_training.stop()}
                else:
                    self.send_error(404);return
                body=json.dumps(result).encode()
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
                self.end_headers();self.wfile.write(body)
            except (ValueError,FileNotFoundError,json.JSONDecodeError) as error:
                self.send_error(400,str(error))
        def log_message(self, format, *args):
            # Keep the log useful; canceled slider requests are expected.
            if args and str(args[1]) == '200':
                return
            super().log_message(format, *args)
    print(f'Whose Mean: http://127.0.0.1:{port}', flush=True)
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['train', 'serve'])
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--port', type=int, default=8765)
    args = p.parse_args()
    train(args.epochs) if args.command == 'train' else serve(args.port)
