"""NGA Open Data pilot using bulk CSV files and public open-access IIIF images."""
import argparse
import csv
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
import torch
from torch import nn
from torch.nn import functional as F

from lab import Generator, Discriminator

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'nga'
SOURCE = DATA / 'source'
IMAGES = DATA / 'images'
OUT = ROOT / 'outputs' / 'nga'
OBJECTS_URL = 'https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/objects.csv'
PUBLISHED_URL = 'https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/published_images.csv'
FLAT_CLASSIFICATIONS = {'painting', 'drawing', 'print', 'photograph'}


class GeneratorHD(nn.Module):
    """128px U-Net with four scales and more capacity than the 64px pilot."""
    def __init__(self):
        super().__init__()
        self.e1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.e2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.InstanceNorm2d(64))
        self.e3 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128))
        self.e4 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256))
        self.d1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.InstanceNorm2d(128))
        self.d2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.InstanceNorm2d(64))
        self.d3 = nn.Sequential(nn.ConvTranspose2d(128, 32, 4, 2, 1), nn.InstanceNorm2d(32))
        self.out = nn.ConvTranspose2d(64, 3, 4, 2, 1)

    def forward(self, x):
        a = F.leaky_relu(self.e1(x), .2)
        b = F.leaky_relu(self.e2(a), .2)
        c = F.leaky_relu(self.e3(b), .2)
        z = F.leaky_relu(self.e4(c), .2)
        d = F.relu(self.d1(z))
        e = F.relu(self.d2(torch.cat([d, c], 1)))
        f = F.relu(self.d3(torch.cat([e, b], 1)))
        return torch.tanh(self.out(torch.cat([f, a], 1)))


class DiscriminatorHD(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(6, 32, 4, 2, 1), nn.LeakyReLU(.2),
                                 nn.Conv2d(32, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.LeakyReLU(.2),
                                 nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(.2),
                                 nn.Conv2d(128, 1, 3, 1, 1))

    def forward(self, x, y):
        return self.net(torch.cat([x, y], 1))


class Generator512(nn.Module):
    """Six-scale U-Net. A 512px image reaches an 8x8 bottleneck."""
    def __init__(self):
        super().__init__()
        def down(a, b, norm=True):
            layers=[nn.Conv2d(a,b,4,2,1)]
            if norm: layers.append(nn.InstanceNorm2d(b))
            return nn.Sequential(*layers)
        def up(a, b):
            return nn.Sequential(nn.ConvTranspose2d(a,b,4,2,1),nn.InstanceNorm2d(b))
        self.e1=down(3,32,False); self.e2=down(32,64); self.e3=down(64,128)
        self.e4=down(128,256); self.e5=down(256,512); self.e6=down(512,512)
        self.d1=up(512,512); self.d2=up(1024,256); self.d3=up(512,128)
        self.d4=up(256,64); self.d5=up(128,32); self.out=nn.ConvTranspose2d(64,3,4,2,1)

    def forward(self,x):
        a=F.leaky_relu(self.e1(x),.2); b=F.leaky_relu(self.e2(a),.2)
        c=F.leaky_relu(self.e3(b),.2); d=F.leaky_relu(self.e4(c),.2)
        e=F.leaky_relu(self.e5(d),.2); z=F.leaky_relu(self.e6(e),.2)
        u=F.relu(self.d1(z)); v=F.relu(self.d2(torch.cat([u,e],1)))
        w=F.relu(self.d3(torch.cat([v,d],1))); q=F.relu(self.d4(torch.cat([w,c],1)))
        r=F.relu(self.d5(torch.cat([q,b],1)))
        return torch.tanh(self.out(torch.cat([r,a],1)))


class Discriminator512(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(6,32,4,2,1),nn.LeakyReLU(.2),
            nn.Conv2d(32,64,4,2,1),nn.InstanceNorm2d(64),nn.LeakyReLU(.2),
            nn.Conv2d(64,128,4,2,1),nn.InstanceNorm2d(128),nn.LeakyReLU(.2),
            nn.Conv2d(128,256,4,2,1),nn.InstanceNorm2d(256),nn.LeakyReLU(.2),
            nn.Conv2d(256,1,3,1,1))
    def forward(self,x,y): return self.net(torch.cat([x,y],1))


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f'Using existing {path.name} ({path.stat().st_size/1_000_000:.1f} MB)', flush=True)
        return
    temporary = path.with_suffix(path.suffix + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'WhoseMean research prototype/0.2'})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open('wb') as output:
        shutil.copyfileobj(response, output, 1024 * 1024)
    temporary.replace(path)
    print(f'Downloaded {path.name} ({path.stat().st_size/1_000_000:.1f} MB)', flush=True)


def metadata():
    download(OBJECTS_URL, SOURCE / 'objects.csv')
    download(PUBLISHED_URL, SOURCE / 'published_images.csv')


def is_flat(record):
    # Keep the initial corpus interpretable. Broad visual-browser tags can pull in
    # furniture and vessels depicted by Index of American Design drawings.
    return record.get('classification', '').strip().casefold() in FLAT_CLASSIFICATIONS


def candidates():
    objects_path, images_path = SOURCE/'objects.csv', SOURCE/'published_images.csv'
    if not objects_path.exists() or not images_path.exists():
        raise SystemExit('Run metadata first.')
    selected = {}
    with objects_path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('accessioned') == '1' and row.get('isvirtual') != '1' and is_flat(row):
                selected[row['objectid']] = row
    primary = {}
    with images_path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            object_id = row.get('depictstmsobjectid')
            if object_id in selected and row.get('openaccess') == '1' and row.get('viewtype') == 'primary':
                previous = primary.get(object_id)
                if previous is None or int(row.get('sequence') or 9999) < int(previous.get('sequence') or 9999):
                    primary[object_id] = row
    joined = []
    for object_id, image in primary.items():
        obj = selected[object_id]
        joined.append({'objectid': int(object_id), 'title': obj.get('title'),
                       'displaydate': obj.get('displaydate'), 'attribution': obj.get('attribution'),
                       'classification': obj.get('classification'), 'subclassification': obj.get('subclassification'),
                       'medium': obj.get('medium'), 'creditline': obj.get('creditline'),
                       'image_uuid': image.get('uuid'), 'iiifurl': image.get('iiifurl'),
                       'width': int(image.get('width') or 0), 'height': int(image.get('height') or 0),
                       'assistivetext': image.get('assistivetext'),
                       'source': f'https://www.nga.gov/artworks/{object_id}', 'openaccess': True})
    return joined


def collect(limit=300, all_images=False, delay=.04):
    metadata()
    pool = candidates()
    rng = random.Random(42)
    rng.shuffle(pool)
    chosen = pool if all_images else pool[:limit]
    IMAGES.mkdir(parents=True, exist_ok=True)
    saved, failures = [], []
    for index, record in enumerate(chosen, 1):
        target = IMAGES / f"{record['objectid']}.jpg"
        try:
            if not target.exists():
                url = record['iiifurl'].rstrip('/') + '/full/!768,768/0/default.jpg'
                request = urllib.request.Request(url, headers={'User-Agent': 'WhoseMean research prototype/0.2'})
                with urllib.request.urlopen(request, timeout=90) as response, target.open('wb') as output:
                    shutil.copyfileobj(response, output)
                with Image.open(target) as check:
                    check.verify()
                time.sleep(delay)
            record['localfile'] = target.name
            record['collected_at'] = datetime.now(timezone.utc).isoformat()
            saved.append(record)
            console_encoding = sys.stdout.encoding or 'utf-8'
            safe_title = str(record['title']).encode(console_encoding, errors='replace').decode(console_encoding)
            print(f"{index:04d}/{len(chosen)}  {record['classification']}  {safe_title}", flush=True)
        except Exception as error:
            target.unlink(missing_ok=True)
            failures.append({'objectid': record['objectid'], 'reason': type(error).__name__, 'message': str(error)[:300]})
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA/'records.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in saved)+'\n', encoding='utf-8')
    (DATA/'failures.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in failures)+('\n' if failures else ''), encoding='utf-8')
    summary = {'eligible_flat_openaccess_primary_images': len(pool), 'requested': len(chosen),
               'saved': len(saved), 'failed': len(failures), 'seed': 42}
    print(json.dumps(summary, indent=2))
    if len(saved) < 8:
        raise SystemExit('Too few usable images were collected.')


def load_records():
    path = DATA/'records.jsonl'
    if not path.exists():
        raise SystemExit('No NGA records found. Run collect first.')
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def square(path, size):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        image = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (size, size), (238, 236, 230))
        canvas.paste(image, ((size-image.width)//2, (size-image.height)//2))
        return np.asarray(canvas).copy()


def array(size):
    records, pixels, used = load_records(), [], []
    for record in records:
        try:
            pixels.append(square(IMAGES/record['localfile'], size)); used.append(record)
        except Exception:
            pass
    if len(pixels) < 8:
        raise SystemExit('Too few readable NGA images.')
    return np.stack(pixels), used


def means(size=256):
    OUT.mkdir(parents=True, exist_ok=True)
    pixels, records = array(size)
    Image.fromarray(pixels.astype(np.float32).mean(0).round().astype(np.uint8)).save(OUT/'pixel_mean.png')
    Image.fromarray(np.median(pixels, axis=0).round().astype(np.uint8)).save(OUT/'pixel_median.png')
    counts = {}
    for record in records:
        label = record.get('classification') or 'Unknown'
        counts[label] = counts.get(label, 0) + 1
    report = {'institution': 'National Gallery of Art, Washington',
              'dataset': 'NGA Open Data bulk CSV + open-access IIIF images', 'license_filter': 'openaccess=1',
              'artworks': len(records), 'classifications': counts, 'sample_seed': 42,
              'mean_definition': 'per-pixel arithmetic mean after contain-fit on RGB(238,236,230)',
              'median_definition': 'per-pixel channel median with identical preprocessing',
              'resolution': size, 'generated_at': datetime.now(timezone.utc).isoformat(),
              'status': 'statistical_means_ready', 'model_trained': False}
    (OUT/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def train(epochs=30, size=64):
    OUT.mkdir(parents=True, exist_ok=True)
    images, records = array(size)
    target = torch.from_numpy(images).permute(0,3,1,2).float()/127.5-1
    source = F.interpolate(F.interpolate(target,(8,8),mode='area'),(size,size),mode='bilinear',align_corners=False)
    order = torch.randperm(len(target), generator=torch.Generator().manual_seed(42))
    n_val = max(1, round(len(target)*.1)); validation, training = order[:n_val], order[n_val:]
    torch.manual_seed(42); g, d = Generator(), Discriminator()
    og=torch.optim.Adam(g.parameters(),lr=.0003,betas=(.5,.999)); od=torch.optim.Adam(d.parameters(),lr=.0003,betas=(.5,.999))
    bce=nn.BCEWithLogitsLoss(); history=[]
    for epoch in range(1,epochs+1):
        losses=[]
        for ids in training[torch.randperm(len(training))].split(8):
            a,b=source[ids],target[ids]; fake=g(a); od.zero_grad()
            rs,fs=d(a,b),d(a,fake.detach()); ld=(bce(rs,torch.ones_like(rs))+bce(fs,torch.zeros_like(fs)))/2
            ld.backward(); od.step()
            for p in d.parameters(): p.requires_grad_(False)
            og.zero_grad(); score=d(a,fake); lg=bce(score,torch.ones_like(score))+50*F.l1_loss(fake,b)
            lg.backward(); og.step()
            for p in d.parameters(): p.requires_grad_(True)
            losses.append((lg.item(),ld.item()))
        with torch.inference_mode(): val=F.l1_loss(g(source[validation]),target[validation]).item()
        history.append({'epoch':epoch,'val_l1':val,'g':float(np.mean(losses,axis=0)[0]),'d':float(np.mean(losses,axis=0)[1])})
        if epoch%5==0 or epoch==epochs: print(json.dumps(history[-1]),flush=True)
    torch.save(g.state_dict(),OUT/'model.pt'); (OUT/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    with torch.inference_mode(): mean_input=source.mean(0,keepdim=True); generated=g(mean_input)
    def rgb(t): return ((t[0].permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
    Image.fromarray(rgb(mean_input)).resize((256,256),Image.Resampling.NEAREST).save(OUT/'model_input_mean.png')
    Image.fromarray(rgb(generated)).resize((256,256),Image.Resampling.NEAREST).save(OUT/'mean_painting.png')
    report=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
    report.update({'model_trained':True,'epochs':epochs,'train_artworks':len(training),'validation_artworks':len(validation),
                   'final_val_l1':history[-1]['val_l1'],'status':'mean_painting_ready',
                   'model_definition':'compact pix2pix-style cGAN: degraded 8x8 input to 64x64 artwork'})
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def edge_loss(output, target):
    horizontal = F.l1_loss(output[:,:,:,1:]-output[:,:,:,:-1], target[:,:,:,1:]-target[:,:,:,:-1])
    vertical = F.l1_loss(output[:,:,1:,:]-output[:,:,:-1,:], target[:,:,1:,:]-target[:,:,:-1,:])
    return horizontal + vertical


def train_hd(epochs=50, size=128):
    if size != 128:
        raise SystemExit('The HD architecture is currently verified at 128px.')
    OUT.mkdir(parents=True, exist_ok=True)
    images, records = array(size)
    target = torch.from_numpy(images).permute(0,3,1,2).float()/127.5-1
    source = F.interpolate(F.interpolate(target,(16,16),mode='area'),(size,size),mode='bilinear',align_corners=False)
    order = torch.randperm(len(target), generator=torch.Generator().manual_seed(42))
    n_val=max(1,round(len(target)*.1)); validation,training=order[:n_val],order[n_val:]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42); g,d=GeneratorHD().to(device),DiscriminatorHD().to(device)
    og=torch.optim.Adam(g.parameters(),lr=.0002,betas=(.5,.999)); od=torch.optim.Adam(d.parameters(),lr=.0002,betas=(.5,.999))
    bce=nn.BCEWithLogitsLoss(); history=[]
    for epoch in range(1,epochs+1):
        losses=[]
        for ids in training[torch.randperm(len(training))].split(4):
            a,b=source[ids].to(device),target[ids].to(device); fake=g(a); od.zero_grad(set_to_none=True)
            rs,fs=d(a,b),d(a,fake.detach()); ld=(bce(rs,torch.ones_like(rs))+bce(fs,torch.zeros_like(fs)))/2
            ld.backward(); od.step()
            for p in d.parameters(): p.requires_grad_(False)
            og.zero_grad(set_to_none=True); score=d(a,fake)
            reconstruction=F.l1_loss(fake,b); edges=edge_loss(fake,b)
            lg=bce(score,torch.ones_like(score))+25*reconstruction+10*edges
            lg.backward(); og.step()
            for p in d.parameters(): p.requires_grad_(True)
            losses.append((lg.item(),ld.item(),reconstruction.item(),edges.item()))
        with torch.inference_mode():
            val=F.l1_loss(g(source[validation].to(device)),target[validation].to(device)).item()
        avg=np.mean(losses,axis=0)
        history.append({'epoch':epoch,'val_l1':val,'g':float(avg[0]),'d':float(avg[1]),'reconstruction':float(avg[2]),'edge':float(avg[3])})
        if epoch%5==0 or epoch==epochs: print(json.dumps(history[-1]),flush=True)
    torch.save(g.state_dict(),OUT/'model_hd.pt'); (OUT/'history_hd.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    with torch.inference_mode():
        mean_input=source.mean(0,keepdim=True).to(device); generated=g(mean_input).cpu(); mean_input=mean_input.cpu()
    def rgb(t): return ((t[0].permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
    Image.fromarray(rgb(mean_input)).resize((512,512),Image.Resampling.LANCZOS).save(OUT/'model_input_mean_hd.png')
    Image.fromarray(rgb(generated)).resize((512,512),Image.Resampling.LANCZOS).save(OUT/'mean_painting_hd.png')
    report=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
    report.update({'hd_model_trained':True,'hd_epochs':epochs,'hd_native_resolution':size,
                   'hd_final_val_l1':history[-1]['val_l1'],'hd_device':str(device),
                   'hd_model_definition':'128px four-scale U-Net cGAN; 16px degraded input; adversarial + L1 + edge losses'})
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def prepare_interactive(size=96):
    """Create interpretable 3D coordinates from image color, light, and edge statistics."""
    pixels, records = array(size)
    raw=[]
    for image in pixels.astype(np.float32)/255:
        gray=image.mean(2)
        warmth=float((image[:,:,0]-image[:,:,2]).mean())
        light=float(gray.mean())
        edge=float((np.abs(np.diff(gray,axis=0)).mean()+np.abs(np.diff(gray,axis=1)).mean())/2)
        saturation=float((image.max(2)-image.min(2)).mean())
        raw.append([warmth,light,edge,saturation])
    raw=np.asarray(raw)
    mean=raw.mean(0); std=raw.std(0)+1e-8
    normalized=np.clip((raw-mean)/std,-2.5,2.5)/2.5
    works=[]
    for record,values,coords in zip(records,raw,normalized):
        works.append({'id':record['objectid'],'title':record.get('title'),'artist':record.get('attribution'),
            'date':record.get('displaydate'),'classification':record.get('classification'),'medium':record.get('medium'),
            'source':record.get('source'),'x':float(coords[0]),'y':float(coords[1]),'z':float(coords[2]),
            'warmth':float(values[0]),'light':float(values[1]),'edge':float(values[2]),
            'saturation':float(values[3]),'mean_weight':1/len(records)})
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'works.json').write_text(json.dumps({'count':len(works),'axes':{'x':'warmth (red minus blue)',
        'y':'mean luminance','z':'edge density'},'works':works},ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'works':len(works),'output':str(OUT/'works.json')}))


def train_512(epochs=15,batch_size=1):
    OUT.mkdir(parents=True,exist_ok=True)
    # Each training run gets its own frame directory so an open browser can keep
    # reading the previous run while Windows/OneDrive writes the new one.
    frame_dir_name=f'frames_512_e{epochs}'
    frames=OUT/frame_dir_name; frames.mkdir(exist_ok=True)
    images,records=array(512)
    target=torch.from_numpy(images).permute(0,3,1,2).float()/127.5-1
    source=F.interpolate(F.interpolate(target,(32,32),mode='area'),(512,512),mode='bilinear',align_corners=False)
    order=torch.randperm(len(target),generator=torch.Generator().manual_seed(42)); n_val=max(1,round(len(target)*.1))
    validation,training=order[:n_val],order[n_val:]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); use_amp=device.type=='cuda'
    if use_amp: torch.backends.cudnn.benchmark=True
    torch.manual_seed(42)
    if use_amp: torch.cuda.manual_seed_all(42)
    g,d=Generator512().to(device),Discriminator512().to(device)
    og=torch.optim.Adam(g.parameters(),lr=.0002,betas=(.5,.999)); od=torch.optim.Adam(d.parameters(),lr=.0002,betas=(.5,.999))
    scaler_g=torch.amp.GradScaler('cuda',enabled=use_amp); scaler_d=torch.amp.GradScaler('cuda',enabled=use_amp)
    bce=nn.BCEWithLogitsLoss(); history=[]; fixed=source.mean(0,keepdim=True)
    best_val=float('inf'); best_epoch=0
    def save_frame(epoch):
        was_training=g.training; g.eval()
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp): result=g(fixed.to(device)).float().cpu()
        if was_training:g.train()
        image=((result[0].permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
        Image.fromarray(image).save(frames/f'epoch_{epoch:03d}.png')
    save_frame(0)
    for epoch in range(1,epochs+1):
        started=time.perf_counter(); losses=[]
        for ids in training[torch.randperm(len(training))].split(batch_size):
            a,b=source[ids].to(device,non_blocking=True),target[ids].to(device,non_blocking=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp): fake=g(a)
            od.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
                rs,fs=d(a,b),d(a,fake.detach()); ld=(bce(rs,torch.ones_like(rs))+bce(fs,torch.zeros_like(fs)))/2
            scaler_d.scale(ld).backward(); scaler_d.step(od); scaler_d.update()
            for p in d.parameters():p.requires_grad_(False)
            og.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
                score=d(a,fake); reconstruction=F.l1_loss(fake,b); edges=edge_loss(fake,b)
                lg=bce(score,torch.ones_like(score))+20*reconstruction+12*edges
            scaler_g.scale(lg).backward(); scaler_g.step(og); scaler_g.update()
            for p in d.parameters():p.requires_grad_(True)
            losses.append((lg.item(),ld.item(),reconstruction.item(),edges.item()))
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
            val=F.l1_loss(g(source[validation].to(device)),target[validation].to(device)).item()
        avg=np.mean(losses,axis=0); elapsed=time.perf_counter()-started
        history.append({'epoch':epoch,'val_l1':val,'g':float(avg[0]),'d':float(avg[1]),
            'reconstruction':float(avg[2]),'edge':float(avg[3]),'seconds':elapsed})
        if val < best_val:
            best_val=val; best_epoch=epoch
            torch.save(g.state_dict(),OUT/'model_512_best.pt')
        save_frame(epoch); (OUT/'history_512.json').write_text(json.dumps(history),encoding='utf-8')
        if epoch%5==0:torch.save(g.state_dict(),OUT/f'model_512_epoch_{epoch:03d}.pt')
        print(json.dumps(history[-1]),flush=True)
    torch.save(g.state_dict(),OUT/'model_512.pt')
    best_frame=frames/f'epoch_{best_epoch:03d}.png'; shutil.copyfile(best_frame,OUT/'mean_painting_512.png')
    input_image=((fixed[0].permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
    Image.fromarray(input_image).save(OUT/'model_input_mean_512.png')
    report=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
    report.update({'model_512_trained':True,'model_512_epochs':epochs,'model_512_native_resolution':512,
        'model_512_final_val_l1':history[-1]['val_l1'],'model_512_best_val_l1':best_val,
        'model_512_best_epoch':best_epoch,'model_512_device':str(device),
        'model_512_seconds':sum(h['seconds'] for h in history),'model_512_frames':epochs+1,
        'model_512_frame_dir':frame_dir_name,
        'model_512_definition':'512px six-scale U-Net cGAN; 32px degraded input; adversarial + L1 + edge losses'})
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


PAINTING_LANDSCAPE_IDS={130865,83883,138735,111194,92234,55818,50302,89680,
                        130897,53138,56571,12158}


def train_pix2pix_subset(classification='Painting',epochs=160,batch_size=1,theme='all'):
    """A pure pix2pix comparison on one coherent museum classification."""
    slug=classification.lower().replace(' ','_')+(f'_{theme}' if theme!='all' else '')
    output=OUT/f'pix2pix_{slug}'; frames=output/'frames'
    output.mkdir(parents=True,exist_ok=True); frames.mkdir(exist_ok=True)
    all_images,all_records=array(512)
    keep=[i for i,r in enumerate(all_records) if r.get('classification')==classification and
          (theme=='all' or (theme=='landscape' and int(r['objectid']) in PAINTING_LANDSCAPE_IDS))]
    if len(keep)<8: raise SystemExit(f'Too few {classification} images: {len(keep)}')
    images=all_images[keep]
    target=torch.from_numpy(images).permute(0,3,1,2).float()/127.5-1
    # A 64px paired input retains coarse subject placement while removing texture.
    source=F.interpolate(F.interpolate(target,(64,64),mode='area'),(512,512),mode='bilinear',align_corners=False)
    order=torch.randperm(len(target),generator=torch.Generator().manual_seed(42))
    n_val=max(2,round(len(target)*.12)); validation,training=order[:n_val],order[n_val:]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); use_amp=device.type=='cuda'
    torch.manual_seed(42)
    if use_amp: torch.cuda.manual_seed_all(42)
    g,d=Generator512().to(device),Discriminator512().to(device)
    og=torch.optim.Adam(g.parameters(),lr=.0002,betas=(.5,.999))
    od=torch.optim.Adam(d.parameters(),lr=.0001,betas=(.5,.999))
    scaler_g=torch.amp.GradScaler('cuda',enabled=use_amp)
    scaler_d=torch.amp.GradScaler('cuda',enabled=use_amp)
    bce=nn.BCEWithLogitsLoss(); history=[]; fixed=source.mean(0,keepdim=True)
    best_val=float('inf'); best_epoch=0
    def render(path):
        was_training=g.training; g.eval()
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
            result=g(fixed.to(device)).float().cpu()
        if was_training:g.train()
        pixels=((result[0].permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
        Image.fromarray(pixels).save(path)
    render(frames/'epoch_000.png')
    for epoch in range(1,epochs+1):
        started=time.perf_counter(); losses=[]
        for ids in training[torch.randperm(len(training))].split(batch_size):
            a,b=source[ids].to(device),target[ids].to(device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp): fake=g(a)
            od.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
                real,fake_score=d(a,b),d(a,fake.detach())
                ld=(bce(real,torch.full_like(real,.9))+bce(fake_score,torch.zeros_like(fake_score)))/2
            scaler_d.scale(ld).backward();scaler_d.step(od);scaler_d.update()
            for p in d.parameters():p.requires_grad_(False)
            og.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
                score=d(a,fake); reconstruction=F.l1_loss(fake,b); edges=edge_loss(fake,b)
                lg=bce(score,torch.ones_like(score))+40*reconstruction+15*edges
            scaler_g.scale(lg).backward();scaler_g.step(og);scaler_g.update()
            for p in d.parameters():p.requires_grad_(True)
            losses.append((lg.item(),ld.item(),reconstruction.item(),edges.item()))
        g.eval()
        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
            val=F.l1_loss(g(source[validation].to(device)),target[validation].to(device)).item()
        g.train(); avg=np.mean(losses,axis=0)
        row={'epoch':epoch,'val_l1':val,'g':float(avg[0]),'d':float(avg[1]),
             'reconstruction':float(avg[2]),'edge':float(avg[3]),'seconds':time.perf_counter()-started}
        history.append(row)
        if val<best_val:
            best_val=val;best_epoch=epoch;torch.save(g.state_dict(),output/'model_best.pt')
        if epoch%5==0 or epoch==epochs: render(frames/f'epoch_{epoch:03d}.png')
        if epoch%10==0: print(json.dumps(row),flush=True)
    g.load_state_dict(torch.load(output/'model_best.pt',map_location=device,weights_only=True))
    render(output/'mean_painting.png')
    input_pixels=((fixed[0].permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
    Image.fromarray(input_pixels).save(output/'mean_input.png')
    (output/'history.json').write_text(json.dumps(history),encoding='utf-8')
    report={'method':'pure pix2pix-style cGAN, no pretrained image generator',
            'classification':classification,'theme':theme,'images':len(keep),'train_images':len(training),
            'validation_images':len(validation),'epochs':epochs,'best_epoch':best_epoch,
            'best_val_l1':best_val,'input_resolution':64,'output_resolution':512,
            'loss':'adversarial + 40*L1 + 15*edge','device':str(device)}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('metadata')
    collect_parser=sub.add_parser('collect'); collect_parser.add_argument('--limit',type=int,default=300); collect_parser.add_argument('--all',action='store_true'); collect_parser.add_argument('--delay',type=float,default=.04)
    mean_parser=sub.add_parser('mean'); mean_parser.add_argument('--size',type=int,default=256)
    train_parser=sub.add_parser('train'); train_parser.add_argument('--epochs',type=int,default=30); train_parser.add_argument('--size',type=int,default=64)
    hd_parser=sub.add_parser('train-hd'); hd_parser.add_argument('--epochs',type=int,default=50); hd_parser.add_argument('--size',type=int,default=128)
    full_parser=sub.add_parser('train-512'); full_parser.add_argument('--epochs',type=int,default=50); full_parser.add_argument('--batch-size',type=int,default=1)
    subset_parser=sub.add_parser('train-pix2pix-subset'); subset_parser.add_argument('--classification',default='Painting')
    subset_parser.add_argument('--theme',choices=['all','landscape'],default='all')
    subset_parser.add_argument('--epochs',type=int,default=160); subset_parser.add_argument('--batch-size',type=int,default=1)
    viz_parser=sub.add_parser('visualize-data'); viz_parser.add_argument('--size',type=int,default=96)
    args=parser.parse_args()
    if args.command=='metadata': metadata()
    elif args.command=='collect': collect(args.limit,args.all,args.delay)
    elif args.command=='mean': means(args.size)
    elif args.command=='train': train(args.epochs,args.size)
    elif args.command=='train-hd': train_hd(args.epochs,args.size)
    elif args.command=='train-512': train_512(args.epochs,args.batch_size)
    elif args.command=='train-pix2pix-subset': train_pix2pix_subset(args.classification,args.epochs,args.batch_size,args.theme)
    else: prepare_interactive(args.size)
