"""Background 512x512 pix2pix training for the local interactive studio."""
import json
import math
import os
import re
import threading
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from .data import DATA, IMAGES, OUT, load_records, square
from .model import Discriminator512, Generator512, edge_loss

LIVE=OUT/'live'
_thread=None
_stop=threading.Event()
_guard=threading.Lock()

CONCEPT_ALIASES={
    'tree':{'tree','trees','forest','forests','woodland','woods'},
    'dog':{'dog','dogs','hound','hounds','terrier','terriers'},
    'sardine':{'sardine','sardines','fish','fishes'},
}


def matching_ids(prompt):
    works=json.loads((OUT/'works.json').read_text(encoding='utf-8'))['works']
    query_tokens=re.findall(r'[a-z0-9]+',str(prompt).lower())
    if not query_tokens: return {int(work['id']) for work in works}
    concepts=[]
    for token in query_tokens:
        aliases=set(CONCEPT_ALIASES.get(token,{token}))
        if token.isascii():
            aliases.update({token+'s',token[:-1] if token.endswith('s') else token})
        concepts.append(aliases)
    matches=set()
    for work in works:
        content=' '.join(str(work.get(key,'')) for key in ('title','artist','classification','medium')).lower()
        words=set(re.findall(r'[a-z0-9]+',content))
        if any(words & aliases for aliases in concepts):
            matches.add(int(work['id']))
    return matches


def _write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+f'.{os.getpid()}.{threading.get_ident()}.writing')
    temporary.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    for attempt in range(100):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(.02)


def status():
    path=LIVE/'current.json'
    if not path.exists():
        return {'state':'idle','message':'No live training run yet.','frames':[]}
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {'state':'loading','message':'Reading training state.','frames':[]}
    run=LIVE/str(data.get('run_id',''))
    frame_dir=run/'frames'
    data['frames']=sorted(int(p.stem.split('_')[-1]) for p in frame_dir.glob('epoch_*.png')) if frame_dir.exists() else []
    data['has_generated_atlas']=(LIVE/str(data.get('run_id',''))/'generated_atlas.jpg').exists()
    data['has_individual_results']=(LIVE/str(data.get('run_id',''))/'generated').exists()
    return data


def start(excluded,epochs=20,prompt='',structure=.65):
    global _thread
    with _guard:
        if _thread and _thread.is_alive(): raise ValueError('A training run is already active.')
        records=load_records(); valid={int(r['objectid']) for r in records}
        excluded=list(dict.fromkeys(int(value) for value in excluded))
        if any(value not in valid for value in excluded): raise ValueError('Unknown artwork in exclusion list.')
        prompt=str(prompt).strip()[:80]
        minimum=1 if prompt else 8
        if len(valid)-len(excluded)<minimum:
            raise ValueError('The concept prompt does not match any artworks.' if prompt else 'At least eight artworks must remain.')
        if not 1<=epochs<=200: raise ValueError('Epochs must be between 1 and 200.')
        structure=float(structure)
        if not 0<=structure<=1: raise ValueError('Structure must be between 0 and 1.')
        run_id=datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
        _stop.clear()
        initial={'state':'queued','run_id':run_id,'epoch':0,'epochs':epochs,'batch':0,
                 'total_batches':len(valid)-len(excluded),'artworks':len(valid)-len(excluded),
                 'excluded':excluded,'prompt':prompt,'structure':structure,
                 'input':'512 x 512 x 3 grayscale','target':'512 x 512 x 3 RGB',
                 'message':'Training queued.','history':[]}
        _write(LIVE/'current.json',initial)
        _thread=threading.Thread(target=_run,args=(run_id,excluded,epochs,prompt,structure),daemon=True,name='pix2pix-live-training')
        _thread.start()
        return initial


def stop():
    if not _thread or not _thread.is_alive(): return False
    _stop.set();return True


def _pair(record):
    target=torch.from_numpy(square(IMAGES/record['localfile'],512)).permute(2,0,1).float()/127.5-1
    rgb=(target+1)/2
    gray=(.299*rgb[0]+.587*rgb[1]+.114*rgb[2])*2-1
    source=gray.unsqueeze(0).repeat(3,1,1)
    return source,target


def _rgb(tensor):
    return ((tensor[0].detach().cpu().permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)


def _run(run_id,excluded,epochs,prompt,structure):
    run=LIVE/run_id;frames=run/'frames';frames.mkdir(parents=True,exist_ok=True)
    state=json.loads((LIVE/'current.json').read_text(encoding='utf-8'))
    try:
        excluded_set=set(excluded)
        records=[r for r in load_records() if int(r['objectid']) not in excluded_set]
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type!='cuda': raise RuntimeError('CUDA is required for the 512px live preset.')
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        g,d=Generator512().to(device),Discriminator512().to(device)
        og=torch.optim.Adam(g.parameters(),lr=2e-4,betas=(.5,.999))
        od=torch.optim.Adam(d.parameters(),lr=1e-4,betas=(.5,.999))
        sg=torch.amp.GradScaler('cuda');sd=torch.amp.GradScaler('cuda');bce=nn.BCEWithLogitsLoss()
        state.update({'state':'preparing','device':str(device),'message':'Preparing 512px pair 0/'+str(len(records))})
        _write(LIVE/'current.json',state)
        total=np.zeros((512,512,3),dtype=np.float64)
        for index,record in enumerate(records,1):
            total+=square(IMAGES/record['localfile'],512)
            if index%25==0 or index==len(records):
                state.update({'prepared':index,'message':f'Preparing 512px pair {index}/{len(records)}'})
                _write(LIVE/'current.json',state)
        mean_rgb=(total/len(records)).astype(np.float32)
        mean_tensor=torch.from_numpy(mean_rgb).permute(2,0,1).unsqueeze(0)/127.5-1
        mean_unit=(mean_tensor+1)/2
        mean_gray=(.299*mean_unit[:,0]+.587*mean_unit[:,1]+.114*mean_unit[:,2])*2-1
        mean_source=mean_gray.unsqueeze(1).repeat(1,3,1,1)
        Image.fromarray(mean_rgb.round().astype(np.uint8)).save(run/'mean_target.png')
        work_path=OUT/'works.json'
        work_map={int(item['id']):item for item in json.loads(work_path.read_text(encoding='utf-8'))['works']}
        candidates=[(record,work_map.get(int(record['objectid']))) for record in records]
        candidates=[pair for pair in candidates if pair[1] is not None]
        coordinates=np.array([[item['x'],item['y'],item['z']] for _,item in candidates],dtype=np.float32)
        center=coordinates.mean(0)
        anchor_index=int(np.square(coordinates-center).sum(1).argmin())
        anchor_record,anchor_work=candidates[anchor_index]
        anchor_source,_=_pair(anchor_record)
        structured_source=(1-structure)*mean_source+structure*anchor_source.unsqueeze(0)
        structured_pixels=((structured_source[0].permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
        Image.fromarray(structured_pixels).save(run/'structured_input.png')
        Image.fromarray(square(IMAGES/anchor_record['localfile'],512)).save(run/'representative.png')
        state.update({'representative_id':int(anchor_record['objectid']),
                      'representative_title':anchor_record.get('title') or 'Untitled',
                      'mean_share':1-structure,'representative_share':structure})
        history=[];state.update({'state':'training','device':str(device),'message':'Training all 512px pairs.'})
        _write(LIVE/'current.json',state)
        def frame(epoch):
            was=g.training;g.eval()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                result=g(structured_source.to(device)).float().cpu()
            if was:g.train()
            Image.fromarray(_rgb(result)).save(frames/f'epoch_{epoch:03d}.png')
        frame(0)
        best=float('inf');best_epoch=0
        for epoch in range(1,epochs+1):
            if _stop.is_set(): break
            started=time.perf_counter();losses=[]
            order=torch.randperm(len(records),generator=torch.Generator().manual_seed(42+epoch)).tolist()
            for batch,index in enumerate(order,1):
                if _stop.is_set(): break
                a,b=_pair(records[index]);a=a.unsqueeze(0).to(device);b=b.unsqueeze(0).to(device)
                with torch.autocast('cuda',dtype=torch.float16):fake=g(a)
                od.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    real_score,fake_score=d(a,b),d(a,fake.detach())
                    ld=(bce(real_score,torch.full_like(real_score,.9))+bce(fake_score,torch.zeros_like(fake_score)))/2
                sd.scale(ld).backward();sd.step(od);sd.update()
                for parameter in d.parameters():parameter.requires_grad_(False)
                og.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    score=d(a,fake);reconstruction=F.l1_loss(fake,b);edges=edge_loss(fake,b)
                    lg=bce(score,torch.ones_like(score))+30*reconstruction+10*edges
                sg.scale(lg).backward();sg.step(og);sg.update()
                for parameter in d.parameters():parameter.requires_grad_(True)
                losses.append((lg.item(),ld.item(),reconstruction.item(),edges.item()))
                if batch%10==0 or batch==len(records):
                    state.update({'state':'training','epoch':epoch,'batch':batch,
                                  'progress':((epoch-1)*len(records)+batch)/(epochs*len(records)),
                                  'message':f'Epoch {epoch}/{epochs} · artwork {batch}/{len(records)}'})
                    _write(LIVE/'current.json',state)
            if _stop.is_set():break
            average=np.mean(losses,axis=0)
            row={'epoch':epoch,'g':float(average[0]),'d':float(average[1]),
                 'train_l1':float(average[2]),'edge':float(average[3]),
                 'seconds':time.perf_counter()-started}
            history.append(row);frame(epoch)
            if row['train_l1']<best:
                best=row['train_l1'];best_epoch=epoch;torch.save(g.state_dict(),run/'model_best.pt')
            torch.save(g.state_dict(),run/'model_latest.pt')
            state.update({'history':history,'best_epoch':best_epoch,'best_train_l1':best})
            _write(LIVE/'current.json',state)
        if _stop.is_set():
            state.update({'state':'stopped','message':'Training stopped by user.','history':history})
            _write(LIVE/'current.json',state);return
        state.update({'state':'generating','message':'Generating all artwork results.','history':history})
        _write(LIVE/'current.json',state)
        atlas_size=64;cols=32;rows=math.ceil(len(records)/cols)
        generated_dir=run/'generated';generated_dir.mkdir(exist_ok=True)
        generated=Image.new('RGB',(cols*atlas_size,rows*atlas_size))
        g.eval()
        for index,record in enumerate(records):
            a,_=_pair(record)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                result=g(a.unsqueeze(0).to(device)).float().cpu()
            full_image=Image.fromarray(_rgb(result))
            full_image.save(generated_dir/f"{record['objectid']}.jpg",quality=92,optimize=True)
            image=full_image.resize((atlas_size,atlas_size),Image.Resampling.LANCZOS)
            generated.paste(image,((index%cols)*atlas_size,(index//cols)*atlas_size))
            if index%25==0:
                state.update({'generated':index+1,'message':f'Generating result {index+1}/{len(records)}'})
                _write(LIVE/'current.json',state)
        generated.save(run/'generated_atlas.jpg',quality=84,optimize=True)
        (run/'records.json').write_text(json.dumps([{'id':r['objectid'],'title':r.get('title')} for r in records],ensure_ascii=False),encoding='utf-8')
        state.update({'state':'complete','epoch':epochs,'batch':len(records),'progress':1,
                      'generated':len(records),'message':'Training and result generation complete.',
                      'history':history,'completed_at':datetime.now(timezone.utc).isoformat()})
        _write(LIVE/'current.json',state)
    except Exception as error:
        state.update({'state':'error','message':f'{type(error).__name__}: {error}'})
        _write(LIVE/'current.json',state)
