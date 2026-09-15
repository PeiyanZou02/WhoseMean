"""Local coherent-mean experiment: NGA subset -> SD 1.5 LoRA -> mean-conditioned image."""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
NGA=ROOT/'data'/'nga'
OUT=ROOT/'outputs'/'diffusion'
DATA=ROOT/'data'/'diffusion'
BASE_MODEL='stable-diffusion-v1-5/stable-diffusion-v1-5'


def records():
    path=NGA/'records.jsonl'
    if not path.exists(): raise SystemExit('Run the NGA collection pipeline first.')
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def square(path,size=512):
    with Image.open(path) as source:
        source=ImageOps.exif_transpose(source).convert('RGB')
        fitted=ImageOps.contain(source,(size,size),Image.Resampling.LANCZOS)
        canvas=Image.new('RGB',(size,size),(238,236,230))
        canvas.paste(fitted,((size-fitted.width)//2,(size-fitted.height)//2))
        return canvas


def caption(record):
    fields=['an artwork from the National Gallery of Art open collection']
    classification=record.get('classification')
    if classification: fields.append(classification.lower())
    medium=record.get('medium')
    if medium: fields.append(str(medium).lower())
    title=record.get('title')
    if title: fields.append(f'titled {title}')
    date=record.get('displaydate')
    if date: fields.append(f'created {date}')
    return ', '.join(fields)


def prepare(classification='Painting'):
    target=DATA/classification.lower(); target.mkdir(parents=True,exist_ok=True)
    rows=[]; total=np.zeros((512,512,3),dtype=np.float64)
    selected=[r for r in records() if r.get('classification')==classification]
    for index,record in enumerate(selected):
        source=NGA/'images'/record['localfile']
        if not source.exists(): continue
        image=square(source); name=f"{int(record['objectid']):07d}.jpg"
        image.save(target/name,quality=94)
        total += np.asarray(image,dtype=np.float64)
        rows.append({'file_name':name,'text':caption(record),'objectid':record['objectid'],
                     'title':record.get('title'),'artist':record.get('attribution')})
    if len(rows)<8: raise SystemExit(f'Too few {classification} images: {len(rows)}')
    mean=Image.fromarray((total/len(rows)).round().astype(np.uint8))
    mean.save(target/'pixel_mean_512.png')
    with (target/'metadata.jsonl').open('w',encoding='utf-8') as output:
        for row in rows: output.write(json.dumps(row,ensure_ascii=False)+'\n')
    report={'classification':classification,'images':len(rows),'resolution':512,
            'base_model':BASE_MODEL,'mean':'exact per-pixel mean of this subset',
            'method':'LoRA learns the subset domain; img2img begins from the exact subset mean'}
    (target/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


class MuseumDataset:
    def __init__(self,folder,tokenizer):
        self.folder=folder; self.tokenizer=tokenizer
        self.rows=[json.loads(line) for line in (folder/'metadata.jsonl').read_text(encoding='utf-8').splitlines()]
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        image=square(self.folder/row['file_name'])
        if random.random()<.5: image=ImageOps.mirror(image)
        pixels=np.asarray(image,dtype=np.float32)/127.5-1
        token=self.tokenizer(row['text'],padding='max_length',truncation=True,
                             max_length=self.tokenizer.model_max_length,return_tensors='pt').input_ids[0]
        return torch.from_numpy(pixels).permute(2,0,1),token


def train_lora(classification='Painting',steps=1200,rank=8,lr=1e-4):
    try:
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
        from diffusers.utils import convert_state_dict_to_diffusers
        from diffusers.training_utils import cast_training_params
        from transformers import CLIPTextModel, CLIPTokenizer
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict
    except ImportError as error:
        raise SystemExit('Install requirements-diffusion.txt first.') from error
    if not torch.cuda.is_available(): raise SystemExit('CUDA GPU is required for this training preset.')
    folder=DATA/classification.lower()
    if not (folder/'metadata.jsonl').exists(): prepare(classification)
    output=OUT/classification.lower(); output.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda'); dtype=torch.bfloat16
    tokenizer=CLIPTokenizer.from_pretrained(BASE_MODEL,subfolder='tokenizer')
    scheduler=DDPMScheduler.from_pretrained(BASE_MODEL,subfolder='scheduler')
    text_encoder=CLIPTextModel.from_pretrained(BASE_MODEL,subfolder='text_encoder',torch_dtype=dtype).to(device).eval()
    vae=AutoencoderKL.from_pretrained(BASE_MODEL,subfolder='vae',torch_dtype=dtype).to(device).eval()
    unet=UNet2DConditionModel.from_pretrained(BASE_MODEL,subfolder='unet',torch_dtype=dtype).to(device)
    vae.requires_grad_(False); text_encoder.requires_grad_(False); unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=rank,lora_alpha=rank,init_lora_weights='gaussian',
                                target_modules=['to_k','to_q','to_v','to_out.0']))
    # Keep the frozen base network compact while optimizing LoRA weights in FP32.
    cast_training_params([unet],dtype=torch.float32)
    trainable=[p for p in unet.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=lr)
    dataset=MuseumDataset(folder,tokenizer)
    generator=torch.Generator().manual_seed(42)
    history=[]; unet.train(); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    for step in range(1,steps+1):
        image,tokens=dataset[int(torch.randint(len(dataset),(1,),generator=generator))]
        image=image.unsqueeze(0).to(device,dtype=dtype); tokens=tokens.unsqueeze(0).to(device)
        with torch.no_grad():
            latents=vae.encode(image).latent_dist.sample()*vae.config.scaling_factor
            text=text_encoder(tokens)[0]
        noise=torch.randn_like(latents); timesteps=torch.randint(0,scheduler.config.num_train_timesteps,(1,),device=device)
        noisy=scheduler.add_noise(latents,noise,timesteps)
        prediction=unet(noisy,timesteps,text).sample
        target=noise if scheduler.config.prediction_type=='epsilon' else scheduler.get_velocity(latents,noise,timesteps)
        loss=F.mse_loss(prediction.float(),target.float())
        if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss at step {step}')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable,1.0); optimizer.step()
        history.append({'step':step,'loss':float(loss.item())})
        if step%50==0: print(json.dumps(history[-1]),flush=True)
        if step%200==0 or step==steps:
            state=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
            checkpoint=output/f'checkpoint-{step:05d}'
            StableDiffusionPipeline.save_lora_weights(checkpoint,unet_lora_layers=state)
            (output/'history.json').write_text(json.dumps(history),encoding='utf-8')
    state=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(output,unet_lora_layers=state)
    print(json.dumps({'status':'trained','steps':steps,'rank':rank,'output':str(output)}))


def generate(classification='Painting',strength=.62,steps=35,seed=42):
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
    except ImportError as error:
        raise SystemExit('Install requirements-diffusion.txt first.') from error
    folder=DATA/classification.lower(); lora=OUT/classification.lower()
    if not (lora/'pytorch_lora_weights.safetensors').exists(): raise SystemExit('Train the LoRA first.')
    pipe=StableDiffusionImg2ImgPipeline.from_pretrained(
        BASE_MODEL,torch_dtype=torch.float16,safety_checker=None,requires_safety_checker=False)
    pipe.load_lora_weights(lora); pipe.enable_attention_slicing(); pipe=pipe.to('cuda')
    prompt=('a coherent singular museum painting from the National Gallery of Art collection, '
            'one readable composition, visible subject, painterly surface, restrained museum colors')
    negative=('collage, contact sheet, tiled pattern, repeated image, multiple frames, blur, haze, '
              'text, watermark, border')
    initial=Image.open(folder/'pixel_mean_512.png').convert('RGB')
    result=pipe(prompt=prompt,negative_prompt=negative,image=initial,strength=strength,
                guidance_scale=7.0,num_inference_steps=steps,
                generator=torch.Generator(device='cuda').manual_seed(seed)).images[0]
    output=OUT/classification.lower(); output.mkdir(parents=True,exist_ok=True)
    path=output/f'coherent_mean_seed_{seed}.png'; result.save(path)
    metadata={'classification':classification,'seed':seed,'strength':strength,'steps':steps,
              'prompt':prompt,'negative_prompt':negative,'path':str(path)}
    (output/'generation.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare'); prep.add_argument('--classification',default='Painting')
    train=sub.add_parser('train-lora'); train.add_argument('--classification',default='Painting')
    train.add_argument('--steps',type=int,default=1200); train.add_argument('--rank',type=int,default=8)
    train.add_argument('--learning-rate',type=float,default=1e-4)
    gen=sub.add_parser('generate'); gen.add_argument('--classification',default='Painting')
    gen.add_argument('--strength',type=float,default=.62); gen.add_argument('--steps',type=int,default=35)
    gen.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.classification)
    elif args.command=='train-lora': train_lora(args.classification,args.steps,args.rank,args.learning_rate)
    else: generate(args.classification,args.strength,args.steps,args.seed)
