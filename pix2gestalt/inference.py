import gc
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision import transforms
from contextlib import nullcontext
import sys
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

def load_model_from_config(config, ckpt, device, verbose=False):
    global closure_device
    closure_device = device

    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=device)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def sample_model(input_im, visible_mask, model, sampler, precision, h, w, ddim_steps, n_samples, scale, \
                 ddim_eta):
    precision_scope = autocast if precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            
            cond = {}   
            """
            Cond Channel 1: CLIP(input_im)
            """
            clip_emb = model.get_learned_conditioning(input_im).tile(n_samples,1,1)
            c = model.cc_projection(clip_emb)
            cond['c_crossattn'] = [c]

            """
            Cond Channel 2: VAE(input_im) + VAE(visible_mask)
            """
            input_im_encoding = model.encode_first_stage((input_im.to(c.device))).mode().detach()
            visible_mask_encoding = model.encode_first_stage((visible_mask.to(c.device))).mode().detach()

            c_concat = torch.cat((input_im_encoding, visible_mask_encoding), dim = 1)
            cond['c_concat'] = [c_concat.repeat(n_samples, 1, 1, 1)]
     
            if scale != 1.0:
                uc = {}
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
                uc['c_concat'] = [torch.zeros(n_samples, 8, h // 8, w // 8).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
            
sam_models = {
  'vit_b': './ckpt/sam_vit_b.pth',
  'vit_l': './ckpt/sam_vit_l.pth',
  'vit_h': './ckpt/sam_vit_h.pth'
}

def get_sam_predictor(model_type='vit_h', device=None, image=None):

  sam = sam_model_registry[model_type](checkpoint=sam_models[model_type])
  sam = sam.to(device)

  predictor = SamPredictor(sam)
  if image is not None:
    predictor.set_image(image)
  return predictor


def run_sam(predictor: SamPredictor, selected_points):

  if len(selected_points) == 0:
    return []
  input_points = [p for p, _ in selected_points]
  input_labels = [int(l) for _, l in selected_points]

  masks, _, _ = predictor.predict(
                    point_coords = np.array(input_points),
                    point_labels = input_labels,
                    multimask_output = False, # single object output
  )
  visible_mask = 255 * np.squeeze(masks).astype(np.uint8) # (256, 256)
  overlay_mask = [(masks,'visible_mask')]
  
  return visible_mask, overlay_mask


def run_inference(input_image, 
                  visible_mask, 
                  model, 
                  guidance_scale, 
                  n_samples, 
                  ddim_steps):
  rgb_visible_mask = np.zeros((visible_mask.shape[0], visible_mask.shape[1], 3))
  rgb_visible_mask[:,:,0] = visible_mask
  rgb_visible_mask[:,:,1] = visible_mask
  rgb_visible_mask[:,:,2] = visible_mask # (256, 256, 3)

  pred_reconstructions = run_pix2gestalt(model, closure_device, input_image, rgb_visible_mask,
                                          scale = guidance_scale, 
                                          n_samples = n_samples, 
                                          ddim_steps = ddim_steps)
  gc.collect()
  torch.cuda.empty_cache()

  return pred_reconstructions 

def run_pix2gestalt(
        model,
        device,
        input_im,
        visible_mask,
        scale=2.0,
        n_samples=6,
        ddim_steps=200,
        ddim_eta=1.0,
        precision="fp32",
        h=256,
        w=256,
    ):    
    # input_im: (256, 256, 3)
    input_im = process_input(input_im).to(device)

    # visible_mask: (256, 256, 3), see run_inference above for an example.
    visible_mask = process_input(visible_mask).to(device)
    sampler = DDIMSampler(model)

    x_samples_ddim = sample_model(input_im, visible_mask, model, sampler, precision, h, w,\
                                  ddim_steps, n_samples, scale, ddim_eta)
    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(x_sample.astype(np.uint8))
    return output_ims


def process_input(input_im):
    normalized_image = torch.from_numpy(input_im).float().permute(2, 0, 1) / 255. # [0, 255] to [0, 1]
    normalized_image = normalized_image * 2 - 1 # [0, 1] to [-1, 1]
    return normalized_image.unsqueeze(0)
