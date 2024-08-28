from typing import Any, Dict
from einops import rearrange

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F

from .modules.embedder import FrozenOpenCLIPEmbedder

#from .modules import *
from .utils.config import cfg
from .diffusion.diffusion_sdedit import GaussianDiffusion
from .diffusion.schedules_sdedit import noise_schedule

from diffusers import AutoencoderKLTemporalDecoder

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoToVideo():
    def __init__(self, generator, device):
        self.device = device
        clip_encoder = FrozenOpenCLIPEmbedder(device=self.device, pretrained='laion2b_s32b_b79k')
        clip_encoder.model.to(self.device)
        self.clip_encoder = clip_encoder
        logger.info(f'Build encoder with {cfg.embedder.type}')

        # generator = ControlledV2VUNet()
        # generator = generator.to(self.device)
        # generator.eval()

        # cfg.model_path = opt.model_path
        # load_dict = torch.load(cfg.model_path, map_location='cpu')
        # if 'state_dict' in load_dict:
        #     load_dict = load_dict['state_dict']
        # ret = generator.load_state_dict(load_dict, strict=True)
        
        # self.generator = generator.half()
        # logger.info('Load model path {}, with local status {}'.format(cfg.model_path, ret))
        self.generator = generator
        sigmas = noise_schedule(
            schedule='logsnr_cosine_interp',
            n=1000,
            zero_terminal_snr=True,
            scale_min=2.0,
            scale_max=4.0)
        diffusion = GaussianDiffusion(sigmas=sigmas)
        self.diffusion = diffusion
        logger.info('Build diffusion with GaussianDiffusion')

        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            'stabilityai/stable-video-diffusion-img2vid', subfolder="vae", variant="fp16")
        vae.eval()
        vae.requires_grad_(False)
        vae.to(self.device)
        self.vae = vae

        torch.cuda.empty_cache()

        self.negative_prompt = cfg.negative_prompt
        self.positive_prompt = cfg.positive_prompt

        negative_y = clip_encoder(self.negative_prompt).detach()
        self.negative_y = negative_y


    def test(self, input: Dict[str, Any], total_noise_levels=1000, \
                 steps=50, solver_mode='fast', guide_scale=7.5, noise_aug=200):
        video_data = input['video_data']
        y = input['y']
        mask_cond = input['mask_cond']
        s_cond = input['s_cond']
        interp_f_num = input['interp_f_num']
        (target_h, target_w) = input['target_res']

        video_data = F.interpolate(video_data, [target_h,target_w], mode='bilinear')

        key_f_num = len(video_data)
        aug_video = []
        for i in range(key_f_num):
            if i == key_f_num - 1:
                aug_video.append(video_data[i:i+1])
            else:
                aug_video.append(video_data[i:i+1].repeat(interp_f_num+1, 1, 1, 1))
        video_data = torch.concat(aug_video, dim=0)

        logger.info(f'video_data shape: {video_data.shape}')
        frames_num, _, h, w = video_data.shape

        padding = pad_to_fit(h, w)
        video_data = F.pad(video_data, padding, 'constant', 1)

        video_data = video_data.unsqueeze(0)
        bs = 1
        video_data = video_data.to(self.device)

        mask_cond = mask_cond.unsqueeze(0).to(self.device)
        s_cond = torch.LongTensor([s_cond]).to(self.device)

        video_data_feature = tensor2latent(video_data, self.vae)
        print("video_data_feature", video_data_feature.shape)

        torch.cuda.empty_cache()

        y = self.clip_encoder(y).detach() 

        with amp.autocast(enabled=True):

            t_hint = torch.LongTensor([noise_aug-1]).to(self.device)
            video_in_low_fps = video_data_feature[:,:,::interp_f_num+1].clone()
            noised_hint = self.diffusion.diffuse(video_in_low_fps, t_hint)

            t = torch.LongTensor([total_noise_levels-1]).to(self.device)
            noised_lr = self.diffusion.diffuse(video_data_feature, t)
            
            model_kwargs = [{'y': y}, {'y': self.negative_y}]
            model_kwargs.append({'hint': noised_hint})
            model_kwargs.append({'mask_cond': mask_cond})
            model_kwargs.append({'s_cond': s_cond})
            model_kwargs.append({'t_hint': t_hint})

            torch.cuda.empty_cache()
            chunk_inds = make_chunks(frames_num, interp_f_num) if frames_num > 32 else None

            solver = 'dpmpp_2m_sde' # 'heun' | 'dpmpp_2m_sde' 
            gen_vid = self.diffusion.sample(
                noise=noised_lr,
                model=self.generator,
                model_kwargs=model_kwargs,
                guide_scale=guide_scale,
                guide_rescale=0.2,
                solver=solver,
                solver_mode=solver_mode,
                steps=steps,
                t_max=total_noise_levels - 1,
                t_min=0,
                discretization='trailing',
                chunk_inds=chunk_inds)
            torch.cuda.empty_cache()
            
            return gen_vid, padding
            vid_tensor_gen = latent2tensor(gen_vid, self.vae)

        w1, w2, h1, h2 = padding
        vid_tensor_gen = vid_tensor_gen[:,:,h1:h+h1,w1:w+w1]

        gen_video = rearrange(
            vid_tensor_gen, '(b f) c h w -> b c f h w', b=bs)

        logger.info(f'sampling, finished.')
        torch.cuda.empty_cache()
        
        return gen_video.type(torch.float32).cpu()


def pad_to_fit(h, w):
    BEST_H, BEST_W = 720, 1280

    if h < BEST_H:
        h1, h2 = _create_pad(h, BEST_H)
    elif h == BEST_H:
        h1 = h2 = 0
    else: 
        h1 = 0
        h2 = int((h + 48) // 64 * 64) + 64 - 48 - h

    if w < BEST_W:
        w1, w2 = _create_pad(w, BEST_W)
    elif w == BEST_W:
        w1 = w2 = 0
    else:
        w1 = 0
        w2 = int(w // 64 * 64) + 64 - w
    return (w1, w2, h1, h2)

def _create_pad(h, max_len):
    h1 = int((max_len - h) // 2)
    h2 = max_len - h1 - h
    return h1, h2


def make_chunks(f_num, interp_f_num):
    MAX_CHUNK_LEN, MAX_O_LEN, MIN_S_NUM = 24, 12, 8
    chunk_len = (MAX_CHUNK_LEN-1)//(1+interp_f_num)*(interp_f_num+1)+1
    o_len = (MAX_O_LEN-1)//(1+interp_f_num)*(interp_f_num+1)+1
    ind = 0
    chunk_inds = []
    while ind<f_num:
        if ind+chunk_len+MIN_S_NUM>=f_num:
            chunk_inds.append(list(range(ind,f_num)))
            break
        else:
            chunk_inds.append(list(range(ind,ind+chunk_len)))
            ind += chunk_len-o_len  
    return chunk_inds


def tensor2latent(t, vae):
    video_length = t.shape[1]
    t = rearrange(t, "b f c h w -> (b f) c h w")
    chunk_size = 1
    latents_list = []
    for ind in range(0,t.shape[0],chunk_size):
        latents_list.append(vae.encode(t[ind:ind+chunk_size]).latent_dist.sample())
    latents = torch.cat(latents_list, dim=0)
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * vae.config.scaling_factor
    return latents


def latent2tensor(latents, vae):
    video_length = latents.shape[2]
    latents = 1. / vae.config.scaling_factor * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    video = []
    chunk_size = 3
    for ind in range(0, latents.shape[0], chunk_size):
        num_frames = latents[ind:ind+chunk_size].shape[0]
        video.append(vae.decode(latents[ind:ind+chunk_size], num_frames=num_frames).sample)
    video = torch.cat(video)
    return video