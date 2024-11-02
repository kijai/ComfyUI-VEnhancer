import os
import torch
import gc

import folder_paths
import comfy.model_management as mm
import comfy.utils

from .video_to_video.video_to_video_model import VideoToVideo
from .video_to_video.utils.seed import setup_seed
from .inference_utils import collate_fn, make_mask_cond, adjust_resolution
from .video_to_video.modules.unet_v2v import ControlledV2VUNet

script_directory = os.path.dirname(os.path.abspath(__file__))

from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    is_accelerate_available = False
    pass

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class DownloadAndLoadVEnhancerModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (["venhancer_paper-fp16.safetensors", "venhancer_v2-fp16.safetensors"], {"default": "venhancer_v2-fp16.safetensors"}),
            "precision": (["fp16", "bf16", "fp8_e4m3fn"], {"default": "fp16"}),
            "torch_compile": ("BOOLEAN", {"default": False, "tooltip": "Enable to compile the model using torch.compile inductor backend."}),
            },
        }
    RETURN_TYPES = ("VENCHANCER_MODEL",)
    RETURN_NAMES = ("venhancer_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "VEnhancer"

    def loadmodel(self, model, precision, torch_compile):
        device = mm.get_torch_device()
        mm.soft_empty_cache()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp8_e4m3fn": torch.float8_e4m3fn}[precision]

        pbar = comfy.utils.ProgressBar(1)
        
        download_path = os.path.join(folder_paths.models_dir, "venhancer")
        model_path = os.path.join(download_path, model)
        
        if not os.path.exists(model_path):
            log.info(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="Kijai/VEnhancer-fp16",
                                allow_patterns=[model], 
                                local_dir=download_path, 
                                local_dir_use_symlinks=False)

        log.info(f"Loading model from: {model_path}")
        pbar.update(1)


        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            generator = ControlledV2VUNet()
        sd = comfy.utils.load_torch_file(model_path)
        if is_accelerate_available:
            for key in sd:
                set_module_tensor_to_device(generator, key, dtype=dtype, device=device, value=sd[key])
        else:
            generator.load_state_dict(sd, strict=False)
            generator.to(dtype)     
        
        leftover_keys = generator.load_state_dict(sd, strict=True)
        generator.to(device).eval()
        print("leftover_keys", leftover_keys)

        if torch_compile:
            generator = torch.compile(generator, dynamic=False, backend="inductor")
    
        self.model = VideoToVideo(generator, device)
        

        return (self.model,)
    
        
class VEnhancerSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "venhancer_model": ("VENCHANCER_MODEL",),
            "images": ("IMAGE",),
            "solver_mode": (['fast', 'normal'], {"default": 'fast', "tooltip": "fast is locked to 15 steps, normal is unlocked"}),
            "steps": ("INT", {"default": 15, "min": 1, "max": 100, "step": 1}),
            "guide_scale": ("FLOAT", {"default": 7.5, "min": 0, "max": 100, "step": 0.1}),
            "s_cond": ("INT", {"default": 8, "min": 0, "max": 100, "step": 1}),
            "up_scale": ("INT", {"default": 4, "min": 1, "max": 200, "step": 1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
            "input_fps": ("INT", {"default": 24, "min": 2, "max": 100, "step": 1}),
            "target_fps": ("INT", {"default": 24, "min": 2, "max": 100, "step": 1}),
            "noise_aug": ("INT", {"default": 300, "min": 0, "max": 1000, "step": 1}),
            "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": "Disable to offload model after processing."}),
            "prompt": ("STRING", {"default": "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing,  hyper sharpness, perfect without deformations.", "multiline": True}),
            "max_chunk_length": ("INT", {"default": 16, "min": 1, "max": 100, "step": 1, "tooltip": "Maximum number of frames in a chunk."}),
            "chunk_overlap_ratio": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.1, "tooltip": "Overlap ratio between chunks."}),
            },
        }

    RETURN_TYPES = ("LATENT", "PADDING")
    RETURN_NAMES = ("samples", "padding",)
    FUNCTION = "process"
    CATEGORY = "VEnhancer"

    def process(self, venhancer_model, images, steps, guide_scale, solver_mode, s_cond, seed, prompt, up_scale, input_fps, target_fps, 
                noise_aug, keep_model_loaded, max_chunk_length, chunk_overlap_ratio):
        
        B, H, W, C = images.shape
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.unload_all_models()
        mm.soft_empty_cache()

        steps = 15 if solver_mode == 'fast' else steps

        interp_f_num = max(round(target_fps/input_fps)-1, 0)
        interp_f_num = min(interp_f_num, 8)
        target_fps = input_fps * (interp_f_num+1)

        video_data = images.permute(0, 3, 1, 2) * 2.0 - 1.0
        
        target_h, target_w = adjust_resolution(H, W, up_scale)

        mask_cond = make_mask_cond(B, interp_f_num)
        mask_cond = torch.Tensor(mask_cond).long()

        noise_aug = min(max(noise_aug,0),300)

        pre_data = {'video_data': video_data, 'y': prompt}
        pre_data['mask_cond'] = mask_cond
        pre_data['s_cond'] = s_cond
        pre_data['interp_f_num'] = interp_f_num
        pre_data['target_res'] = (target_h, target_w)
        pre_data['t_hint'] = noise_aug

        #video_data_feature torch.Size([1, 4, 15, 130, 160])
        total_noise_levels = 900
        setup_seed(seed)
        venhancer_model.generator.to(device)
        data_tensor = collate_fn(pre_data, device)
        output, padding = venhancer_model.test(
            data_tensor, 
            total_noise_levels, 
            steps=steps, 
            solver_mode=solver_mode, 
            guide_scale=guide_scale, 
            noise_aug=noise_aug, 
            max_chunk_length=max_chunk_length, 
            chunk_overlap_ratio=chunk_overlap_ratio
            )
        
        if not keep_model_loaded:
            venhancer_model.generator.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        output = output[0].permute(1, 0, 2, 3) / 0.18215

        return {"samples": output}, padding,

class VEnhancerUnpad:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            "padding": ("PADDING",),
          },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "VEnhancer"

    def process(self, images, padding):
        
        B, H, W, C = images.shape
        w1, w2, h1, h2 = padding
        print(padding)
        print(images.shape)
        images = images[:, h1:H-h2, w1:W-w2, :]

        return images,



NODE_CLASS_MAPPINGS = {
    "VEnhancerSampler": VEnhancerSampler,
    "DownloadAndLoadVEnhancerModel": DownloadAndLoadVEnhancerModel,
    "VEnhancerUnpad": VEnhancerUnpad
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VEnhancerSampler": "VEnhancerSampler",
    "DownloadAndLoadVEnhancerModel": "DownloadAndLoadVEnhancerModel",
    "VEnhancerUnpad": "VEnhancerUnpad"
}
