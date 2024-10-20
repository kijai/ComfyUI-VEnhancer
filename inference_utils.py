import torch
from PIL import Image
from typing import Mapping
from einops import rearrange
import numpy as np
import torchvision.transforms.functional as transforms_F

def tensor2vid(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    video = video * 255.0
    images = rearrange(video, 'b c f h w -> b f h w c')[0]
    return images


def preprocess(input_frames):
    out_frame_list = []
    for pointer in range(len(input_frames)):
        frame = input_frames[pointer]
        frame = frame[:, :, ::-1]
        frame = Image.fromarray(frame.astype('uint8')).convert('RGB')
        frame = transforms_F.to_tensor(frame)
        out_frame_list.append(frame)
    out_frames = torch.stack(out_frame_list, dim=0)
    out_frames.clamp_(0, 1)
    mean = out_frames.new_tensor([0.5, 0.5, 0.5]).view(-1)
    std = out_frames.new_tensor([0.5, 0.5, 0.5]).view(-1)
    out_frames.sub_(mean.view(1, -1, 1, 1)).div_(std.view(1, -1, 1, 1))
    return out_frames


def adjust_resolution(h, w, up_scale):
    if h * w * up_scale * up_scale < 720 * 1280 * 1.5:
        up_s = np.sqrt(720 * 1280 * 1.5 / (h * w))
        target_h = int(up_s * h // 2 * 2)
        target_w = int(up_s * w // 2 * 2)
    elif h * w * up_scale * up_scale > 1152 * 2048:
        up_s = np.sqrt(1152 * 2048 / (h * w))
        target_h = int(up_s * h // 2 * 2)
        target_w = int(up_s * w // 2 * 2)
    else:
        target_h = int(up_scale * h // 2 * 2)
        target_w = int(up_scale * w // 2 * 2)
    return (target_h, target_w)


def make_mask_cond(in_f_num, interp_f_num):
    mask_cond = []
    interp_cond = [-1 for _ in range(interp_f_num)]
    for i in range(in_f_num):
        mask_cond.append(i)
        if i != in_f_num - 1:
            mask_cond += interp_cond
    return mask_cond

def collate_fn(data, device):
    """Prepare the input just before the forward function.
    This method will move the tensors to the right device.
    Usually this method does not need to be overridden.

    Args:
        data: The data out of the dataloader.
        device: The device to move data to.

    Returns: The processed data.

    """
    from torch.utils.data.dataloader import default_collate

    def get_class_name(obj):
        return obj.__class__.__name__

    if isinstance(data, dict) or isinstance(data, Mapping):
        return type(data)({
            k: collate_fn(v, device) if k != 'img_metas' else v
            for k, v in data.items()
        })
    elif isinstance(data, (tuple, list)):
        if 0 == len(data):
            return torch.Tensor([])
        if isinstance(data[0], (int, float)):
            return default_collate(data).to(device)
        else:
            return type(data)(collate_fn(v, device) for v in data)
    elif isinstance(data, np.ndarray):
        if data.dtype.type is np.str_:
            return data
        else:
            return collate_fn(torch.from_numpy(data), device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (bytes, str, int, float, bool, type(None))):
        return data
    else:
        raise ValueError(f'Unsupported data type {type(data)}')