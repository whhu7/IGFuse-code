#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
import os
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def rotation_matrix_to_quaternion(R):
    q = torch.zeros((R.shape[0], 4), dtype=R.dtype, device=R.device)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    eps = 1e-8

    trace_mask = trace > 0
    if trace_mask.any():
        s = torch.sqrt(torch.clamp_min(trace[trace_mask] + 1.0, eps)) * 2.0
        q[trace_mask, 0] = 0.25 * s
        q[trace_mask, 1] = (R[trace_mask, 2, 1] - R[trace_mask, 1, 2]) / s
        q[trace_mask, 2] = (R[trace_mask, 0, 2] - R[trace_mask, 2, 0]) / s
        q[trace_mask, 3] = (R[trace_mask, 1, 0] - R[trace_mask, 0, 1]) / s

    x_mask = (~trace_mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if x_mask.any():
        s = torch.sqrt(torch.clamp_min(1.0 + R[x_mask, 0, 0] - R[x_mask, 1, 1] - R[x_mask, 2, 2], eps)) * 2.0
        q[x_mask, 0] = (R[x_mask, 2, 1] - R[x_mask, 1, 2]) / s
        q[x_mask, 1] = 0.25 * s
        q[x_mask, 2] = (R[x_mask, 0, 1] + R[x_mask, 1, 0]) / s
        q[x_mask, 3] = (R[x_mask, 0, 2] + R[x_mask, 2, 0]) / s

    y_mask = (~trace_mask) & (~x_mask) & (R[:, 1, 1] > R[:, 2, 2])
    if y_mask.any():
        s = torch.sqrt(torch.clamp_min(1.0 + R[y_mask, 1, 1] - R[y_mask, 0, 0] - R[y_mask, 2, 2], eps)) * 2.0
        q[y_mask, 0] = (R[y_mask, 0, 2] - R[y_mask, 2, 0]) / s
        q[y_mask, 1] = (R[y_mask, 0, 1] + R[y_mask, 1, 0]) / s
        q[y_mask, 2] = 0.25 * s
        q[y_mask, 3] = (R[y_mask, 1, 2] + R[y_mask, 2, 1]) / s

    z_mask = (~trace_mask) & (~x_mask) & (~y_mask)
    if z_mask.any():
        s = torch.sqrt(torch.clamp_min(1.0 + R[z_mask, 2, 2] - R[z_mask, 0, 0] - R[z_mask, 1, 1], eps)) * 2.0
        q[z_mask, 0] = (R[z_mask, 1, 0] - R[z_mask, 0, 1]) / s
        q[z_mask, 1] = (R[z_mask, 0, 2] + R[z_mask, 2, 0]) / s
        q[z_mask, 2] = (R[z_mask, 1, 2] + R[z_mask, 2, 1]) / s
        q[z_mask, 3] = 0.25 * s

    return torch.nn.functional.normalize(torch.nan_to_num(q), dim=-1)


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_scaling_R(s, R):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(torch.device("cuda:0"))
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True