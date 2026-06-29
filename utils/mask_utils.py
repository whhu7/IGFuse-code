import colorsys

import numpy as np
import torch


def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)
    s = 0.5 + (id % 2) * 0.5
    l = 0.5

    rgb = np.zeros((3, ), dtype=np.uint8)
    if id == 0:
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r * 255), int(g * 255), int(b * 255)

    return rgb


def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask


def get_mask(gaussians, classifier, obj_id):
    logits3d = classifier(gaussians._objects_dc.permute(2, 0, 1))
    pred_obj = torch.argmax(logits3d, dim=0)
    pred_obj = pred_obj.squeeze(-1).cpu().numpy().astype(np.uint8)
    mask = (pred_obj == obj_id)
    mask3d = torch.tensor(mask, device='cuda')
    mask3d = mask3d.float()[:, None, None]
    return mask3d


def get_mask3d_list(gaussians, classifier, obj_ids):
    logits3d = classifier(gaussians._objects_dc.permute(2, 0, 1))
    pred_obj = torch.argmax(logits3d, dim=0).squeeze(-1)
    mask3d_list = []
    for obj_id in obj_ids:
        mask3d = (pred_obj == obj_id + 1).float()[:, None, None]
        mask3d_list.append(mask3d)
    return mask3d_list, pred_obj
