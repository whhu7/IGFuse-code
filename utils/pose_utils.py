import json
import os
from glob import glob

import torch


def load_position_data_from_path(position_path):
    with open(position_path, 'r') as f:
        return json.load(f)


def load_position_data_from_dir(path):
    return load_position_data_from_path(os.path.join(path, "objects_poses.json"))


def get_scene_source_path(dataset, state_id):
    return dataset.source_paths[state_id]


def load_state_position_data(dataset, state_id):
    return load_position_data_from_dir(get_scene_source_path(dataset, state_id))


def load_pseudo_position_paths(dataset):
    if not dataset.pseudo_position_dir:
        return []
    return sorted(glob(os.path.join(dataset.pseudo_position_dir, "objects_poses*.json")))


def calculate_transform_matrix(source_matrix, target_matrix):
    """Compute rigid transform that maps source object pose to target object pose."""
    inv_source = torch.linalg.inv(source_matrix)
    return target_matrix @ inv_source


def build_transform_list(source_data, target_data):
    """Build per-object rigid transforms from source to target layout.

    Iteration order follows source_data.keys(); mask index j must match object j+1.
    All states should use the same object key order in objects_poses.json.
    """
    transform_list = []
    for key in source_data.keys():
        if key not in target_data:
            raise KeyError(f"Object key {key} not found in target state")
        source_matrix = torch.tensor(source_data[key]['world_matrix'], dtype=torch.float, device='cuda')
        target_matrix = torch.tensor(target_data[key]['world_matrix'], dtype=torch.float, device='cuda')
        transform_matrix = calculate_transform_matrix(source_matrix, target_matrix)
        transform_list.append({
            "translation": transform_matrix[:3, 3],
            "rotation": transform_matrix[:3, :3],
            "scaling": 1,
        })
    return transform_list
