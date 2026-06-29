"""Co-prune and cross-state Gaussian copy during the fusion phase."""

import numpy as np
import open3d as o3d
import torch

from utils.pointcloud_utils import points_inside_bbox


def run_coprune_and_copy(
    GsDict,
    SceneDict,
    opt,
    iteration,
    fusion_start_iteration,
    coprune_pair,
    obj_ids,
    transform_cache,
    build_mask3d_list,
    get_pred_obj,
):
    target_state_for = {
        coprune_pair[0]: coprune_pair[1],
        coprune_pair[1]: coprune_pair[0],
    }

    for i in coprune_pair:
        target_i = target_state_for[i]
        GsDict[f"points{i}"] = GsDict[f"gs{i}"].get_xyz.clone().detach().cpu().numpy()
        mask3d_list = build_mask3d_list(i)
        GsDict[f"gs{i}"].transform_setup(opt, mask3d_list, transform_cache[(i, target_i)])
        if iteration == fusion_start_iteration:
            pred_obj = get_pred_obj(i)
            for obj_id in obj_ids:
                mask3d = (pred_obj == obj_id + 1)
                # Ground Gaussians under each object bbox become copy candidates.
                mask3d_bbox = points_inside_bbox(GsDict[f"gs{i}"].get_trans_xyz.detach(), mask3d)
                mask3d_bbox = mask3d_bbox.float()[:, None, None]
                GsDict[f"mask_copy_gs{i}_obj{(obj_id + 1)}"] = mask3d_bbox.squeeze(-1).bool()

    for i in coprune_pair:
        target_i = target_state_for[i]
        trans_matrix = np.identity(4)
        threshold = opt.coprune_threshold

        source_cloud = o3d.geometry.PointCloud()
        source_cloud.points = o3d.utility.Vector3dVector(GsDict[f"gs{i}"].get_trans_xyz.clone().cpu().numpy())
        target_cloud = o3d.geometry.PointCloud()
        target_cloud.points = o3d.utility.Vector3dVector(GsDict[f"points{target_i}"])

        evaluation = o3d.pipelines.registration.evaluate_registration(
            source_cloud, target_cloud, threshold, trans_matrix
        )
        correspondence = np.array(evaluation.correspondence_set)
        mask_consistent = torch.zeros((GsDict[f"gs{i}"].get_trans_xyz.shape[0], 1)).cuda()
        if correspondence.size > 0:
            mask_consistent[correspondence[:, 0], :] = 1
        GsDict[f"mask_inconsistent_gs{i}"] = ~(mask_consistent.bool())

    if iteration == fusion_start_iteration:
        for i in coprune_pair:
            for obj_id in obj_ids:
                copy_mask = GsDict[f"mask_copy_gs{i}_obj{(obj_id + 1)}"]
                if copy_mask.sum() == 0:
                    continue
                copy_indices = torch.where(copy_mask)[0]
                GsDict[f"copy_gs{i}_obj{(obj_id + 1)}_xyz"] = GsDict[f"gs{i}"]._trans_xyz[copy_indices].clone().detach()
                GsDict[f"copy_gs{i}_obj{(obj_id + 1)}_features_dc"] = GsDict[f"gs{i}"]._features_dc[copy_indices].clone().detach()
                GsDict[f"copy_gs{i}_obj{(obj_id + 1)}_features_rest"] = GsDict[f"gs{i}"]._features_rest[copy_indices].clone().detach()
                GsDict[f"copy_gs{i}_obj{(obj_id + 1)}_opacity"] = GsDict[f"gs{i}"]._opacity[copy_indices].clone().detach()
                GsDict[f"copy_gs{i}_obj{(obj_id + 1)}_scaling"] = GsDict[f"gs{i}"]._trans_scaling[copy_indices].clone().detach()
                GsDict[f"copy_gs{i}_obj{(obj_id + 1)}_rotation"] = GsDict[f"gs{i}"]._trans_rotation[copy_indices].clone().detach()
                GsDict[f"copy_gs{i}_obj{(obj_id + 1)}_objects_dc"] = GsDict[f"gs{i}"]._objects_dc[copy_indices].clone().detach()
                # Do not co-prune points reserved for copying into the paired state.
                GsDict[f"mask_inconsistent_gs{i}"] = torch.logical_and(GsDict[f"mask_inconsistent_gs{i}"], ~copy_mask)

    for i in coprune_pair:
        GsDict[f"gs{i}"].prune_points(GsDict[f"mask_inconsistent_gs{i}"].squeeze())

    if iteration == fusion_start_iteration:
        for i in coprune_pair:
            source_idx = i
            target_idx = target_state_for[i]
            for obj_id in obj_ids:
                copy_mask = GsDict[f"mask_copy_gs{source_idx}_obj{(obj_id + 1)}"].squeeze()
                if copy_mask.sum() > 0:
                    GsDict[f"gs{target_idx}"].densification_postfix(
                        GsDict[f"copy_gs{source_idx}_obj{(obj_id + 1)}_xyz"],
                        GsDict[f"copy_gs{source_idx}_obj{(obj_id + 1)}_features_dc"],
                        GsDict[f"copy_gs{source_idx}_obj{(obj_id + 1)}_features_rest"],
                        GsDict[f"copy_gs{source_idx}_obj{(obj_id + 1)}_opacity"],
                        GsDict[f"copy_gs{source_idx}_obj{(obj_id + 1)}_scaling"],
                        GsDict[f"copy_gs{source_idx}_obj{(obj_id + 1)}_rotation"],
                        GsDict[f"copy_gs{source_idx}_obj{(obj_id + 1)}_objects_dc"],
                    )
            SceneDict[f"scene{i}"].save(iteration)
