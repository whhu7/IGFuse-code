import numpy as np
import open3d as o3d
import torch


def save_pcd(points, colors, path):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, point_cloud)


def points_inside_bbox(point_cloud, mask, expand=0.8, distance_threshold=0.02):
    """
    Given a point cloud and a mask indicating a subset of points, this function:
    1. Removes outliers from masked points
    2. Builds an expanded bbox (only expands downward in z direction)
    3. Gets points inside bbox but outside mask (ground part)
    4. Fits a ground plane using RANSAC and filters points by distance to plane
    """
    masked_points = point_cloud[mask].cpu().numpy()
    if masked_points.shape[0] == 0:
        return torch.zeros_like(mask, dtype=torch.bool)
    Q1 = np.percentile(masked_points, 25, axis=0)
    Q3 = np.percentile(masked_points, 75, axis=0)
    IQR = Q3 - Q1
    outlier_mask = (masked_points < (Q1 - IQR)) | (masked_points > (Q3 + IQR))
    filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    if filtered_masked_points.shape[0] == 0:
        filtered_masked_points = masked_points

    min_xyz = filtered_masked_points.min(axis=0)
    max_xyz = filtered_masked_points.max(axis=0)
    bbox_size = max_xyz - min_xyz
    min_xyz[2] = min_xyz[2] - expand * bbox_size[2]

    pc_np = point_cloud.cpu().numpy()
    bbox_mask = np.all((pc_np >= min_xyz) & (pc_np <= max_xyz), axis=1)
    bbox_mask = torch.tensor(bbox_mask, device=point_cloud.device)

    ground_mask = bbox_mask & (~mask)

    ground_points = point_cloud[ground_mask].cpu().numpy()
    if ground_points.shape[0] < 3:
        return torch.zeros_like(mask, dtype=torch.bool)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ground_points)

    _, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=100,
    )
    ground_indices = np.array(inliers)

    idx_ground = torch.where(ground_mask)[0]
    final_ground_mask = torch.zeros_like(mask, dtype=torch.bool)
    if len(idx_ground) > 0:
        final_ground_mask[idx_ground[ground_indices]] = True

    return final_ground_mask
