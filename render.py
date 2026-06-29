# Copyright (c) 2026 IGFuse
#
# Modified from codes in Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, RenderParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import json
import copy
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.decomposition import PCA
from utils.dataset_utils import configure_render_dataset_package_args
from utils.mask_utils import get_mask, visualize_obj
from utils.pose_utils import build_transform_list, get_scene_source_path, load_position_data_from_dir as load_position_data
from utils.pointcloud_utils import save_pcd

def compute_iou(pred_mask, true_mask):
    """Intersection-over-union between two boolean masks."""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def mean_iou(pred_mask, gt_mask):
    """Mean IoU over non-background object classes."""
    ids_gt = np.unique(gt_mask)
    ids_gt = ids_gt[ids_gt != 0]
    
    if len(ids_gt) == 0:
        return 0.0
    
    miou = []
    for id_gt in ids_gt:
        pred = (pred_mask == id_gt)
        gt = (gt_mask == id_gt)
        iou = compute_iou(pred, gt)
        miou.append(iou)
    
    return np.mean(miou)

def feature_to_rgb(features):
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())
    pca_result = pca_result.reshape(H, W, 3)
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    return pca_normalized.astype('uint8')

def make_test_dataset_args(dataset, test_source_path):
    """Build a single-state dataset view for loading test cameras only."""
    test_dataset = copy.copy(dataset)
    test_dataset.source_paths = [os.path.abspath(test_source_path)]
    test_dataset.trained_paths = [dataset.model_path]
    return test_dataset

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, is_trans=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_feature16")
    gt_colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_objects_color")
    makedirs(gt_colormask_path, exist_ok=True)
    gt_object_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_objects")
    makedirs(gt_object_path, exist_ok=True)

    pred_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_pred")
    test_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_test")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    makedirs(test_obj_path, exist_ok=True)

    total_psnr = 0
    total_ssim = 0
    total_miou = 0
    num_images = len(views)
    num_miou_images = 0

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background, is_trans)

        rendering = results["render"]
        gt = view.original_image[0:3, :, :]
        render_np = rendering.cpu().numpy().transpose(1, 2, 0)
        gt_np = gt.cpu().numpy().transpose(1, 2, 0)

        psnr_val = psnr(gt_np, render_np, data_range=1.0)
        ssim_val = ssim(gt_np, render_np, data_range=1.0, channel_axis=2, multichannel=True)
        total_psnr += psnr_val
        total_ssim += ssim_val
        print(f"Image {view.image_name} - PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
       
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{}'.format(view.image_name) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{}'.format(view.image_name) + ".png"))

        rendering_obj = results["render_seg"]
        
        logits = classifier(rendering_obj)
        pred_obj = torch.argmax(logits,dim=0)
        pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
        rgb_mask = feature_to_rgb(rendering_obj)
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{}'.format(view.image_name) + ".png"))
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, '{}'.format(view.image_name) + ".png"))
        Image.fromarray(pred_obj.cpu().numpy().astype(np.uint8)).save(os.path.join(test_obj_path, '{}'.format(view.image_name) + ".png"))

        if view.objects != None:
            gt_objects = view.objects
            gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))
            Image.fromarray(gt_objects.cpu().numpy().astype(np.uint8)).save(os.path.join(gt_object_path, '{}'.format(view.image_name) + ".png"))
            Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, '{}'.format(view.image_name) + ".png"))

            pred_obj_np = pred_obj.cpu().numpy().astype(np.uint8)
            gt_objects_np = gt_objects.cpu().numpy().astype(np.uint8)
            current_miou = mean_iou(pred_obj_np, gt_objects_np)
            print(f"Image {view.image_name} - mIoU: {current_miou:.4f}")

            total_miou += current_miou
            num_miou_images += 1


    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    avg_miou = total_miou / num_miou_images if num_miou_images > 0 else None
    print(f"\nAverage metrics for {name} set:")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

    if avg_miou is not None:
        print(f"Average mIoU: {avg_miou:.4f}")
    else:
        print("No ground truth objects found for mIoU calculation")
    metrics = {
        "name": name,
        "num_images": num_images,
        "num_miou_images": num_miou_images,
        "psnr": float(avg_psnr),
        "ssim": float(avg_ssim),
        "miou": None if avg_miou is None else float(avg_miou),
    }
    metrics_path = os.path.join(model_path, name, "ours_{}".format(iteration), "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics

def render_all(model_path, name, iteration, views, gaussians, pipeline, background, classifier, transform_list):
        is_trans = transform_list is not None
        if is_trans:
            obj_ids = np.arange(len(transform_list))
            mask3d_list = [get_mask(gaussians, classifier, obj_id + 1) for obj_id in obj_ids]
            gaussians.transform_setup(None, mask3d_list, transform_list)
        gaussians.save_ply(os.path.join(model_path, name, "ours_{}".format(iteration),"gaussians.ply"))
        metrics = render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, is_trans)
        logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
        pred_obj = torch.argmax(logits3d,dim=0)
        pred_obj = pred_obj.squeeze(-1).cpu().numpy().astype(np.uint8)
        pred_xyz = gaussians.get_trans_xyz.cpu().numpy() if is_trans else gaussians.get_xyz.cpu().numpy()
        pred_vis_mask = visualize_obj(pred_obj)
        save_pcd(pred_xyz, pred_vis_mask/255, os.path.join(model_path, name, "ours_{}".format(iteration),"pred_obj.ply"))
        return metrics
      
def render_sets(dataset : ModelParams, pipeline : PipelineParams, render_params : RenderParams, train_state_ids, test_source_path, test_name, use_test_cameras):
    with torch.no_grad():
        if not train_state_ids:
            raise ValueError("--train_state_ids must contain at least one training state id")
        if not test_source_path:
            raise ValueError("--test_source_path is required")

        first_state_id = train_state_ids[0]
        first_source_path = get_scene_source_path(dataset, first_state_id)
        first_position_data = load_position_data(first_source_path)
        obj_ids = np.arange(len(first_position_data))
        num_classes = obj_ids.shape[0]+1

        test_dataset = make_test_dataset_args(dataset, test_source_path)
        # scene_id=0 indexes the single entry in test_dataset.source_paths.
        test_scene = Scene(test_dataset, None, shuffle=False, scene_id=0, load_cameras_only=True)
        target_views = test_scene.getTestCameras() if use_test_cameras else test_scene.getTrainCameras()
        target_position_data = load_position_data(test_source_path)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        summary = {}

        for source_state_id in train_state_ids:
            print(f"\nRendering source state {source_state_id} -> test state {test_name}")
            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=render_params.iteration, shuffle=False, scene_id=source_state_id)
            source_position_data = load_position_data(get_scene_source_path(dataset, source_state_id))
            transform_list = build_transform_list(source_position_data, target_position_data)

            classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
            classifier.cuda()
            classifier_path = os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(scene.loaded_iter), "classifier.pth")
            classifier.load_state_dict(torch.load(classifier_path))

            output_name = f"{test_name}/from_state_{source_state_id}"
            summary[f"from_state_{source_state_id}"] = render_all(dataset.model_path, output_name, scene.loaded_iter, target_views, gaussians,
                                                                  pipeline, background, classifier, transform_list)

        summary_dir = os.path.join(dataset.model_path, test_name)
        makedirs(summary_dir, exist_ok=True)
        with open(os.path.join(summary_dir, "summary_metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)
       

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    render_params = RenderParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--train_state_ids", nargs="+", type=int, default=[0], help="Training state ids to render into the test state")
    parser.add_argument("--test_source_path", type=str, default="", help="Source path of the target/test state")
    parser.add_argument("--test_name", type=str, default="test_state", help="Output folder name for this test state")
    parser.add_argument("--test_state_index", type=int, default=0, help="Index of test/state_* to render when using --dataset_path")
    parser.add_argument("--use_test_cameras", action="store_true", default=False, help="Render test cameras instead of train cameras")
    args = get_combined_args(parser)
    configure_render_dataset_package_args(args)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), pipeline.extract(args), render_params.extract(args),
                args.train_state_ids, args.test_source_path, args.test_name, args.use_test_cameras)
