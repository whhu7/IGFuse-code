# Copyright (c) 2026 IGFuse
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping

import os
import torch
from random import randint
from utils.loss_utils import l1_loss,ssim, loss_cls_3d
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import numpy as np
import random
import torch.nn.functional as F
from utils.dataset_utils import configure_train_dataset_package_args
from utils.fusion_utils import run_coprune_and_copy
from utils.mask_utils import get_mask3d_list
from utils.pose_utils import (
    build_transform_list,
    load_position_data_from_path,
    load_pseudo_position_paths,
    load_state_position_data,
)

def image_consistency_loss(image_a, image_b, lambda_dssim):
    Ll1 = l1_loss(image_a, image_b)
    return (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image_a, image_b))

def soft_semantic_consistency_loss(logits_a, logits_b, num_classes):
    log_prob_a = F.log_softmax(logits_a, dim=0)
    log_prob_b = F.log_softmax(logits_b, dim=0)
    prob_a = torch.softmax(logits_a.detach(), dim=0)
    prob_b = torch.softmax(logits_b.detach(), dim=0)
    loss_ab = -(prob_b * log_prob_a).sum(dim=0).mean()
    loss_ba = -(prob_a * log_prob_b).sum(dim=0).mean()
    return 0.5 * (loss_ab + loss_ba) / torch.log(torch.tensor(num_classes, device=logits_a.device))
        
def training(dataset, opt, pipe, saving_iterations, fusion_start_iteration, pseudo_start_iterations, pseudo_end_iterations, debug_from, use_wandb):
    first_iter = 0
    prepare_output_and_logger(dataset)
    GsDict = {}
    SceneDict = {}
    for i in range(opt.gaussiansN):
        GsDict[f"gs{i}"] = GaussianModel(dataset.sh_degree)
        SceneDict[f"scene{i}"] = Scene(dataset, GsDict[f"gs{i}"], load_iteration=-1, shuffle=True,scene_id=i)
        GsDict[f"gs{i}"].training_setup(opt)

    if opt.gaussiansN < 2:
        raise ValueError("IGFuse joint training needs at least two observed states")

    observed_state_ids = list(range(opt.gaussiansN))
    position_data = {state_id: load_state_position_data(dataset, state_id) for state_id in observed_state_ids}
    pseudo_position_paths = load_pseudo_position_paths(dataset)
    use_pseudo = len(pseudo_position_paths) > 0
    pseudo_end = pseudo_end_iterations if pseudo_end_iterations > 0 else opt.iterations
    if use_pseudo:
        print(
            f"Pseudo layouts: {len(pseudo_position_paths)} files from {dataset.pseudo_position_dir} "
            f"(lambda_pseudo={opt.lambda_pseudo})"
        )
        print(f"Fusion phase starts at iter {fusion_start_iteration} (align + Gaussian copy + co-prune)")
        print(f"Pseudo loss: iter {pseudo_start_iterations} -> {pseudo_end}")
    elif dataset.pseudo_position_dir:
        print(f"Pseudo layouts dir set but empty: {dataset.pseudo_position_dir}")
        print(f"Fusion phase starts at iter {fusion_start_iteration} (align + Gaussian copy + co-prune)")
    else:
        print(f"Fusion phase starts at iter {fusion_start_iteration} (align + Gaussian copy + co-prune)")

    transform_cache = {}
    for source_id in observed_state_ids:
        for target_id in observed_state_ids:
            if source_id == target_id:
                continue
            transform_cache[(source_id, target_id)] = build_transform_list(position_data[source_id], position_data[target_id])

    # Background is class 0; object classes are 1..K.
    obj_ids = np.arange(len(position_data[observed_state_ids[0]]))
    num_classes = obj_ids.shape[0]+1

    classifier = torch.nn.Conv2d(GsDict[f"gs0"].num_objects, num_classes, kernel_size=1)
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
    classifier.cuda()
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1


    view_Dict = {}
    for i in range(opt.gaussiansN):
        view_Dict[f"stack_{i}"] = []

    def fusion_active(iteration):
        return iteration >= fusion_start_iteration

    def pseudo_active(iteration):
        return (
            use_pseudo
            and iteration >= pseudo_start_iterations
            and iteration <= pseudo_end
        )

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        for i in range(opt.gaussiansN):
            GsDict[f"gs{i}"].update_learning_rate(iteration)

        pair_state_ids = None
        if fusion_active(iteration):
            pair_state_ids = random.sample(observed_state_ids, 2)
            active_state_ids = pair_state_ids
        else:
            active_state_ids = observed_state_ids

        for i in active_state_ids:
            if not view_Dict[f"stack_{i}"]:
                view_Dict[f"stack_{i}"] = SceneDict[f"scene{i}"].getTrainCameras().copy()
            viewpoint_cam = view_Dict[f"stack_{i}"].pop(randint(0, len(view_Dict[f"stack_{i}"])-1))
            view_Dict[f"cam_{i}"] = viewpoint_cam
        
        RenderDict = {}
        LossDict = {}
        Loss = torch.tensor(0.0, device="cuda")
        loss_obj = torch.tensor(0.0, device="cuda")
        loss_obj_3d = None
        mask_cache = {}
        pred_obj_cache = {}

        def build_mask3d_list(state_id):
            gaussians = GsDict[f"gs{state_id}"]
            current_n = gaussians.get_xyz.shape[0]
            if state_id in mask_cache:
                cached_n = mask_cache[state_id][0].shape[0]
                if cached_n != current_n:
                    del mask_cache[state_id]
                    pred_obj_cache.pop(state_id, None)
            if state_id not in mask_cache:
                mask_cache[state_id], pred_obj_cache[state_id] = get_mask3d_list(gaussians, classifier, obj_ids)
            return mask_cache[state_id]

        def get_pred_obj(state_id):
            build_mask3d_list(state_id)
            return pred_obj_cache[state_id]

        if (iteration - 1) == debug_from:
            pipe.debug = True
     
        for i in active_state_ids:
            RenderDict[f"render_pkg_gs{i}"] = render(view_Dict[f"cam_{i}"], GsDict[f"gs{i}"], pipe, background, False)
            RenderDict[f"image_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["render"]
            RenderDict[f"viewspace_point_tensor_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["viewspace_points"]
            RenderDict[f"visibility_filter_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["visibility_filter"]
            RenderDict[f"radii_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["radii"]
            RenderDict[f"objects_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["render_seg"]

        for i in active_state_ids:
            gt_obj =  view_Dict[f"cam_{i}"].objects.cuda().long()
            logits = classifier(RenderDict[f"objects_gs{i}"])
            loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
            loss_obj = loss_obj / torch.log(torch.tensor(num_classes, device=logits.device))
            gt_image = view_Dict[f"cam_{i}"].original_image.cuda()
            L_img = image_consistency_loss(RenderDict[f"image_gs{i}"], gt_image, opt.lambda_dssim)

            loss_obj_3d = None
            if iteration % opt.reg3d_interval == 0:
                logits3d = classifier(GsDict[f"gs{i}"]._objects_dc.permute(2,0,1))
                prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0)
                loss_obj_3d = loss_cls_3d(GsDict[f"gs{i}"]._xyz.squeeze().detach(), prob_obj3d, opt.reg3d_k, opt.reg3d_lambda_val, opt.reg3d_max_points, opt.reg3d_sample_size)
                loss = loss_obj + loss_obj_3d + L_img
            else:
                loss = loss_obj + L_img    
            LossDict[f"loss_gs{i}"] = loss
            Loss += loss
            
        if fusion_active(iteration):
            align_loss = torch.tensor(0.0, device="cuda")
            pseudo_loss = torch.tensor(0.0, device="cuda")
            pair_i, pair_j = pair_state_ids
            pseudo_transform_cache = {}

            # Warm up transformed Gaussians once when pseudo loss starts.
            if pseudo_active(iteration) and iteration == pseudo_start_iterations:
                pseudo_position_path = random.choice(pseudo_position_paths)
                pseudo_position_data = load_position_data_from_path(pseudo_position_path)
                for i in pair_state_ids:
                    with torch.no_grad():
                        pseudo_transform_cache[i] = build_transform_list(position_data[i], pseudo_position_data)
                        GsDict[f"gs{i}"].transform_setup(opt, build_mask3d_list(i), pseudo_transform_cache[i])

            for source_id, target_id in [(pair_i, pair_j), (pair_j, pair_i)]:
                GsDict[f"gs{source_id}"].transform_setup(opt, build_mask3d_list(source_id), transform_cache[(source_id, target_id)])
                target_cam = view_Dict[f"cam_{target_id}"]
                render_pkg_T = render(target_cam, GsDict[f"gs{source_id}"], pipe, background, True)
                image_T = render_pkg_T["render"]
                objects_T = render_pkg_T["render_seg"]
                gt_image = target_cam.original_image.cuda()
                gt_obj = target_cam.objects.cuda().long()
                logits = classifier(objects_T)
                loss_obj_T = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
                loss_obj_T = loss_obj_T / torch.log(torch.tensor(num_classes, device=logits.device))
                align_loss = align_loss + image_consistency_loss(image_T, gt_image, opt.lambda_dssim) + loss_obj_T

            if pseudo_active(iteration):
                pseudo_position_path = random.choice(pseudo_position_paths)
                pseudo_position_data = load_position_data_from_path(pseudo_position_path)
                pseudo_cam = view_Dict[f"cam_{random.choice(pair_state_ids)}"]
                pseudo_render = {}
                for source_id in pair_state_ids:
                    pseudo_transform = build_transform_list(position_data[source_id], pseudo_position_data)
                    GsDict[f"gs{source_id}"].transform_setup(opt, build_mask3d_list(source_id), pseudo_transform)
                    render_pkg_P = render(pseudo_cam, GsDict[f"gs{source_id}"], pipe, background, True)
                    pseudo_render[source_id] = {
                        "image": render_pkg_P["render"],
                        "logits": classifier(render_pkg_P["render_seg"]),
                    }

                pseudo_loss = image_consistency_loss(pseudo_render[pair_i]["image"], pseudo_render[pair_j]["image"], opt.lambda_dssim)
                pseudo_loss = pseudo_loss + soft_semantic_consistency_loss(pseudo_render[pair_i]["logits"], pseudo_render[pair_j]["logits"], num_classes)

            LossDict["loss_align"] = align_loss
            LossDict["loss_pseudo"] = pseudo_loss
            Loss = Loss + opt.lambda_align * align_loss + opt.lambda_pseudo * pseudo_loss

        Loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * Loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if  iteration < opt.densify_until_iter:
                for i in active_state_ids:
                    viewspace_point_tensor = RenderDict[f"viewspace_point_tensor_gs{i}"]
                    visibility_filter = RenderDict[f"visibility_filter_gs{i}"]
                    radii = RenderDict[f"radii_gs{i}"]
                   
                    GsDict[f"gs{i}"].max_radii2D[visibility_filter] = torch.max(GsDict[f"gs{i}"].max_radii2D[visibility_filter], radii[visibility_filter])
                    GsDict[f"gs{i}"].add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    for i in active_state_ids:
                        GsDict[f"gs{i}"].densify_and_prune(opt.densify_grad_threshold,0.005, SceneDict[f"scene{i}"].cameras_extent, size_threshold)                              

                for i in active_state_ids:
                    if iteration  % opt.opacity_reset_interval == 0:
                        GsDict[f"gs{i}"].reset_opacity()

            if iteration < opt.iterations:
                for i in range(opt.gaussiansN):
                    GsDict[f"gs{i}"].optimizer.step()
                    GsDict[f"gs{i}"].optimizer.zero_grad(set_to_none = True)
                cls_optimizer.step()
                cls_optimizer.zero_grad()

            # Co-prune inconsistent Gaussians; at fusion_start also copy ground points across states.
            if fusion_active(iteration) and iteration % 500 == 0:
                run_coprune_and_copy(
                    GsDict,
                    SceneDict,
                    opt,
                    iteration,
                    fusion_start_iteration,
                    pair_state_ids,
                    obj_ids,
                    transform_cache,
                    build_mask3d_list,
                    get_pred_obj,
                )

            log_training_metrics(iteration, loss_obj, Loss, iter_start.elapsed_time(iter_end), loss_obj_3d, use_wandb)
            if (iteration in saving_iterations):
                for i in range(opt.gaussiansN):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    save_dir = os.path.join(SceneDict[f"scene{i}"].model_path, "point_cloud/iteration_{}".format(iteration))
                    os.makedirs(save_dir, exist_ok=True)
                    SceneDict[f"scene{i}"].save(iteration)
                    torch.save(classifier.state_dict(), os.path.join(save_dir,'classifier.pth'))
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def log_training_metrics(iteration, loss_obj, loss, elapsed, loss_obj_3d, use_wandb):
    if use_wandb:
        if loss_obj_3d:
            wandb.log({"train_loss_patches/total_loss": loss.item(), "train_loss_patches/loss_obj": loss_obj.item(), "train_loss_patches/loss_obj_3d": loss_obj_3d.item(), "iter_time": elapsed, "iter": iteration})
        else:
            wandb.log({"train_loss_patches/total_loss": loss.item(), "train_loss_patches/loss_obj": loss_obj.item(), "iter_time": elapsed, "iter": iteration})

    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000,15_000,20_000])
    parser.add_argument("--fusion_start_iteration", type=int, default=10_000,
                        help="Start dual-state align, co-prune, and one-time Gaussian copy from paired state")
    parser.add_argument("--pseudo_start_iterations", type=int, default=10_000,
                        help="Start pseudo-layout loss")
    parser.add_argument("--pseudo_end_iterations", type=int, default=-1,
                        help="Stop pseudo loss after this iteration; -1 means train to the end")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--use_wandb", action='store_true', default=False, help="Use wandb to record loss value")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    configure_train_dataset_package_args(args)

    if args.pseudo_start_iterations < args.fusion_start_iteration:
        raise ValueError(
            f"pseudo_start_iterations ({args.pseudo_start_iterations}) must be >= "
            f"fusion_start_iteration ({args.fusion_start_iteration})"
        )
    if args.pseudo_end_iterations > 0 and args.pseudo_end_iterations < args.pseudo_start_iterations:
        raise ValueError(
            f"pseudo_end_iterations ({args.pseudo_end_iterations}) must be >= "
            f"pseudo_start_iterations ({args.pseudo_start_iterations}) or -1"
        )
    
    print("Optimizing " + args.model_path)

    if args.use_wandb:
        wandb.init(project="IGFuse")
        wandb.config.args = args
        run_name = "_".join(args.model_path.split("/")[1:])
        wandb.run.name = run_name

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations,
             args.fusion_start_iteration, args.pseudo_start_iterations, args.pseudo_end_iterations,
             args.debug_from, args.use_wandb)

    print("\nTraining complete.")