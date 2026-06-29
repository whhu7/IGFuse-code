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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

def get_scene_path(args, field_name, scene_id):
    list_name = f"{field_name}s"
    values = getattr(args, list_name, None)
    if values and scene_id < len(values):
        return values[scene_id]
    raise ValueError(f"{list_name} must contain an entry for scene_id={scene_id}")

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], scene_id=1, load_cameras_only=False):
        """Load a scene from COLMAP or Blender-style transforms."""
        if load_cameras_only:
            assert gaussians is None, "load_cameras_only expects gaussians=None"
        self.model_path = args.model_path
        self.trained_model_path = get_scene_path(args, "trained_path", scene_id)
        self.source_path = get_scene_path(args, "source_path", scene_id)
        self.loaded_iter = None
        self.pretrained_ply_path = None
        self.gaussians = gaussians
        self.scene_id = scene_id
        self.load_from_trained = getattr(args, "load_from_trained", False)
        if load_iteration:
            if load_iteration == -1:
                if self.load_from_trained:
                    if self.trained_model_path.endswith(".ply"):
                        self.pretrained_ply_path = self.trained_model_path
                        self.loaded_iter = -1
                    else:
                        packed_ply_path = os.path.join(self.trained_model_path, "gaussian", "point_cloud.ply")
                        if os.path.exists(packed_ply_path):
                            self.pretrained_ply_path = packed_ply_path
                            self.loaded_iter = -1
                        else:
                            self.loaded_iter = searchForMaxIteration(os.path.join(self.trained_model_path, "point_cloud"))
                else:
                    self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration

        self.train_cameras = {}
        self.test_cameras = {}
        if os.path.exists(os.path.join(self.source_path, "sparse")) or os.path.exists(os.path.join(self.source_path, "colmap")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](self.source_path, args.images, args.eval, args.object_path)
        elif os.path.exists(os.path.join(self.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](self.source_path, args.white_background, args.eval, extension=".png", objects_folder=args.object_path)
        else:
            assert False, "Could not recognize scene type!"

        if not load_cameras_only and not self.loaded_iter:
            # First-time training setup: cache input point cloud and camera list.
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        if self.gaussians is not None:
            if self.loaded_iter:
                if self.load_from_trained:
                    if self.pretrained_ply_path:
                        self.gaussians.load_ply(self.pretrained_ply_path)
                    else:
                        self.gaussians.load_ply(os.path.join(self.trained_model_path,
                                                                "point_cloud",
                                                                "iteration_" + str(self.loaded_iter),
                                                                "point_cloud.ply"))
                else:
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud_{}.ply".format(self.scene_id)))
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_{}.ply".format(self.scene_id)))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]