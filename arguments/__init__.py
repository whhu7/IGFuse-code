#
# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco
#

from argparse import ArgumentParser, Namespace
import ast
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 

            # Skip CLI short flags when the option name ends with a digit (e.g. sh_degree).
            if shorthand and key[-1].isdigit():
                shorthand = False
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._dataset_path = ""
        self._model_path = ""
        self._pseudo_position_dir = ""
        
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self._object_path = "masks"
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        if g.dataset_path:
            g.dataset_path = os.path.abspath(g.dataset_path)
        if g.pseudo_position_dir:
            g.pseudo_position_dir = os.path.abspath(g.pseudo_position_dir)
        g.source_paths = [os.path.abspath(path) for path in getattr(args, "source_paths", [])]
        g.trained_paths = [os.path.abspath(path) for path in getattr(args, "trained_paths", [])]
        g.load_from_trained = getattr(args, "load_from_trained", False)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.gaussiansN = 2
        self.iterations = 20_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 20_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 5_000
        self.densify_until_iter = 10_000
        self.densify_grad_threshold = 0.0002

        self.reg3d_interval = 5
        self.reg3d_k = 5
        self.reg3d_lambda_val = 2
        self.reg3d_max_points = 200_000
        self.reg3d_sample_size = 1000
        self.lambda_align = 1.0
        self.lambda_pseudo = 1.0
        self.coprune_threshold = 0.5

        super().__init__(parser, "Optimization Parameters")

class RenderParams(ParamGroup):
    def __init__(self, parser):
        self.iteration = -1
        super().__init__(parser, "Rendering Parameters")

def _parse_namespace_literal(namespace_literal):
    expr = ast.parse(namespace_literal, mode="eval").body
    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name) or expr.func.id != "Namespace":
        raise ValueError("cfg_args must be a Namespace(...) literal")
    if expr.args:
        raise ValueError("cfg_args Namespace must use keyword arguments only")
    return Namespace(**{kw.arg: ast.literal_eval(kw.value) for kw in expr.keywords})


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    args_cfgfile = Namespace()
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            args_cfgfile = _parse_namespace_literal(cfg_file.read())
    except TypeError:
        print("Config file not found at")
    except FileNotFoundError:
        print("Config file not found at")

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)