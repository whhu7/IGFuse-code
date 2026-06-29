import os
from glob import glob


def configure_train_dataset_package_args(args):
    if not args.dataset_path:
        return

    dataset_path = os.path.abspath(args.dataset_path)
    train_root = os.path.join(dataset_path, "train")
    train_state_paths = sorted(
        path for path in glob(os.path.join(train_root, "state_*"))
        if os.path.isdir(path)
    )
    if len(train_state_paths) < 2:
        raise ValueError(f"--dataset_path needs at least two train/state_* folders: {train_root}")

    args.dataset_path = dataset_path
    args.source_paths = train_state_paths
    args.trained_paths = train_state_paths
    args.gaussiansN = len(train_state_paths)
    args.object_path = "masks"
    args.load_from_trained = True

    if getattr(args, "pseudo_position_dir", ""):
        pseudo_dir = os.path.abspath(args.pseudo_position_dir)
    else:
        pseudo_dir = os.path.join(dataset_path, "pseudo")
    pseudo_paths = sorted(glob(os.path.join(pseudo_dir, "objects_poses*.json")))
    args.pseudo_position_dir = pseudo_dir if pseudo_paths else ""
    if pseudo_paths:
        print(
            f"Pseudo layouts: {pseudo_dir} ({len(pseudo_paths)} files, "
            f"lambda_pseudo={args.lambda_pseudo})"
        )
    else:
        args.lambda_pseudo = 0.0
        print(f"Pseudo layouts: none ({pseudo_dir} empty or missing), lambda_pseudo=0")

    print(f"Loaded dataset package: {dataset_path}")
    print(f"Training states: {len(train_state_paths)}")
    for state_id, state_path in enumerate(train_state_paths):
        print(f"  state_{state_id:03d}: {state_path}")


def configure_render_dataset_package_args(args):
    if not args.dataset_path:
        if args.test_source_path:
            args.test_source_path = os.path.abspath(args.test_source_path)
        args.load_from_trained = False
        return

    dataset_path = os.path.abspath(args.dataset_path)
    train_root = os.path.join(dataset_path, "train")
    test_root = os.path.join(dataset_path, "test")
    train_state_paths = sorted(
        path for path in glob(os.path.join(train_root, "state_*"))
        if os.path.isdir(path)
    )
    test_state_paths = sorted(
        path for path in glob(os.path.join(test_root, "state_*"))
        if os.path.isdir(path)
    )

    if not train_state_paths:
        raise ValueError(f"--dataset_path needs train/state_* folders: {train_root}")
    if not test_state_paths:
        raise ValueError(f"--dataset_path needs test/state_* folders: {test_root}")

    test_state_index = getattr(args, "test_state_index", 0)
    if test_state_index < 0 or test_state_index >= len(test_state_paths):
        raise ValueError(f"--test_state_index {test_state_index} is out of range for {len(test_state_paths)} test states")

    args.dataset_path = dataset_path
    args.source_paths = train_state_paths
    args.trained_paths = train_state_paths
    args.train_state_ids = list(range(len(train_state_paths)))
    args.test_source_path = test_state_paths[test_state_index]
    args.test_name = f"test_state_{test_state_index:03d}"
    args.object_path = "masks"
    args.load_from_trained = False

    print(f"Loaded dataset package: {dataset_path}")
    print(f"Rendering train states: {args.train_state_ids}")
    print(f"Test state: {args.test_source_path}")


def configure_pseudo_dataset_package_args(args):
    if not args.dataset_path:
        if not args.input_jsons:
            raise ValueError("Either --dataset_path or --input_jsons is required.")
        if not args.output_dir:
            raise ValueError("--output_dir is required when using --input_jsons.")
        return

    dataset_path = os.path.abspath(args.dataset_path)
    train_root = os.path.join(dataset_path, "train")
    input_jsons = sorted(glob(os.path.join(train_root, "state_*", "objects_poses.json")))
    if len(input_jsons) < 2:
        raise ValueError(f"--dataset_path needs at least two train/state_*/objects_poses.json files: {train_root}")

    args.dataset_path = dataset_path
    args.input_jsons = input_jsons
    if not args.output_dir:
        args.output_dir = os.path.join(dataset_path, "pseudo")

    print(f"Loaded dataset package: {dataset_path}")
    print("Input pose jsons:")
    for path in args.input_jsons:
        print(f"  {path}")
    print(f"Output pseudo dir: {args.output_dir}")
