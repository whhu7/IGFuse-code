# Copyright (c) 2026 IGFuse


import argparse
import json
import math
import os
import random
import statistics
from itertools import combinations

from utils.dataset_utils import configure_pseudo_dataset_package_args
from utils.pose_utils import load_position_data_from_path


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def object_keys(states):
    keys = list(states[0].keys())
    for state in states[1:]:
        if list(state.keys()) != keys:
            raise ValueError("All input json files must contain the same object ids in the same order.")
    return keys


def xy_from_matrix(matrix):
    return matrix[0][3], matrix[1][3]


def z_from_matrix(matrix):
    return matrix[2][3]


def collect_layout_stats(states, keys, padding_ratio):
    all_x = []
    all_y = []
    all_pair_distances = []
    z_values = {key: [] for key in keys}

    for state in states:
        xy = []
        for key in keys:
            matrix = state[key]["world_matrix"]
            x, y = xy_from_matrix(matrix)
            xy.append((x, y))
            all_x.append(x)
            all_y.append(y)
            z_values[key].append(z_from_matrix(matrix))

        for i, j in combinations(range(len(xy)), 2):
            all_pair_distances.append(math.dist(xy[i], xy[j]))

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_pad = (x_max - x_min) * padding_ratio
    y_pad = (y_max - y_min) * padding_ratio
    min_observed_distance = min(all_pair_distances) if all_pair_distances else 0.0
    mean_observed_distance = sum(all_pair_distances) / len(all_pair_distances) if all_pair_distances else 0.0
    median_observed_distance = statistics.median(all_pair_distances) if all_pair_distances else 0.0
    mean_z = {key: sum(values) / len(values) for key, values in z_values.items()}

    return {
        "x_range": (x_min - x_pad, x_max + x_pad),
        "y_range": (y_min - y_pad, y_max + y_pad),
        "min_observed_distance": min_observed_distance,
        "mean_observed_distance": mean_observed_distance,
        "median_observed_distance": median_observed_distance,
        "mean_z": mean_z,
    }


def sample_layout(keys, stats, min_distance, rng, max_attempts):
    positions = {}
    placed = []

    for key in keys:
        for _ in range(max_attempts):
            x = rng.uniform(*stats["x_range"])
            y = rng.uniform(*stats["y_range"])
            if all(math.dist((x, y), old_xy) >= min_distance for old_xy in placed):
                positions[key] = (x, y)
                placed.append((x, y))
                break
        else:
            return None

    return positions


def sample_layout_with_retries(keys, stats, min_distance, rng, max_attempts, max_layout_attempts):
    for _ in range(max_layout_attempts):
        positions = sample_layout(keys, stats, min_distance, rng, max_attempts)
        if positions is not None:
            return positions
    raise RuntimeError(
        "Failed to sample a pseudo layout that satisfies the minimum distance. "
        "Try reducing --min_distance_ratio, increasing --padding_ratio, or increasing --max_layout_attempts."
    )


def random_z_rotation_matrix(x, y, z, rng):
    angle = rng.uniform(0.0, 2.0 * math.pi)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return [
        [cos_a, -sin_a, 0.0, x],
        [sin_a, cos_a, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def build_pseudo_state(template_state, keys, positions, stats, rng):
    pseudo_state = {}

    for key in keys:
        x, y = positions[key]
        z = stats["mean_z"][key]

        pseudo_state[key] = {
            "object_name": template_state[key].get("object_name", key),
            "world_matrix": random_z_rotation_matrix(x, y, z, rng),
        }

    return pseudo_state


def pairwise_distances(positions, keys):
    distances = []
    for key_i, key_j in combinations(keys, 2):
        distances.append(math.dist(positions[key_i], positions[key_j]))
    return distances


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate pseudo object pose states from multiple real objects_poses.json files."
    )
    parser.add_argument("--dataset_path", default="", help="Dataset package root containing train/state_* folders.")
    parser.add_argument("--input_jsons", nargs="+", default=[], help="Observed objects_poses.json files.")
    parser.add_argument("--output_dir", default="", help="Directory to save generated pseudo json files.")
    parser.add_argument("--num_pseudo", type=int, default=100, help="Number of pseudo states to generate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--padding_ratio", type=float, default=0.05, help="Expand XY bounds estimated from observed states.")
    parser.add_argument(
        "--min_distance_ratio",
        type=float,
        default=1.0,
        help="Minimum pseudo XY distance = observed median pair distance * this ratio.",
    )
    parser.add_argument("--max_attempts", type=int, default=1000)
    parser.add_argument(
        "--max_layout_attempts",
        type=int,
        default=100,
        help="Maximum full-layout resampling attempts before failing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_pseudo_dataset_package_args(args)
    os.makedirs(args.output_dir, exist_ok=True)

    states = [load_position_data_from_path(path) for path in args.input_jsons]
    keys = object_keys(states)
    template_state = states[0]
    stats = collect_layout_stats(states, keys, args.padding_ratio)
    min_distance = stats["median_observed_distance"] * args.min_distance_ratio
    rng = random.Random(args.seed)

    print("Estimated layout bounds from observed states:")
    print(f"  x_range: {stats['x_range'][0]:.6f} -> {stats['x_range'][1]:.6f}")
    print(f"  y_range: {stats['y_range'][0]:.6f} -> {stats['y_range'][1]:.6f}")
    print(f"  observed min distance: {stats['min_observed_distance']:.6f}")
    print(f"  observed median distance: {stats['median_observed_distance']:.6f}")
    print(f"  observed mean distance: {stats['mean_observed_distance']:.6f}")
    print(f"  pseudo min distance: {min_distance:.6f}")

    for scene_id in range(args.num_pseudo):
        positions = sample_layout_with_retries(
            keys,
            stats,
            min_distance,
            rng,
            args.max_attempts,
            args.max_layout_attempts,
        )
        pseudo_state = build_pseudo_state(template_state, keys, positions, stats, rng)
        distances = pairwise_distances(positions, keys)
        output_file = os.path.join(args.output_dir, f"objects_poses_{scene_id:04d}.json")
        save_json(pseudo_state, output_file)

        min_pair_distance = min(distances) if distances else 0.0
        mean_pair_distance = sum(distances) / len(distances) if distances else 0.0
        print(
            f"pseudo {scene_id:04d}: min_dist={min_pair_distance:.6f}, "
            f"mean_dist={mean_pair_distance:.6f}"
        )

    print(f"Generated {args.num_pseudo} pseudo states in {args.output_dir}")


if __name__ == "__main__":
    main()
