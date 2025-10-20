import os
import torch
import random
from tqdm import tqdm
from pathlib import Path
from configargparse import ArgumentParser
from models.flow_processor import FlowProcessor
from models.utils import get_all_sequence_dirs


# Extract flow from a dataset without using db


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--flow_method", type=str, default="sea_raft", help="Flow estimation method")
    parser.add_argument("--flow_batch_size", type=int, default=8, help="Batch size for flow estimation")
    args, _ = parser.parse_known_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    flow_estimator = FlowProcessor(
        flow_method=args.flow_method,
        image_suffix="rgb.png",
        flow_suffix="flow.png",
        mask_suffix="mask.png",
        flow_batch_size=args.flow_batch_size,
        device=device,
    )
    all_track_dirs = get_all_sequence_dirs(args.data_dir)
    for track_path in tqdm(all_track_dirs):
        skip_track = True
        all_images = sorted([os.path.join(track_path, f) for f in os.listdir(track_path) if f.endswith("rgb.png")])
        for img_path in all_images[:-1]:
            flow_path = str(img_path).replace("rgb.png", "flow.png")
            if not os.path.exists(flow_path):
                skip_track = False
                break
        if not skip_track:
            print(f"Processing: {track_path}")
            flow_estimator.run_track(track_path)
