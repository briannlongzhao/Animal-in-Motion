import os

import cv2
from tqdm import tqdm
from pathlib import Path
from shutil import copytree
from configargparse import ArgumentParser

from database import Database, parse_version

"""
Create a database with user provided video for processing.
Video directory should contain subdirectories of different categories
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, is_config_file=True, help="Path to yaml config file")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to video directory")
    parser.add_argument("--video_suffix", type=str, default=".mp4", help="Suffix of video files to process")
    parser.add_argument("--base_path", type=str, help="Base path for the dataset")
    parser.add_argument("--db_path", type=str, help="Path to save the database file")
    parser.add_argument("--version", type=str, default=None, help="Version of the dataset")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    video_version = parse_version(args.version).get("video")
    video_data_dir = Path(args.base_path) / (f"video_{video_version}" if video_version else "video")
    if args.base_path:
        os.makedirs(args.base_path, exist_ok=True)
    video_dir = Path(args.video_dir)
    db = Database(db_path=args.db_path, version=args.version)
    db.make_video_table()
    for category_dir in video_dir.iterdir():
        if not category_dir.is_dir():
            continue
        if args.base_path:
            copytree(
                str(category_dir), str(video_data_dir / category_dir.name), dirs_exist_ok=True
            )
            category_dir = video_data_dir / category_dir.name
        category = category_dir.name
        for video_file in tqdm(list(category_dir.glob(f"*{args.video_suffix}")), desc=f"Processing {category}"):
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                print(f"Failed to open video file: {video_file}")
                continue
            print(f"Processing video: {video_file}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frames / fps if fps > 0 else None
            db.insert_video(
                video_id=video_file.stem,
                category=category,
                fps=fps,
                duration=duration,
                frames=frames,
                video_path=str(video_file),
                query_text="",
            )
