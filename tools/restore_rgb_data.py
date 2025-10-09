import os
import cv2
import json
from tqdm import tqdm
from pathlib import Path
from shutil import rmtree
from random import shuffle
from argparse import ArgumentParser
from pytubefix.cli import on_progress

from models.youtube_downloader import YouTubeBypassAge
from models.utils import crop


"""Restore rgb frame data in redacted data directory"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to the data directory containing track folders"
    )
    parser.add_argument(
        "--video_dir", type=str, required=True,
        default="./raw_videos",
        help="Path to a temporary directory for downloading videos",
    )
    parser.add_argument(
        "--cleanup", action='store_true',
        help="Whether to delete the temporary video directory after processing",
    )
    parser.add_argument(
        "--rgb_suffix", type=str, default="rgb.png",
        help="Suffix for RGB frame files"
    )
    return parser.parse_args()



def process_track(track_dir, rgb_suffix, verbose=True):
    try:
        track_id = track_dir.stem
        with open(track_dir / f"{track_id}_info.json", 'r') as f:
            track_info = json.load(f)
        video_id = track_info["video_id"]

        # Download video
        video_path = video_temp_dir / f"{video_id}.mp4"
        if not video_path.exists():
            yt = YouTubeBypassAge(
                "https://youtube.com/watch?v=" + video_id, on_progress_callback=on_progress
            )
            streams = yt.streams.filter(
                type="video", file_extension="mp4", progressive=False,
                custom_filter_functions=[lambda s: not s.video_codec.startswith("av01")]
            ).order_by("resolution")
            stream = streams.last()
            num_frames = int(stream.fps * yt.length)
            if verbose:
                print(
                    f"Downloading {video_id}: duration {yt.length}s (~{num_frames} frames), "
                    f"codec {stream.video_codec}, resolution {stream.resolution}",
                    flush=True
                )
            stream.download(
                output_path=video_path.parent, filename=f"{video_id}.mp4", filename_prefix=None, skip_existing=True,
                timeout=None, max_retries=3,
            )
            assert video_path.exists(), f"Failed to download {video_id}"

        # Extract and saveframes
        all_metadata = list(track_dir.glob(f"*metadata.json"))
        for metadata_file in tqdm(all_metadata):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            video_frame_id = metadata["video_frame_id"]
            video_frame_time = metadata["video_frame_time"]
            frame_id = video_frame_id
            frame_time = video_frame_time
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            # cap.set(cv2.CAP_PROP_POS_MSEC, int(frame_time * 1000))
            ret, frame = cap.read()
            assert ret, f"Failed to read frame {frame_id} from {video_id}"
            frame_path = track_dir / (metadata_file.name.replace("metadata.json", rgb_suffix))
            # Crop frame
            crop_box = metadata["crop_box_xyxy"]
            crop_height = metadata["crop_height"]
            crop_width = metadata["crop_width"]
            frame = crop(frame, crop_box, crop_width, crop_height)
            cv2.imwrite(str(frame_path), frame)
            if verbose:
                print(f"Saved frame {frame_path}", flush=True)
    except Exception as e:
        print(f"{type(e).__name__}: {str(e)}", flush=True)
        for rgb_file in track_dir.glob(f"*{rgb_suffix}"):
            os.remove(rgb_file)



if __name__ == "__main__":
    args = parse_args()
    data_dir = Path(args.data_dir)
    video_dir = Path(args.video_dir)
    rgb_suffix = args.rgb_suffix
    assert not video_dir.exists(), f"temp dir {video_dir} exist"
    video_dir.mkdir(parents=True)
    # Get all data directories
    track_dirs = []
    for path in data_dir.rglob('*'):
        if path.is_dir():
            if not any(item.is_dir() for item in path.iterdir()):
                track_dirs.append(path)
    shuffle(track_dirs)
    for track_dir in track_dirs:
        process_track(track_dir, rgb_suffix, verbose=True)
    if args.cleanup:
        rmtree(video_dir, ignore_errors=True)




