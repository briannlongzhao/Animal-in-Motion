import sys
import traceback
import torch
import warnings
from pathlib import Path
from configargparse import ArgumentParser
from models.animal_tracker import AnimalTracker
from models.utils import Logger
from database import parse_version



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, is_config_file=True, help="Path to yaml config file")
    parser.add_argument("--category", type=str, help="Category name to process")
    parser.add_argument("--base_path", type=str, help='Path to base directory of data')
    parser.add_argument("--db_path", type=str, help="Path to database file")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--version", type=str, help="version of the dataset")
    parser.add_argument("--fps", type=int, default=10, help="Resample frames to this fps before tracking")
    parser.add_argument(
        "--categories", type=str, nargs='+', default=None,
        help='Category ids or names to process, available categories in categories/__init__.py'
    )
    parser.add_argument(
        "--compute_occlusion", action="store_true",
        help="Compute depth map and detect object occlusion"
    )
    parser.add_argument(
        "--compute_flow", action="store_true",
        help="Compute optical flow for tracks"
    )
    parser.add_argument(
        "--compute_pose", action="store_true",
        help="Compute pose estimation for animals"
    )
    parser.add_argument(
        "--sam2_prompt_type", type=str, default="mask", choices=["mask", "point", "box"],
        help="Prompt type to prompt SAM2 video predictor"
    )
    parser.add_argument(
        "--overlap_iou_threshold", type=float, default=0.1,
        help="IoU threshold for removing overlapped object in the same frame"
    )
    parser.add_argument(
        "--grounding_text_format", type=str, default="{}.",
        help="Format of grounding text input, the string will be formatted by category description"
    )
    parser.add_argument(
        "--truncation_border_width", type=int, default=3,
        help="Threshold of percentage of mask pixels that lies on the border of the frames"
    )
    parser.add_argument(
        "--bbox_area_threshold", type=int, default=256**2,
        help="Threshold of smallest bbox area to keep"
    )
    parser.add_argument(
        "--crop_margin", type=float, default=0.1,
        help="Proportion of bbox size reserved for crop margin, i.e, crop_width = bbox_width+2*bbox_width*<crop_margin>"
    )
    parser.add_argument(
        "--crop_height", type=int, default=512,
        help="Frame height of the output crop to be resized into"
    )
    parser.add_argument(
        "--crop_width", type=int, default=512,
        help="Frame width of the output crop to be resized into"
    )
    parser.add_argument(
        "--min_scene_len", type=float, default=20,
        help="Minimum length of a track"
    )
    parser.add_argument(
        "--consistency_iou_threshold", type=float, default=0.5,
        help="IOU threshold for track consistency, tracks with consecutive frames with small bbox iou will be filtered"
    )
    parser.add_argument(
        "--depth_method", type=str, choices=["patchfusion", "depth_anything", "depth_anything_v2", "midas"],
        default="depth_anything", help="Method used for depth estimation on each frame"
    )
    parser.add_argument(
        "--occlusion_batch_size", type=int, default=32,
        help="Batch size used for occlusion and depth estimation"
    )
    parser.add_argument(
        "--flow_method", type=str, default="sea_raft", choices=["sea_raft"],
        help="Method used for depth estimation on each frame"
    )
    parser.add_argument(
        "--flow_batch_size", type=int, default=4,
        help="Batch size used for flow estimation"
    )
    parser.add_argument(
        "--pose_method", type=str, default="vitpose", choices=["vitpose"],
        help="Method used for animal pose estimation on each frame"
    )
    parser.add_argument(
        "--use_gpt_filter", action="store_true", help="Use GPT to filter tracks"
    )
    parser.add_argument(
        "--pose_batch_size", type=int, default=1,
        help="Batch size used for pose estimation"
    )
    parser.add_argument(
        "--tracking_method", type=str, choices=["naive", "sam_track", "grounded_sam2"], default="grounded_sam2",
        help="naive for location/iou based matching, sam_track for Segment and Track Anything"
    )
    parser.add_argument(
        "--tracking_sam_step", type=int, default=50,
        help="Frame step interval for using Grounded SAM to detect new object in video"
    )
    parser.add_argument(
        "--max_track_gap", type=int, default=5,
        help=(
            "Maximum gap of a track between two consecutive frames to consider as the same track, "
            "used in removing discontinuous tracks and final smoothing"
        )
    )
    parser.add_argument("--save_visualization", action="store_true", help="Save visualization of tracking results")
    parser.add_argument(
        "--save_local_dir", type=str, default=None, help="Save tracking results to local ssd of compute node"
    )
    parser.add_argument("--image_suffix", type=str, default="rgb.png", help="Suffix of the image file")
    parser.add_argument("--masked_image_suffix", type=str, default="rgb_masked.png", help="Suffix of the masked image file")
    parser.add_argument("--mask_suffix", type=str, default="mask.png", help="Suffix of the mask file")
    parser.add_argument("--metadata_suffix", type=str, default="metadata.json", help="Suffix of the metadata file")
    parser.add_argument("--occlusion_suffix", type=str, default="occlusion.png", help="Suffix of the occlusion file")
    parser.add_argument("--flow_suffix", type=str, default="flow.png", help="Suffix of the flow file")
    parser.add_argument("--depth_suffix", type=str, default="depth.png", help="Suffix of the depth file")
    parser.add_argument("--keypoint_suffix", type=str, default="keypoint.txt", help="Suffix of the keypoint file")
    parser.add_argument("--pose_image_suffix", type=str, default="pose.png", help="Suffix of the pose image file")
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    track_version = parse_version(args.version).get("track")
    track_dir = f"track_{track_version}" if track_version else "track"
    output_dir = Path(args.base_path) / track_dir
    sys.stdout = Logger(output_dir / "log.txt")
    if args.save_local_dir is not None:
        args.save_local_dir = Path(args.save_local_dir) / track_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    animal_tracker = AnimalTracker(
        output_dir=output_dir,
        tracking_method=args.tracking_method,
        depth_method=args.depth_method,
        db_path=args.db_path,
        compute_occlusion=args.compute_occlusion,
        compute_flow=args.compute_flow,
        flow_method=args.flow_method,
        flow_batch_size=args.flow_batch_size,
        compute_pose=args.compute_pose,
        pose_method=args.pose_method,
        pose_batch_size=args.pose_batch_size,
        tracking_sam_step=args.tracking_sam_step,
        overlap_iou_threshold=args.overlap_iou_threshold,
        truncation_border_width=args.truncation_border_width,
        grounding_text_format=args.grounding_text_format,
        bbox_area_threshold=args.bbox_area_threshold,
        sam2_prompt_type=args.sam2_prompt_type,
        crop_height=args.crop_height,
        crop_width=args.crop_width,
        crop_margin=args.crop_margin,
        min_scene_len=args.min_scene_len,
        verbose=args.verbose,
        save_visualization=args.save_visualization,
        consistency_iou_threshold=args.consistency_iou_threshold,
        occlusion_batch_size=args.occlusion_batch_size,
        use_gpt_filter=args.use_gpt_filter,
        categories=args.categories,
        max_track_gap=args.max_track_gap,
        fps=args.fps,
        save_local_dir=args.save_local_dir,
        version=args.version,
        device=device,
        image_suffix= args.image_suffix,
        masked_image_suffix=args.masked_image_suffix,
        mask_suffix=args.mask_suffix,
        metadata_suffix=args.metadata_suffix,
        occlusion_suffix=args.occlusion_suffix,
        flow_suffix=args.flow_suffix,
        depth_suffix=args.depth_suffix,
        keypoint_suffix=args.keypoint_suffix,
        pose_image_suffix=args.pose_image_suffix,
    )
    animal_tracker.run()


