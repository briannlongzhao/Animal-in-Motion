import os
from pathlib import Path
from shutil import rmtree, copy2
import cv2
import sys
import torch
import hydra
import traceback
from glob import glob
from tqdm import tqdm
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader
from accelerate import Accelerator
from random import shuffle
from subprocess import run
from models.utils import images_to_video, validate_all_to_device, get_all_sequence_dirs, copy_results


submodule_dir = os.path.abspath("externals/Animals")
sys.path.append(submodule_dir)
from models.animal_models.models import MagicponyVideo, FaunaVideo
from models.animal_models.datasets import SingleSequenceDataset, ImageDataset
from externals.Animals.model.utils.wandb_writer import WandbWriter
from externals.Animals.model.utils.misc import setup_runtime
submodule_dir = os.path.abspath("externals/Animals")
checkpoint_path = os.path.join(submodule_dir, "results/fauna/pretrained_fauna/pretrained_fauna.pth")
    

def update_config(cfg):
    with open_dict(cfg):
        cfg.method = cfg.get("method", "4dfauna")
        cfg.local_dir = cfg.get("local_dir")
        cfg.arti_recon_epoch = 25 if cfg.method == "4dfauna" else 0
        cfg.shape_recon_epoch = 0
        cfg.dataset.num_frames = 8
        cfg.dataset.batch_size = 1
        cfg.dataset.in_image_size = 512
        cfg.dataset.load_keypoint = True
        cfg.dataset.load_dino_feature = False
        cfg.use_logger = False
        cfg.model.cfg_predictor_instance.enable_deform = True
        cfg.model.cfg_predictor_instance.cfg_articulation.legs_to_body_joint_indices = [3, 6, 6, 3]
        cfg.model.cfg_predictor_instance.cfg_articulation.enable_refine = False
        cfg.model.cfg_predictor_instance.cfg_articulation.use_fauna_constraints = False
        cfg.model.cfg_optim_instance.lr = 0.1
        if cfg.method == "4dfauna":
            smooth_loss_weight = 50
            cfg.model.cfg_loss.keypoint_projection_loss_weight = 50
        else:
            smooth_loss_weight = 0
            cfg.model.cfg_loss.keypoint_projection_loss_weight = 0
        cfg.model.cfg_loss.arti_smooth_loss_weight = smooth_loss_weight
        cfg.model.cfg_loss.artivel_smooth_loss_weight = smooth_loss_weight
        cfg.model.cfg_loss.bone_smooth_loss_weight = smooth_loss_weight
        cfg.model.cfg_loss.bonevel_smooth_loss_weight = smooth_loss_weight
        cfg.model.cfg_loss.campose_smooth_loss_weight = smooth_loss_weight
        cfg.dataset.data_type = "sequence"
        cfg.category_mean = True
    return cfg


@hydra.main(config_path=os.path.join(submodule_dir, "config"), config_name="train_fauna")
def main(cfg: DictConfig):
    cfg = update_config(cfg)
    data_dir = cfg.get("data_dir")
    if cfg.local_dir is not None:
        os.makedirs(cfg.local_dir, exist_ok=True)
    assert data_dir is not None, "Please specify +data_dir=/path/to/data in command line"
    all_data_dir = get_all_sequence_dirs(data_dir)
    for d in tqdm(all_data_dir):
        print(f"Processing {d}")
        try:
            process_single_sequence(d, cfg)
        except Exception as e:
            print(f"Error processing {d}: {e}")
            traceback.print_exc()


def process_single_sequence(data_dir, cfg):
    output_path = os.path.join(data_dir, os.path.basename(data_dir) + "_{}")
    output_rgb_overlayed_path = output_path.format(f"rgb_overlayed_{cfg.method}.mp4")
    output_mask_diff_path = output_path.format(f"mask_diff_{cfg.method}.mp4")
    output_shading_path = output_path.format(f"shading_{cfg.method}.mp4")
    output_shading_bones_path = output_path.format(f"shading_bones_{cfg.method}.mp4")
    # out_mesh_path = os.path.join(data_dir, f"{os.path.basename(data_dir)}_mesh.glb")
    if os.path.exists(output_rgb_overlayed_path):
        print("skip already processed", data_dir)
        return
    if cfg.use_logger:
        logger = WandbWriter(project=f"{cfg.method}_video", config=cfg, local_dir=cfg.local_dir)
    else:
        logger = None
    if cfg.local_dir is not None:
        save_dir = os.path.join(cfg.local_dir, os.path.basename(data_dir))
    else:
        save_dir = data_dir
    accelerator = Accelerator()
    device = setup_runtime(cfg)
    model = FaunaVideo(cfg.model) if "fauna" in cfg.method else MagicponyVideo(cfg.model)
    print(f"Loading checkpoint from {checkpoint_path}")
    cp = torch.load(checkpoint_path, map_location="cpu")
    epoch, total_iter = cp["epoch"], cp["total_iter"]
    model.load_model_state(cp)
    model.to(device)
    model.reset_optimizers()
    for name, value in vars(model).items():
        if isinstance(value, torch.nn.Module):
            setattr(model, name, accelerator.prepare_model(value))
        if isinstance(value, torch.optim.Optimizer):
            setattr(model, name, accelerator.prepare_optimizer(value))
        if isinstance(value, torch.optim.lr_scheduler._LRScheduler):
            setattr(model, name, accelerator.prepare_scheduler(value))
    model.accelerator = accelerator

    dataset = SingleSequenceDataset(
        data_dir,
        num_frames=cfg.dataset.num_frames,
        in_image_size=cfg.dataset.in_image_size,
        out_image_size=cfg.dataset.out_image_size,
        load_keypoint=cfg.dataset.load_keypoint,
        load_dino_feature=cfg.dataset.load_dino_feature,
        random_sample=False,
        dense_sample=True,
        local_dir=cfg.local_dir
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True
    )
    dataloader = accelerator.prepare_data_loader(dataloader)

    if hasattr(model, "compute_mean_feature"):
        if hasattr(cfg, "category_mean") and cfg.category_mean:
            category_dir = os.path.dirname(data_dir)
            mean_feature_path = os.path.join(category_dir, "mean_feature.pth")
            if os.path.exists(mean_feature_path):
                print(f"Loading mean feature from {mean_feature_path}")
                mean_feature = torch.load(mean_feature_path)
                model.get_predictor("netBase").mean_feature = mean_feature
            else:
                category_dataset = ImageDataset(
                    category_dir,
                    in_image_size=cfg.dataset.in_image_size, out_image_size=cfg.dataset.out_image_size,
                    load_keypoint=False, load_dino_feature=False, load_depth=False, load_articulation=False, shuffle=False
                )
                category_dataloader = DataLoader(
                    category_dataset,
                    batch_size=cfg.dataset.batch_size * cfg.dataset.num_frames,
                    num_workers=1, shuffle=False, pin_memory=True
                )
                category_dataloader = accelerator.prepare_data_loader(category_dataloader)
                mean_feature = model.compute_mean_feature(category_dataloader)
                torch.save(mean_feature, mean_feature_path)
                print(f"Saved mean feature to {mean_feature_path}")
        else:
            model.compute_mean_feature(dataloader)

    # Finetune articulation
    model.set_finetune_arti()
    for _ in tqdm(range(cfg.arti_recon_epoch), total=cfg.arti_recon_epoch, desc="Finetune articulation"):
        for iteration, batch in enumerate(dataloader):
            total_iter += 1
            batch = validate_all_to_device(batch, device=device)
            m = model.forward_finetune_arti(batch, epoch=epoch, total_iter=total_iter, is_training=True)
            model.backward()
            # if "keypoint_projection_loss" in m:
            #     print("keypoint_projection_loss", m["keypoint_projection_loss"].item())
            # if "instance_normal_reg_loss" in m:
            #     print("instance_normal_reg_loss", m["instance_normal_reg_loss"].item())
            if logger is not None:
                for name, loss in m.items():
                    logger.add_scalar(f'train_loss/{name}', loss, total_iter)

    # Finetune texture
    model.set_finetune_texture()
    for _ in tqdm(range(cfg.shape_recon_epoch), total=cfg.shape_recon_epoch, desc="Finetune texture"):
        for iteration, batch in enumerate(dataloader):
            total_iter += 1
            if total_iter % 50 == 0:
                use_logger = logger
            else:
                use_logger = None
            batch = validate_all_to_device(batch, device=device)
            m = model.forward_finetune_texture(batch, epoch=epoch, total_iter=total_iter, is_training=True,
                                               logger=use_logger)
            model.backward()
            if logger and total_iter % 10 == 0:
                for name, loss in m.items():
                    logger.add_scalar(f'train_loss/{name}', loss, total_iter)

    # Inference dataset and dataloader
    dataset = SingleSequenceDataset(
        data_dir,
        num_frames=cfg.dataset.num_frames,
        in_image_size=cfg.dataset.in_image_size,
        out_image_size=cfg.dataset.out_image_size,
        load_keypoint=cfg.dataset.load_keypoint,
        load_dino_feature=cfg.dataset.load_dino_feature,
        random_sample=False,
        dense_sample=False,
        local_dir=cfg.local_dir
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True
    )
    dataloader = accelerator.prepare_data_loader(dataloader)

    # Inference
    model.set_inference()
    for k in model.netInstance.articulation_dict.previous_tensor_mean_dict.keys():  # for debugging only, no-op
        model.netInstance.articulation_dict.previous_tensor_mean_dict[k] = model.netInstance.articulation_dict.tensor_dict[k].mean().item()

    for iteration, batch in tqdm(enumerate(dataloader), desc="Inference"):
        batch = validate_all_to_device(batch, device=device)
        with torch.no_grad():
            m = model.inference(
                batch, epoch=epoch, total_iter=total_iter, local_save_dir=save_dir, image_suffix=cfg.method
            )


    # Save normal video
    # image_files = list(Path(save_dir).rglob(f"*normal.png"))
    # images_to_video(image_files, out_normal_path)
    # Save mask_diff video
    image_files = sorted(list(Path(save_dir).rglob(f"*mask_diff_{cfg.method}.png")))
    images_to_video(image_files, output_mask_diff_path)
    # Save shading video
    image_files = sorted(list(Path(save_dir).rglob(f"*shading_{cfg.method}.png")))
    images_to_video(image_files, output_shading_path)
    # Save shading_bones video
    image_files = sorted(list(Path(save_dir).rglob(f"*shading_bones_{cfg.method}.png")))
    images_to_video(image_files, output_shading_bones_path)
    # Save rgb_overlayed video
    image_files = sorted(list(Path(save_dir).rglob(f"*rgb_overlayed_{cfg.method}.png")))
    images_to_video(image_files, output_rgb_overlayed_path)

    # Save mesh glb files (not fully tested)
    # run(["python", "tools/generate_glb.py", "--input_dir", save_dir, "--output_path", out_mesh_path])

    if cfg.local_dir is not None:
        copy_results(
            src_dir=save_dir, dst_dir=data_dir,
            suffixes=[
                f"{cfg.method}.png", f"{cfg.method}.txt", f"{cfg.method}.obj", f"{cfg.method}.mp4",
                # "normal.png", "mask_pred.png", "mask_diff.png", "shading.png", "shading_bones.png", "rgb_overlayed.png",
                # "keypoint_pred.txt", "mesh.obj",
            ]
        )
        run(["rm", "-rf", save_dir])

    if logger is not None:
        logger.finish()

    print("Finished processing", data_dir)


if __name__ == "__main__":
    main()

