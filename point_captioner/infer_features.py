# modified from distill.py
import os
import random
import numpy as np
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from MinkowskiEngine import SparseTensor
from util import config
from dataset.feature_loader import FusedFeatureLoaderSMAP, collation_fn_smap
from models.disnet import DisNet as Model
from tqdm import tqdm


def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def generate_per_view_lidar_mask(coords, dataset="nuscenes", **kwargs):
    if dataset == "nuscenes":
        n_views = kwargs.get("n_views", 6)
        step = 2 * np.pi / n_views
        start_angle = -np.pi

        locs_new = cart2polar(coords)
        mask = np.zeros((locs_new.shape[0], n_views), dtype=np.bool_)
        for i in range(n_views):
            angle = start_angle + i * step
            mask[:, i] = np.logical_and(locs_new[:, 1] >= angle, locs_new[:, 1] < angle + step)
    elif dataset == "scannet":
        square_size = kwargs.get("square_size", 1.0)
        th_valid_point = kwargs.get("th_valid_point", 200)
        minn = torch.min(coords, dim=0)[0]
        maxx = torch.max(coords, dim=0)[0]
        # Traverse the 3D space and find the points that are in the same square
        res_mask = []
        for x0 in np.arange(minn[0], maxx[0], square_size):
            for y0 in np.arange(minn[1], maxx[1], square_size):
                mask = torch.logical_and(coords[:, 0] >= x0, coords[:, 0] < x0 + square_size)
                mask = torch.logical_and(mask, coords[:, 1] >= y0)
                mask = torch.logical_and(mask, coords[:, 1] < y0 + square_size)
                if mask.sum() >= th_valid_point:
                    res_mask.append(mask.unsqueeze(1))
        mask = torch.cat(res_mask, axis=1)
    return mask


def get_parser():
    """Parse the config file."""

    parser = argparse.ArgumentParser(description="OpenScene 3D distillation.")
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default=None,
        help="the full path of checkpoint file to load",
    )
    parser.add_argument(
        "--save_dir_feat",
        type=str,
        default=None,
        help="path to save the features",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default=None,
        help="path to the config file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        # default="train",
        help="split",
    )
    parser.add_argument(
        "--per_view_strategy",
        type=str,
        default="lidar",
        choices=["camera", "lidar"],
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug mode",
    )
    args = parser.parse_args()
    if args.save_dir_feat is not None:
        os.makedirs(args.save_dir_feat, exist_ok=True)

    return args


def get_model(cfg):
    """Get the 3D model."""

    model = Model(cfg=cfg)
    return model


def main():
    # ################# User defined variables #################
    args = get_parser()
    cfg = config.load_cfg_from_cfg_file(args.cfg_path)
    # ################# Overwrite configs in training cfg file #################
    cfg.train_gpu = [0]
    cfg.batch_size_val = 1

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in cfg.train_gpu)
    cudnn.benchmark = False
    if cfg.manual_seed is not None:
        random.seed(cfg.manual_seed)
        np.random.seed(cfg.manual_seed)
        torch.manual_seed(cfg.manual_seed)
        torch.cuda.manual_seed(cfg.manual_seed)
        torch.cuda.manual_seed_all(cfg.manual_seed)

    # By default we use shared memory for training
    if not hasattr(cfg, "use_shm"):
        cfg.use_shm = True

    print(
        "torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s"
        % (
            torch.__version__,
            torch.version.cuda,
            torch.backends.cudnn.version(),
            torch.backends.cudnn.enabled,
        )
    )

    cfg.distributed = cfg.world_size > 1 or cfg.multiprocessing_distributed
    cfg.ngpus_per_node = len(cfg.train_gpu)
    if len(cfg.train_gpu) == 1:
        cfg.sync_bn = False
        cfg.distributed = False
        cfg.multiprocessing_distributed = False
        cfg.use_apex = False

    cfg.method = "smap"

    model = get_model(cfg)
    model = model.cuda()

    # ####################### Data Loader ####################### #
    val_data = FusedFeatureLoaderSMAP(
        datapath_prefix=cfg.data_root,
        datapath_prefix_feat=cfg.data_root_2d_fused_feature,
        voxel_size=cfg.voxel_size,
        split=args.split,
        aug=False,
        memcache_init=cfg.use_shm,
        eval_all=False,
        debug=args.debug,
        smap_dir=cfg.smap_dir if hasattr(cfg, "smap_dir") else None,
        max_num_views=6 if not hasattr(cfg, "max_num_views") else cfg.max_num_views,
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data) if cfg.distributed else None
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=cfg.batch_size_val,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collation_fn_smap,
        sampler=val_sampler,
    )

    # ####################### Inference for intermediate ####################### #
    checkpoint_file = args.checkpoint_file
    if os.path.isfile(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage.cuda())
        checkpoint["state_dict"] = {
            k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()
        }
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint["epoch"]))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_file))

    model.eval()
    with torch.no_grad():
        for index, batch_data in enumerate(tqdm(val_loader)):
            scene_id = val_loader.dataset.data_paths[index].split("/")[-1].split(".")[0]
            (
                coords,
                feat,
                label_3d,
                feat_3d,
                mask,
                camera_id_mask,
                img_feats,
                voxel_float_coords,
            ) = batch_data

            sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
            mask = mask.cuda(non_blocking=True)

            # used for oracle testing
            img_feats = img_feats.cuda(non_blocking=True).to(dtype=torch.float32)

            if args.per_view_strategy == "camera":
                input_dict = {
                    "sinput": sinput,
                    "mask": mask,
                    "camera_id_mask": camera_id_mask,
                }
            elif args.per_view_strategy == "lidar":
                if "nuscenes" in cfg.data_root:
                    camera_id_mask = generate_per_view_lidar_mask(
                        voxel_float_coords.cpu(), dataset="nuscenes", n_views=12
                    )  # mask for 10 frames
                    camera_id_mask = torch.from_numpy(camera_id_mask)
                    camera_id_mask = camera_id_mask[mask]  # mask for last frames
                elif "scannet" in cfg.data_root:
                    camera_id_mask = generate_per_view_lidar_mask(
                        voxel_float_coords.cpu(),
                        dataset="scannet",
                        square_size=0.5,
                        th_vaild_point=150,
                    )  # mask for 10 frames

                input_dict = {
                    "sinput": sinput,
                    "mask": mask,
                    "camera_id_mask": camera_id_mask,
                }
            else:
                raise NotImplementedError

            if "nuscenes" in cfg.data_root:
                output_dict = model(input_dict)
                output_smap = output_dict["output_smap"]
            elif "scannet" in cfg.data_root:
                output_dict = model.net3d(input_dict)
                n_view_total = camera_id_mask.shape[1]
                n_view_step = 20
                output_smap = torch.zeros((1, n_view_total, 512))
                for i in range(0, n_view_total, n_view_step):
                    end_i = min(i + n_view_step, n_view_total)
                    output_dict["camera_id_mask"] = camera_id_mask[:, i:end_i]
                    output_dict = model.attn_layer(output_dict)
                    output_smap[:, i:end_i] = output_dict["output_smap"]

            torch.save(
                {
                    "output_smap": output_smap.cpu(),
                    "img_feats": img_feats.cpu(),
                    # "voxel_float_coords": voxel_float_coords.cpu(),
                },
                os.path.join(args.save_dir_feat, f"{scene_id}.pth"),
            )


if __name__ == "__main__":
    main()
