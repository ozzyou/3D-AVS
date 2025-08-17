import os
import pdb
import time
import random
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils import model_zoo
import wandb
from tensorboardX import SummaryWriter

import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor
from util import config
from util.util import (
    AverageMeter,
    LossMeters,
    poly_learning_rate,
    save_checkpoint,
    export_pointcloud,
    get_palette,
    convert_labels_with_palette,
    extract_clip_feature,
    latest_file,
)
from dataset.label_constants import *
from dataset.feature_loader import (
    FusedFeatureLoaderSMAP,
    collation_fn_smap,
)
from models.disnet import DisNet as Model
from tqdm import tqdm


best_iou = 0.0
best_loss = 1e10


def worker_init_fn(worker_id):
    """Worker initialization."""
    random.seed(time.time() + worker_id)


def merge_args_with_config(args, cfg):
    # Get dictionary representation of args
    args_dict = vars(args)
    # Loop through arguments and update config
    for arg, value in args_dict.items():
        if value is not None:
            # Split argument name by '_' to get nested structure
            keys = arg.split(".")
            if keys[0] == "opts" or keys[0] == "config":
                continue
            # Use the last part as the main key
            main_key = keys[-1]
            # Check if the main key exists in the config
            cfg[main_key] = value

    return cfg


def get_parser():
    """Parse the config file."""

    parser = argparse.ArgumentParser(description="OpenScene 3D distillation.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/scannet/distill_openseg.yaml",
        help="config file",
    )
    parser.add_argument(
        "opts",
        default=None,
        help="see config/scannet/distill_openseg.yaml for all options",
        nargs=argparse.REMAINDER,
    )
    # parser.add_argument(
    #     "--local_rank", type=int, default=-1, metavar="N", help="Local process rank."
    # )  # you need this argument in your scripts for DDP to work
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="wandb project name"
    )
    parser.add_argument("--exp_name", type=str, default=None, help="experiment name")
    args_in = parser.parse_args()
    assert args_in.config is not None
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)
    os.makedirs(cfg.save_path, exist_ok=True)
    model_dir = os.path.join(cfg.save_path, "model")
    result_dir = os.path.join(cfg.save_path, "result")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir + "/last", exist_ok=True)
    os.makedirs(result_dir + "/best", exist_ok=True)

    cfg = merge_args_with_config(args_in, cfg)

    return args_in, cfg


def get_logger():
    """Define logger."""

    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in


def main_process():
    return not cfg.multiprocessing_distributed or (
        cfg.multiprocessing_distributed and cfg.rank % cfg.ngpus_per_node == 0
    )


def main():
    """Main function."""

    args, cfg = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in cfg.train_gpu)
    cudnn.benchmark = True
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

    if cfg.multiprocessing_distributed:
        cfg.world_size = cfg.ngpus_per_node * cfg.world_size
        mp.spawn(main_worker, nprocs=cfg.ngpus_per_node, args=(cfg.ngpus_per_node, cfg))
    else:
        main_worker(cfg.train_gpu, cfg.ngpus_per_node, cfg)


def main_worker(gpu, ngpus_per_node, argss):
    global cfg
    global best_iou
    global best_loss
    cfg = argss

    if cfg.distributed:
        if cfg.multiprocessing_distributed:
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=cfg.dist_backend,
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=cfg.rank,
        )

    use_wandb = False
    if main_process():
        if hasattr(cfg, "wandb_project") and cfg.wandb_project is not None:
            use_wandb = True
            wandb.login()
            wandb.init(project=cfg.wandb_project, name=cfg.exp_name, config=cfg)

    cfg.method = "smap"
    model = get_model(cfg)
    if cfg.distributed:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(cfg.save_path)
        logger.info(cfg)
        logger.info("=> creating model ...")

    # ####################### Optimizer ####################### #
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr)
    cfg.index_split = 0

    # Freeze backbone
    if hasattr(model, "freeze_backbone") and model.freeze_backbone:
        for param in model.net3d.parameters():
            param.requires_grad = False

    if hasattr(cfg, "model_path") and cfg.model_path is not None:
        log_info("=> loading checkpoint '{}'".format(cfg.model_path))
        checkpoint = model_zoo.load_url(cfg.model_path, progress=True)
        checkpoint["state_dict"] = {
            k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()
        }
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    if cfg.distributed:
        torch.cuda.set_device(gpu)
        # cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
        # cfg.batch_size_val = int(cfg.batch_size_val / ngpus_per_node)
        # cfg.workers = int(cfg.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[gpu]
        )
    else:
        model = model.cuda()

    if cfg.resume:
        log_info("=> loading checkpoint '{}'".format(cfg.resume))
        try:
            if os.path.isfile(cfg.resume):
                checkpoint_file = cfg.resume
            elif os.path.isdir(cfg.resume):
                checkpoint_file = latest_file(cfg.resume)
                if checkpoint_file is None:
                    raise FileNotFoundError
            else:
                raise TypeError

            checkpoint = torch.load(
                checkpoint_file, map_location=lambda storage, loc: storage.cuda()
            )
            cfg.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_iou = checkpoint["best_iou"]
            best_loss = checkpoint.get("best_loss", best_loss)
            log_info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    checkpoint_file, checkpoint["epoch"]
                )
            )
        except:
            log_info("=> no checkpoint found at '{}'".format(cfg.resume))

    # ####################### Data Loader ####################### #
    if not hasattr(cfg, "input_color"):
        # by default we do not use the point color as input
        cfg.input_color = False
    train_data = FusedFeatureLoaderSMAP(
        datapath_prefix=cfg.data_root,
        datapath_prefix_feat=cfg.data_root_2d_fused_feature,
        voxel_size=cfg.voxel_size,
        split="train",
        aug=cfg.aug,
        memcache_init=cfg.use_shm,
        loop=cfg.loop,
        # limited_files=20,
        smap_dir=cfg.smap_dir if hasattr(cfg, "smap_dir") else None,
        max_num_views=cfg.max_num_views if hasattr(cfg, "max_num_views") else 6,
    )
    coll_fn = collation_fn_smap
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_data)
        if cfg.distributed
        else None
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=coll_fn,
        worker_init_fn=worker_init_fn,
    )
    if cfg.evaluate:
        val_data = FusedFeatureLoaderSMAP(
            datapath_prefix=cfg.data_root,
            datapath_prefix_feat=cfg.data_root_2d_fused_feature,
            voxel_size=cfg.voxel_size,
            split="val",
            aug=False,
            memcache_init=cfg.use_shm,
            loop=cfg.loop,
            smap_dir=cfg.smap_dir if hasattr(cfg, "smap_dir") else None,
            max_num_views=30,
        )
        func_collation_fn_eval = collation_fn_smap
        val_sampler = (
            torch.utils.data.distributed.DistributedSampler(val_data)
            if cfg.distributed
            else None
        )
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=cfg.batch_size_val,
            shuffle=False,
            num_workers=cfg.workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=func_collation_fn_eval,
            sampler=val_sampler,
        )

    # ####################### Distill ####################### #
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
            if cfg.evaluate:
                val_sampler.set_epoch(epoch)
        loss_dict = distill(train_loader, model, optimizer, epoch, use_wandb, cfg)
        epoch_log = epoch + 1
        add_scalar(loss_dict, epoch_log, use_wandb=use_wandb)

        is_best = False
        if cfg.evaluate and (epoch_log % cfg.eval_freq == 0):
            if main_process():
                logger.info("Start Evaluation! Epoch: {}".format(epoch_log))
            criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda(gpu)
            validate_dict = validate(val_loader, model, criterion)
            add_scalar(validate_dict, epoch_log, use_wandb=use_wandb)
            if main_process():
                logger.info("Evaluation Done! Epoch: {}".format(epoch_log))
                # remember best iou and save checkpoint
                loss_val = validate_dict["val_loss"]
                is_best = loss_val < best_loss
                best_loss = min(best_loss, loss_val)

        if (epoch_log % cfg.save_freq == 0) and main_process():
            save_checkpoint(
                {
                    "epoch": epoch_log,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_iou": best_iou,
                    "best_loss": best_loss,
                },
                is_best,
                os.path.join(cfg.save_path, "model"),
                filename="model_epoch_%d.pth.tar" % epoch_log,
            )

    if main_process():
        writer.close()
        logger.info("==>Training done!\nBest Iou: %.3f" % (best_iou))


def get_model(cfg):
    """Get the 3D model."""
    model = Model(cfg=cfg)
    return model


def add_scalar(info_dict, iter, use_wandb=False):
    if main_process():
        wandb_dict = {}
        for key, value in info_dict.items():
            writer.add_scalar(key, value, iter)
            if use_wandb:
                if "train" in key:
                    split = "train/"
                    key = key.replace("train", "")
                elif "val" in key:
                    split = "val/"
                    key = key.replace("val", "")
                else:
                    split = ""

                key = key.replace("__", "_")
                key = key[:-1] if key[-1] == "_" else key
                key = key[1:] if key[0] == "_" else key

                wandb_dict[split + key] = value

        if use_wandb:
            wandb.log(wandb_dict)


def log_info(info):
    if main_process():
        logger.info(info)


def obtain_text_features_and_palette():
    """obtain the CLIP text feature and palette."""

    if "scannet" in cfg.data_root:
        labelset = list(SCANNET_LABELS_20)
        labelset[-1] = "other"
        palette = get_palette()
        dataset_name = "scannet"
    elif "matterport" in cfg.data_root:
        labelset = list(MATTERPORT_LABELS_21)
        palette = get_palette(colormap="matterport")
        dataset_name = "matterport"
    elif "nuscenes" in cfg.data_root:
        labelset = list(NUSCENES_LABELS_16)
        palette = get_palette(colormap="nuscenes16")
        dataset_name = "nuscenes"

    if not os.path.exists("saved_text_embeddings"):
        os.makedirs("saved_text_embeddings")

    if "openseg" in cfg.feature_2d_extractor:
        model_name = "ViT-L/14@336px"
        postfix = "_768"  # the dimension of CLIP features is 768
    elif "lseg" in cfg.feature_2d_extractor:
        model_name = "ViT-B/32"
        postfix = "_512"  # the dimension of CLIP features is 512
    else:
        raise NotImplementedError

    clip_file_name = "saved_text_embeddings/clip_{}_labels{}.pt".format(
        dataset_name, postfix
    )

    try:  # try to load the pre-saved embedding first
        log_info("Load pre-computed embeddings from {}".format(clip_file_name))
        text_features = torch.load(clip_file_name).cuda()
    except:  # extract CLIP text features and save them
        log_info("Loading pre-computed embeddings failed, re-compute them.")
        text_features = extract_clip_feature(labelset, model_name=model_name)
        torch.save(text_features, clip_file_name)
    # text_features = None

    return text_features, palette


def distill(
    train_loader,
    model,
    optimizer,
    epoch,
    use_wandb=False,
    cfg=None,
):
    """Distillation pipeline."""

    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meter = AverageMeter()  # sum loss
    loss_meters = LossMeters(keys=["train_smap"])

    model.train()
    end = time.time()
    max_iter = cfg.epochs * len(train_loader)

    # start the distillation process
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        losses = distill_one_batch(batch_data, model, cfg)
        loss_meters.update(losses, cfg.batch_size)

        loss = sum(losses.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), cfg.batch_size)
        batch_time.update(time.time() - end)

        # adjust learning rate
        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(
            cfg.base_lr, current_iter, max_iter, power=cfg.power
        )

        for index in range(0, cfg.index_split):
            optimizer.param_groups[index]["lr"] = current_lr
        for index in range(cfg.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]["lr"] = current_lr * 10

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % cfg.print_freq == 0 and main_process():
            log_info(
                "Epoch: [{}/{}][{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Remain {remain_time} "
                "Loss {loss_meter.val:.4f} ".format(
                    epoch + 1,
                    cfg.epochs,
                    i + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    remain_time=remain_time,
                    loss_meter=loss_meter,
                )
            )

        per_batch_info_dict = {
            "loss_train_batch": loss_meter.val,
            "learning_rate": current_lr,
        }
        per_batch_info_dict.update(loss_meters.get_dict_val(key_postfix="_batch"))
        add_scalar(
            per_batch_info_dict,
            current_iter,
            use_wandb=use_wandb,
        )

        end = time.time()

    res_dict = {"train_loss": loss_meter.avg}
    res_dict.update(loss_meters.get_dict_avg())

    return res_dict


def validate(val_loader, model, criterion):
    """Validation."""

    torch.backends.cudnn.enabled = False
    loss_meters = LossMeters(keys=["val_loss", "val_clip", "val_smap"])

    # obtain the CLIP feature
    text_features, _ = obtain_text_features_and_palette()

    with torch.no_grad():
        for batch_data in val_loader:
            input_dict = {}
            (coords, feat, label_3d, feat_3d, mask, camera_id_mask, img_feats, _) = (
                batch_data
            )
            mask = mask.cuda(non_blocking=True).to(dtype=torch.bool)
            camera_id_mask = camera_id_mask.cuda(non_blocking=True).to(dtype=torch.bool)
            input_dict["mask"] = mask
            input_dict["camera_id_mask"] = camera_id_mask

            sinput = SparseTensor(
                feat.cuda(non_blocking=True), coords.cuda(non_blocking=True)
            )

            input_dict["sinput"] = sinput
            output_dict = model(input_dict)

            batch_loss = 0
            img_feats = img_feats.cuda(non_blocking=True).to(dtype=torch.float32)
            output_smap = output_dict["output_smap"]
            loss = torch.nn.MSELoss()(output_smap, img_feats)
            loss_meters.meters["val_smap"].update(loss.item(), cfg.batch_size)
            batch_loss += loss.item()

            loss_meters.meters["val_loss"].update(batch_loss, cfg.batch_size)

    res_dict = {}
    res_dict.update(loss_meters.get_dict_avg())

    return res_dict


def distill_one_batch(batch_data, model, cfg):
    input_dict = {}
    (coords, feat, label_3d, feat_3d, mask, camera_id_mask, img_feats, _) = batch_data
    mask = mask.cuda(non_blocking=True).to(dtype=torch.bool)
    camera_id_mask = camera_id_mask.cuda(non_blocking=True).to(dtype=torch.bool)
    img_feats = img_feats.cuda(non_blocking=True).to(dtype=torch.float32)
    input_dict = {
        "mask": mask,
        "camera_id_mask": camera_id_mask,
    }

    coords[:, 1:4] += (torch.rand(3) * 100).type_as(coords)
    sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))

    input_dict["sinput"] = sinput
    output_dict = model(input_dict)

    losses = {}
    # High-resolution clip loss
    if output_dict.get("output_clip", None) is not None and not cfg.freeze_backbone:
        output_3d = output_dict["output_clip"][mask]
        if hasattr(cfg, "loss_type") and cfg.loss_type == "cosine":
            losses["train_clip"] = (
                1 - torch.nn.CosineSimilarity()(output_3d, feat_3d)
            ).mean()
        elif hasattr(cfg, "loss_type") and cfg.loss_type == "l1":
            losses["train_clip"] = torch.nn.L1Loss()(output_3d, feat_3d)
        else:
            raise NotImplementedError

    if cfg.get("smap_loss", None) is not None:
        if cfg.smap_loss == "mse":
            losses["train_smap"] = torch.nn.MSELoss()(
                output_dict["output_smap"], img_feats
            )

    return losses


def post_epoch_save_visualized_point_cloud(
    coords,
    mask,
    output_3d,
    feat_3d,
    text_features,
    label_3d,
    palette,
    epoch,
):
    # mask_first = coords[mask][:, 0] == 0
    mask_first = coords[mask][:, 0] == 1
    output_3d = output_3d[mask_first]
    feat_3d = feat_3d[mask_first]
    logits_pred = output_3d.half() @ text_features.t()
    logits_img = feat_3d.half() @ text_features.t()
    logits_pred = torch.max(logits_pred, 1)[1].cpu().numpy()
    logits_img = torch.max(logits_img, 1)[1].cpu().numpy()
    mask = mask.cpu().numpy()
    logits_gt = label_3d.numpy()[mask][mask_first.cpu().numpy()]
    logits_gt[logits_gt == 255] = cfg.classes

    pcl = coords[:, 1:].cpu().numpy()

    seg_label_color = convert_labels_with_palette(logits_img, palette)
    pred_label_color = convert_labels_with_palette(logits_pred, palette)
    gt_label_color = convert_labels_with_palette(logits_gt, palette)
    pcl_part = pcl[mask][mask_first.cpu().numpy()]

    export_pointcloud(
        os.path.join(
            cfg.save_path,
            "result",
            "last",
            "{}_{}.ply".format(cfg.feature_2d_extractor, epoch),
        ),
        pcl_part,
        colors=seg_label_color,
    )
    export_pointcloud(
        os.path.join(cfg.save_path, "result", "last", "pred_{}.ply".format(epoch)),
        pcl_part,
        colors=pred_label_color,
    )
    export_pointcloud(
        os.path.join(cfg.save_path, "result", "last", "gt_{}.ply".format(epoch)),
        pcl_part,
        colors=gt_label_color,
    )


if __name__ == "__main__":
    main()
