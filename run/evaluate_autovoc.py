import os
import pdb
import random
import numpy as np
import logging
import argparse
import urllib
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from util import metric
from torch.utils import model_zoo

from MinkowskiEngine import SparseTensor
from util import config
from tools_func import write_json_file
from util.util import (
    export_pointcloud,
    get_palette,
    convert_labels_with_palette,
    extract_text_feature,
    load_text_feature,
    visualize_labels,
    read_json,
    concat_fixed_voc_and_autovoc_labels,
    get_autovoc_w2w_mapper,
    map_autovoc2nus16,
    map_autovoc2fv,
    get_unique_autovoc_labelset,
    my_load_state_dict,
    merge_queries_from_multi_views,
    AverageMeter,
    max_cosine_similarity,
    merge_str_lists,
)
from tqdm import tqdm
from run.distill import get_model
from datetime import datetime


from evaluate import get_parser, get_logger, is_url
from dataset.label_constants import *


def replace_keys(dict, mapping_oldkey2newkey):
    """Replace old_key with new_key in dict."""
    for old_key, new_key in mapping_oldkey2newkey.items():
        old_key = old_key + ".jpg"
        dict[new_key] = dict.pop(old_key)
    return dict


def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0
    )


def read_and_merge_autovoc(json_path):
    # 2024/09/05: Deprecated.
    # xxx_vocabulary.json will be processed before being calling by this file.

    # 1. Read autovoc data from json file
    # 2. Merge queries from 6-view images to one key-value pair, namely {token: list of queries}
    # 3. Clean autovoc labels

    assert "json" in json_path
    # if labelset_name is a json file, there is auto-voc data in the file
    # otherwise, it is a labelset name
    # json_path = os.path.join("data", "nuscenes", "selfseg_subset_words", json_path)
    # filepath_name2token = "./data/nuscenes/selfseg_subset_words/name2token.json"

    autovoc_data = read_json(json_path)  # raw data read from json file
    # change newer file (since Mar. 25) to older format
    key_0 = list(autovoc_data.keys())[0]
    if isinstance(autovoc_data[key_0][0], list):
        # this part is for Mar. 25 versionl, namely subset2. It has a different format
        # {token: [[idx0, query0], [idx1, query1], ...]}
        # transform this format to default format
        autovoc_data = {k: [pair[1] for pair in v] for k, v in autovoc_data.items()}
    # merge queries from different views and remove duplicate queries for each token
    dict_token2queries = merge_queries_from_multi_views(autovoc_data)

    return dict_token2queries


def precompute_text_related_properties(labelset_name):
    """pre-compute text features, labelset, palette, and mapper.
    @param labelset_name: "nuscenes_3d" or a path to autovoc json file
    @return:
        - text_features: torch.Tensor or str "text_features_saved"
        - labelset: list of strings, each string is a label/noun
        - mapper: map full indexes to nuscenes_16 + 'unlabelled' for closed-set evaluation
        - palette: list of RGB values
        - dict_label2idx: a dictionary that maps all labels to index
        - dict_token2queries: a dictionary that lookds up queries by token
    """

    """
    Clarify these terms/variables:
        - labelset: list of strings, each string is a label/noun
        - filename: the original file name of an image/point cloud, e.g. n015-2018-09-25-11-10-38+0800__CAM_FRONT__1537845225262460.jpg
        - token: a string that is a unique identifier for an image/point cloud
        - query: a string indicates a semantic category, e.g. "car"
        - dict_A2B: a dictionary that maps A to B
    """

    mapper = None
    dict_token2queries = None
    which_autovoc_labelset = args.get(
        "autovoc_labelset", "nuscenes_details"
    )  # usually defined in config file
    if labelset_name == "scannet_3d":
        labelset = list(SCANNET_LABELS_20)
        labelset[-1] = "other"  # change 'other furniture' to 'other'
        palette = get_palette(colormap="scannet")
    elif which_autovoc_labelset == "nuscenes_16":
        # Not using nuscenes_16 because some of them are ambiguous.
        # Therefore, using nuscenes_details which is defined by OpenScene
        labelset = list(NUSCENES_LABELS_16)
        palette = get_palette(colormap="nuscenes16")
    elif which_autovoc_labelset == "nuscenes_details":
        # need a mapper to map nuscenes_details to nuscenes_16
        labelset = list(NUSCENES_LABELS_DETAILS)
        palette = get_palette(colormap="nuscenes16")
        mapper = torch.tensor(MAPPING_NUSCENES_DETAILS, dtype=int)
    elif (
        which_autovoc_labelset == "nuscenes_details_plus_autovoc"
        or which_autovoc_labelset == "scannet_plus_autovoc"
        or which_autovoc_labelset == "scannet200_plus_autovoc"
    ):

        # 0. read auto voc data and return dict_token2queries and autovoc_labelset
        # dict_token2queries = read_and_merge_autovoc(json_path=labelset_name)
        dict_token2queries = read_json(labelset_name)
        # concat, unique, and order all labels
        autovoc_labelset = get_unique_autovoc_labelset(dict_token2queries)

        # 1. concat nuscenes_labels_details and autovoc_labels
        # 2. take unique labels and keep the order as [nuscenes_labels_details, autovoc_labels]
        if "nuscenes" in which_autovoc_labelset:
            dataset_name = "nuscenes"
            prefix_labelset = NUSCENES_LABELS_DETAILS
        elif "scannet200" in which_autovoc_labelset:
            dataset_name = "scannet200"
            prefix_labelset = SCANNET_LABELS_200
        elif "scannet" in which_autovoc_labelset:
            dataset_name = "scannet"
            prefix_labelset = SCANNET_LABELS_20
        else:
            raise ValueError(f"Unknown dataset name: {which_autovoc_labelset}")
        labelset = concat_fixed_voc_and_autovoc_labels(list(prefix_labelset), autovoc_labelset)

        # 3. create the mapper
        #   3.1 autovoc (variable length) -> nus-details (43)
        lave_mapping_path = args.get("lave_mapping_path", None)
        len_prefix = len(prefix_labelset)
        if lave_mapping_path is not None:
            av2fv_w2w = get_autovoc_w2w_mapper(labelset[len_prefix:], len_prefix, lave_mapping_path)
            # autovoc2nus_details: ['person', 'void', 'building', ...]
            if dataset_name == "nuscenes":
                #   3.2 (nuscenes) nus-details (43) -> nus eval (16)
                if "blip3" in args.autovoc_path_merged:
                    autovoc2nus16 = map_autovoc2nus16(
                        av2fv_w2w,
                        NUSCENES_LABELS_DETAILS_BLIP3,
                        MAPPING_NUSCENES_DETAILS_BLIP3,
                        ignored_ix=len(labelset),
                    )
                    # autovoc2nus16: [6, 16, 14, ...]
                else:
                    autovoc2nus16 = map_autovoc2nus16(
                        av2fv_w2w,
                        NUSCENES_LABELS_DETAILS,
                        MAPPING_NUSCENES_DETAILS,
                        ignored_ix=len(labelset),
                    )
                    # autovoc2nus16: [6, 16, 14, ...]
                mapper = list(MAPPING_NUSCENES_DETAILS) + autovoc2nus16
            else:
                #   3.2 (scannet) 20 + N -> 20
                av2fv = map_autovoc2fv(av2fv_w2w, prefix_labelset, ignored_ix=len(labelset))
                mapper = list(np.arange(len_prefix)) + av2fv
            mapper = torch.tensor(mapper, dtype=int)

        ## delete words that mapping to void
        if args.get("delete_words_mapping_to_void", False):
            auto2predefined = read_json(lave_mapping_path)
            for token in dict_token2queries:
                new_queries = []
                for q in dict_token2queries[token]:
                    if auto2predefined[q] != "void" or auto2predefined[q] != "background":
                        new_queries.append(q)
                dict_token2queries[token] = new_queries

        if dataset_name == "nuscenes":
            # 16+1 RGB values (=> 17x3 = 51). the 17th color is for unlabeled, hence move it to the end
            palette_fv = get_palette(num_cls=16, colormap="nuscenes16")
            len_prefix = 16
        elif dataset_name == "scannet200":
            palette_fv = get_palette(num_cls=len_prefix, colormap="scannet200")
        elif dataset_name == "scannet":
            palette_fv = get_palette(num_cls=len_prefix, colormap="scannet")
        palette = get_palette(num_cls=len(labelset) - len_prefix, colormap="varialbe_length")
        palette = tuple(
            [0, 0, 0] + list(palette_fv)[: len_prefix * 3] + list(palette) + list(palette_fv[-3:])
        )
    else:
        raise NotImplementedError

    if args.get("include_predefined_categories", False):
        for token, queries in dict_token2queries.items():
            dict_token2queries[token] = merge_str_lists(list(prefix_labelset), queries)

    # palette = get_palette(num_cls=len(labelset), colormap='varialbe_length')
    if labelset != "scannet_3d":
        dict_label2idx = {label: idx for idx, label in enumerate(labelset)}

    # for token in dict_token2queries:
    #     for q in dict_token2queries[token]:
    #         if mapper[dict_label2idx[q]] == 16:
    #             print(f"Warning: {q} is mapped to void")

    # len(nuscenes_details_labelset) = 43
    text_features = extract_text_feature(labelset, args, sandl_check=len(labelset) > 43)
    labelset = tuple(
        list(labelset)
        + [
            "unlabeled",
        ]
    )

    return text_features, labelset, mapper, palette, dict_label2idx, dict_token2queries


def main():
    """Main function."""

    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.test_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    print(
        "torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s"
        % (
            torch.__version__,
            torch.version.cuda,
            torch.backends.cudnn.version(),
            torch.backends.cudnn.enabled,
        )
    )

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.test_gpu)
    if len(args.test_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    # By default we do not use shared memory for evaluation
    if not hasattr(args, "use_shm"):
        args.use_shm = False
    if args.use_shm:
        if args.multiprocessing_distributed:
            args.world_size = args.ngpus_per_node * args.world_size
            mp.spawn(
                main_worker,
                nprocs=args.ngpus_per_node,
                args=(args.ngpus_per_node, args),
            )
    else:
        main_worker(args.test_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    model = get_model(args)
    if main_process():
        global logger
        logger = get_logger()
        logger.info(args)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
        args.test_workers = int(args.test_workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    if args.feature_type == "fusion":
        pass  # do not need to load weight
    elif is_url(args.model_path):  # load from url
        checkpoint = model_zoo.load_url(args.model_path, progress=True)
        checkpoint["state_dict"] = {
            k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()
        }
        model.load_state_dict(checkpoint["state_dict"], strict=True)

    elif args.model_path is not None and os.path.isfile(args.model_path):
        # load from directory
        if main_process():
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
        model = my_load_state_dict(model, checkpoint, strict=True, logger=logger)

        if main_process():
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint["epoch"])
            )
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    # ####################### Data Loader ####################### #
    if not hasattr(args, "input_color"):
        # by default we do not use the point color as input
        args.input_color = False

    from dataset.feature_loader import FusedFeatureLoader, collation_fn_eval_all

    val_data = FusedFeatureLoader(
        datapath_prefix=args.data_root,
        datapath_prefix_feat=args.data_root_2d_fused_feature,
        voxel_size=args.voxel_size,
        split=args.split,
        aug=False,
        memcache_init=args.use_shm,
        eval_all=True,
        identifier=6797,
        input_color=args.input_color,
        # limited_files=20,
    )
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collation_fn_eval_all,
        sampler=val_sampler,
    )

    if "scannet200" in args.data_root:
        dataset_name = "scannet200"
    else:
        dataset_name = val_data.dataset_name
    args.tpss_file_name = f"{dataset_name}_{args.voc_src}"
    if args.voc_src == "image":
        args.autovoc_path = args.autovoc_path_img
        # args.lave_mapping_path = (
        #     args.lave_mapping_path_img if hasattr(args, "lave_mapping_path_img") else None
        # )
    elif args.voc_src == "point":
        args.autovoc_path = args.autovoc_path_pt
        # args.lave_mapping_path = (
        #     args.lave_mapping_path_pt if hasattr(args, "lave_mapping_path_pt") else None
        # )
    elif args.voc_src == "merged":
        args.autovoc_path = args.autovoc_path_merged
        # args.lave_mapping_path = (
        #     args.lave_mapping_path_merged if hasattr(args, "lave_mapping_path_merged") else None
        # )
    else:
        raise NotImplementedError

    # ####################### Test ####################### #
    labelset_name = args.data_root.split("/")[-1]
    if hasattr(args, "labelset"):
        # if the labelset is specified
        labelset_name = args.labelset
    if hasattr(args, "autovoc_path") and args.autovoc_path is not None:
        if os.path.exists(args.autovoc_path):
            labelset_name = args.autovoc_path
        else:
            print(
                f"Warning: {args.autovoc_path} does not exist. Use labelset_name = {labelset_name} instead."
            )

    evaluate(model, val_loader, labelset_name, dataset_name)


def evaluate(model, val_data_loader, labelset_name="scannet_3d", dataset_name="nuscenes_3d"):
    """Evaluate our OpenScene model."""
    assert args.test_batch_size == 1, "Only support test batch size 1, otherwise performance drop"

    torch.backends.cudnn.enabled = False

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)

    if args.save_feature_as_numpy:  # save point features to folder
        out_root = os.path.commonprefix([args.save_folder, args.model_path])
        saved_feature_folder = os.path.join(out_root, "saved_feature")
        os.makedirs(saved_feature_folder, exist_ok=True)

    # short hands
    tmp_save_folder = "./tmp/"
    save_folder = args.save_folder
    feature_type = args.feature_type
    eval_iou = False
    if hasattr(args, "eval_iou"):
        eval_iou = args.eval_iou
    mark_no_feature_to_unknown = False
    if (
        hasattr(args, "mark_no_feature_to_unknown")
        and args.mark_no_feature_to_unknown
        and feature_type == "fusion"
    ):
        # some points do not have 2D features from 2D feature fusion. Directly assign 'unknown' label to those points during inference
        mark_no_feature_to_unknown = True
    vis_input = False
    if hasattr(args, "vis_input") and args.vis_input:
        vis_input = True
    vis_pred = False
    if hasattr(args, "vis_pred") and args.vis_pred:
        vis_pred = True
    vis_gt = False
    if hasattr(args, "vis_gt") and args.vis_gt:
        vis_gt = True
    eval_tsc = False
    if hasattr(args, "eval_tsc") and args.eval_tsc:
        eval_tsc = True
        TSC = AverageMeter()

    all_text_features, labelset, mapper, palette, dict_label2idx, dict_token2queries = (
        precompute_text_related_properties(labelset_name)
    )
    num_labels = len(labelset)

    if dict_token2queries is not None:
        max_num_label_per_scene = max([len(v) for v in dict_token2queries.values()])
    else:
        max_num_label_per_scene = len(labelset)

    if isinstance(all_text_features, str) and all_text_features == "text_features_saved":
        all_text_features = load_text_feature(labelset)
        all_text_features = np.array(all_text_features)
        all_text_features = torch.tensor(all_text_features).cuda()

    unique_logits_pred = []
    with torch.no_grad():
        model.eval()
        store = 0.0
        tsc_score = {}
        for rep_i in range(args.test_repeats):
            sparse_preds, gts = [], []
            sparse2dense_mappings = []
            val_data_loader.dataset.offset = rep_i
            if main_process():
                logger.info("\nEvaluation {} out of {} runs...\n".format(rep_i + 1, args.test_repeats))

            # repeat the evaluation process
            # to account for the randomness in MinkowskiNet voxelization
            if rep_i > 0:
                seed = np.random.randint(10000)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            if mark_no_feature_to_unknown:
                masks = []

            cnt_valid_class_per_scene = 0.0
            for i, (coords, feat, label, feat_3d, mask, inds_reverse, _) in enumerate(
                tqdm(val_data_loader)
            ):
                token = val_data_loader.dataset.get_token(i)
                if token.endswith("_vh_clean_2") and args.voc_src != "point":
                    token = token[: -len("_vh_clean_2")]
                if dict_token2queries is not None:
                    # pre-defined vocabulary
                    query_index = [dict_label2idx[query] for query in dict_token2queries[token]]
                    query_index = np.array(query_index)
                    query_index_tensor = torch.tensor(query_index).cuda(non_blocking=True)
                    text_features = all_text_features[query_index_tensor]
                else:
                    text_features = all_text_features

                sinput = SparseTensor(feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
                coords = coords[inds_reverse, :]
                pcl = coords[:, 1:].cpu().numpy()

                if feature_type == "distill":
                    output_dict = model({"sinput": sinput})
                    predictions = output_dict["output_clip"]
                    predictions = predictions[inds_reverse, :]
                    pred = predictions.half() @ text_features.t()
                    logits_pred = torch.max(pred, 1)[1].cpu()
                elif feature_type == "fusion":
                    predictions = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
                    pred = predictions.half() @ text_features.t()
                    logits_pred = torch.max(pred, 1)[1].detach().cpu()
                    if mark_no_feature_to_unknown:
                        # some points do not have 2D features from 2D feature fusion.
                        # Directly assign 'unknown' label to those points during inference.
                        logits_pred[~mask[inds_reverse]] = len(labelset) - 1

                elif feature_type == "ensemble":
                    feat_fuse = feat_3d.cuda(non_blocking=True)[inds_reverse, :]
                    # pred_fusion = feat_fuse.half() @ text_features.t()
                    pred_fusion = (
                        feat_fuse / (feat_fuse.norm(dim=-1, keepdim=True) + 1e-5)
                    ).half() @ text_features.t()

                    # output_dict = model(sinput)
                    output_dict = model({"sinput": sinput})
                    predictions = output_dict["output_clip"]
                    predictions = predictions[inds_reverse, :]
                    # pred_distill = predictions.half() @ text_features.t()
                    pred_distill = (
                        predictions / (predictions.norm(dim=-1, keepdim=True) + 1e-5)
                    ).half() @ text_features.t()

                    # logits_distill = torch.max(pred_distill, 1)[1].detach().cpu()
                    # mask_ensem = pred_distill<pred_fusion # confidence-based ensemble
                    # pred = pred_distill
                    # pred[mask_ensem] = pred_fusion[mask_ensem]
                    # logits_pred = torch.max(pred, 1)[1].detach().cpu()

                    feat_ensemble = predictions.clone().half()
                    mask_ = pred_distill.max(dim=-1)[0] < pred_fusion.max(dim=-1)[0]
                    feat_ensemble[mask_] = feat_fuse[mask_]
                    pred = feat_ensemble @ text_features.t()
                    logits_pred = torch.max(pred, 1)[1].detach().cpu()

                    predictions = feat_ensemble  # if we need to save the features
                else:
                    raise NotImplementedError

                if eval_tsc:
                    scores = max_cosine_similarity(feat_ensemble, text_features, logits_pred)
                    avg_score = scores.mean().item()
                    TSC.update(avg_score, len(scores))
                    tsc_score[token] = avg_score
                    # print('scores: ', scores.mean().item(), 'TSC: ', TSC.avg)

                if dict_token2queries is not None:
                    # pre-defined vocabulary
                    logits_pred = np.array(query_index)[logits_pred.cpu().numpy()]
                    # reverse the label mapping from sparse to dense
                    # dense_pred = np.zeros((len(pred), num_labels), dtype=np.float32)
                    # dense_pred[:, query_index_tensor.cpu().numpy()] = pred.cpu()
                else:
                    logits_pred = logits_pred.cpu().numpy()
                    # dense_pred = pred.cpu().numpy()

                if args.save_feature_as_numpy:
                    scene_name = val_data_loader.dataset.data_paths[i].split("/")[-1].split(".pth")[0]
                    np.save(
                        os.path.join(
                            saved_feature_folder,
                            "{}_openscene_feat_{}.npy".format(scene_name, feature_type),
                        ),
                        predictions.cpu().numpy(),
                    )

                # Visualize the input, predictions and GT

                # special case for nuScenes, where points for evaluation are only a subset of input
                if "nuscenes" in labelset_name or "json" in labelset_name or "scannet" in labelset_name:
                    label_mask = label != 255  # label 255 is the unannotathed points in the previous 9
                    # frames
                    label = label[label_mask]
                    logits_pred = logits_pred[label_mask]
                    pred = pred[label_mask]
                    pcl = torch.load(val_data_loader.dataset.data_paths[i])[0][label_mask]

                unique_logits_pred.extend(list(np.unique(logits_pred)))
                cnt_valid_class_per_scene += len(np.unique(logits_pred))

                if vis_input:
                    input_color = torch.load(val_data_loader.dataset.data_paths[i])[1]
                    input_color = input_color[label_mask, :]
                    export_pointcloud(
                        os.path.join(save_folder, "{}_{}_input.ply".format(token, feature_type)),
                        pcl,
                        colors=(input_color + 1) / 2,
                    )

                if vis_pred:
                    pred_label_color = convert_labels_with_palette(logits_pred, palette, auto_voc=True)
                    ### Special visualizaiton for "0a5355" scene
                    # if "0a5355" in val_data_loader.dataset.data_paths[i]:
                    #     from util.util import manually_set_color_for_0a5355
                    #
                    #     pred_label_color[label == 14] = np.array([192, 192, 192], dtype=float) / 255.0
                    #     pred_label_color, palette = manually_set_color_for_0a5355(
                    #         pred_label_color, palette, labelset, logits_pred
                    #     )
                    #     pred_label_color[label != 14] = np.ones((1, 3), dtype=float) / 2
                    export_pointcloud(
                        # os.path.join(save_folder, "{}_{}_op.ply".format(token, feature_type)),
                        os.path.join(save_folder, "{}_{}.ply".format(token, feature_type)),
                        pcl,
                        colors=pred_label_color,
                    )
                    visualize_labels(
                        list(np.unique(logits_pred)),
                        labelset,
                        palette,
                        os.path.join(save_folder, "{}_labels_pred.png".format(token)),
                        # os.path.join(save_folder, "{}_labels_pred_op.png".format(token)),
                        ncol=4,
                    )

                    # if mapper is not None:
                    #     pred_label_color = convert_labels_with_palette(
                    #         mapper[logits_pred].numpy(), palette, auto_voc=True
                    #     )
                    #     export_pointcloud(
                    #         os.path.join(
                    #             save_folder,
                    #             "{}_{}_mapped.ply".format(token, feature_type),
                    #         ),
                    #         pcl,
                    #         colors=pred_label_color,
                    #     )
                    #     visualize_labels(
                    #         list(mapper[np.unique(logits_pred)]),
                    #         NUSCENES_LABELS_16 + labelset[16:-1] + ("void",),  # labelset
                    #         # is just for placeholder so that I do not change visualize_labels
                    #         palette,
                    #         os.path.join(save_folder, "{}_labels_pred_mapped.png".format(token)),
                    #         ncol=5,
                    #     )

                # Visualize GT labels
                if vis_gt:
                    # for points not evaluating
                    label[label == 255] = len(labelset) - 1
                    gt_label_color = convert_labels_with_palette(label.cpu().numpy(), palette)
                    export_pointcloud(
                        os.path.join(save_folder, "{}_gt.ply".format(token)),
                        pcl,
                        colors=gt_label_color,
                    )
                    visualize_labels(
                        list(np.unique(label.cpu().numpy())),
                        NUSCENES_LABELS_16 if dataset_name == "nuscenes" else SCANNET_LABELS_20,
                        palette,
                        os.path.join(save_folder, "{}_labels_gt.png".format(token)),
                        ncol=1,
                    )

                    # Special case for "0a535567" for visualiztion in teaser image
                    # gt_label_color = np.ones((len(label), 3), dtype=float) / 2
                    # gt_label_color[label == 14] = np.array([256, 64, 00], dtype=float) / 255
                    # export_pointcloud(
                    #     os.path.join(save_folder, "{}_gt.ply".format(token)),
                    #     pcl,
                    #     colors=gt_label_color,
                    # )
                    # visualize_labels(
                    #     list(np.unique(label.cpu().numpy())),
                    #     NUSCENES_LABELS_16,
                    #     palette,
                    #     os.path.join(save_folder, "{}_labels_gt.pdf".format(token)),
                    #     ncol=1,
                    # )

                    # if 'nuscenes' in labelset_name:
                    #     # Weijie: seems only useful for one-row legend
                    #     all_digits = np.unique(np.concatenate([np.unique(mapper[logits_pred].numpy()), np.unique(label)]))
                    #     labelset = list(NUSCENES_LABELS_16)
                    #     labelset[4] = 'construct. vehicle'
                    #     labelset[10] = 'road'
                    #     visualize_labels(list(all_digits), labelset,
                    #         palette, os.path.join(save_folder, '{}_label.png'.format(i)), ncol=all_digits.shape[0])

                if eval_iou:
                    if mark_no_feature_to_unknown:
                        if "nuscenes" in labelset_name:  # special case
                            mask = mask[inds_reverse][label_mask]
                            masks.append(mask)

                            logits_pred = logits_pred[mask]
                            pred = pred[mask]
                        else:
                            masks.append(mask[inds_reverse])

                    if args.test_repeats == 1:
                        # save directly the logits
                        sparse_preds.append(logits_pred)

                        # dirt_w = max_num_label_per_scene - len(query_index)
                        # query_index = np.hstack((query_index, np.zeros((dirt_w))))
                        # query_index = np.repeat(
                        #     np.expand_dims(query_index, axis=0),
                        #     repeats=len(logits_pred),
                        #     axis=0)
                        # sparse2dense_mappings.append(query_index)
                    else:
                        # only save the dot-product results, for repeat prediction
                        dirt_w = max_num_label_per_scene - pred.shape[1]
                        pred = torch.cat(
                            (
                                pred,
                                torch.zeros((len(pred), dirt_w)).to(device=pred.device),
                            ),
                            dim=1,
                        )
                        sparse_preds.append(pred.cpu())

                        if rep_i == 0:
                            if dict_token2queries is not None:
                                dirt_w = max_num_label_per_scene - len(query_index)
                                query_index = np.hstack((query_index, np.zeros((dirt_w))))
                                query_index = np.repeat(
                                    np.expand_dims(query_index, axis=0),
                                    repeats=len(pred),
                                    axis=0,
                                )
                                sparse2dense_mappings.append(query_index)

                    gts.append(label.cpu())

            if eval_tsc:
                logger.info(f"TSC: {TSC.avg}")
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                write_json_file(
                    os.path.join(
                        "./slurm-records/tpss_scores", f"{args.tpss_file_name}" f"_{current_time}.json"
                    ),
                    tsc_score,
                )

            if eval_iou:
                if args.test_repeats > 1:
                    pred = torch.cat(sparse_preds)
                    np.save(
                        tmp_save_folder + "pred_{}.npy".format(rep_i),
                        pred.cpu().numpy(),
                    )
                    del pred
                    if rep_i == 0:
                        gt = torch.cat(gts)
                        np.save(tmp_save_folder + "gt.npy", gt.cpu().numpy())
                        # sparse_preds = np.hstack(sparse_preds)
                        # sparse_pred = torch.Tensor(sparse_preds).to(dtype=int)
                        sparse2dense_mapping = np.vstack(sparse2dense_mappings)
                        sparse2dense_mapping = torch.Tensor(sparse2dense_mapping).to(dtype=int)
                        np.save(
                            tmp_save_folder + "sparse2dense_mapping.npy",
                            sparse2dense_mapping.cpu().numpy(),
                        )
                        del sparse2dense_mapping, sparse2dense_mappings
                        if mapper is not None:
                            np.save(tmp_save_folder + "mapper.npy", mapper.cpu().numpy())
                        if mark_no_feature_to_unknown:
                            mask = torch.cat(masks).cpu().numpy()
                            np.save(tmp_save_folder + "mask.npy", mask)
                            del mask

                else:
                    gt = torch.cat(gts)
                    # sparse_pred = torch.cat(sparse_preds)
                    sparse_pred = np.hstack(sparse_preds)

                    if mapper is not None:
                        pred_logit = mapper[sparse_pred]

                    if mark_no_feature_to_unknown:
                        mask = torch.cat(masks).cpu().numpy()
                        pred_logit[~mask] = 256

                    logger.info(f"unique mapped logits: {torch.unique(pred_logit)}")
                    pred_logit[pred_logit == torch.max(mapper)] = 256

                    current_iou = metric.evaluate(
                        pred_logit, gt.numpy(), dataset=dataset_name, stdout=True
                    )
                    logger.info("Current IoU: {:.5f}".format(current_iou))

            logger.info(f"Avg # classes per scene: {cnt_valid_class_per_scene / len(val_data_loader)}")

    unique_logits_pred = np.unique(unique_logits_pred)
    logger.info(f"# classes in total: {len(unique_logits_pred)}")
    with open("tmp/unique_logits_pred.pkl", "wb") as f:
        pickle.dump(
            {
                "unique_logits_pred": unique_logits_pred,
                "labelset": labelset,
            },
            f,
        )
    # with open("tmp/unique_logits_pred.pkl", "r") as f:
    #     data = pickle.load(f)
    #     unique_logits_pred = data["unique_logits_pred"]
    #     labelset = data["labelset"]


if __name__ == "__main__":
    main()
