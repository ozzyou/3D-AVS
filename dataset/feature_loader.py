"""Dataloader for fused point features."""

import copy
import pdb
from glob import glob
from os.path import join
import torch
import numpy as np
import SharedArray as SA

from util.util import pickle_load
from dataset.point_loader import Point3DLoader

cam_locs = [
    "back",
    "back_left",
    "back_right",
    "front",
    "front_left",
    "front_right",
]


class FusedFeatureLoader(Point3DLoader):
    """Dataloader for fused point features."""

    def __init__(
        self,
        datapath_prefix,
        datapath_prefix_feat,
        voxel_size=0.05,
        split="train",
        aug=False,
        memcache_init=False,
        identifier=7791,
        loop=1,
        eval_all=False,
        input_color=False,
        limited_files=None,
        debug=False,
    ):
        super().__init__(
            datapath_prefix=datapath_prefix,
            voxel_size=voxel_size,
            split=split,
            aug=aug,
            memcache_init=memcache_init,
            identifier=identifier,
            loop=loop,
            eval_all=eval_all,
            input_color=input_color,
        )
        self.aug = aug
        self.input_color = (
            input_color  # decide whether we use point color values as input
        )

        # prepare for 3D features
        self.datapath_feat = datapath_prefix_feat

        if limited_files is not None:
            self.data_paths = self.data_paths[:limited_files]  # for debugging
        elif debug:
            self.data_paths = self.data_paths[:10]  # for debugging

        # Precompute the occurances for each scene
        # for training sets, ScanNet and Matterport has 5 each, nuscene 1
        # for evaluation/test sets, all has just one
        if "nuscenes" in self.dataset_name:  # only one file for each scene
            self.list_occur = None
        else:
            self.list_occur = []
            for data_path in self.data_paths:
                if "scannet" in self.dataset_name:
                    scene_name = data_path[:-15].split("/")[-1]
                else:
                    scene_name = data_path[:-4].split("/")[-1]
                    scene_name = data_path[:-4].split("/")[-1]
                file_dirs = glob(join(self.datapath_feat, scene_name + "_*.pt"))
                self.list_occur.append(len(file_dirs))
            # some scenes in matterport have no features at all
            ind = np.where(np.array(self.list_occur) != 0)[0]
            if np.any(np.array(self.list_occur) == 0):
                data_paths, list_occur = [], []
                for i in ind:
                    data_paths.append(self.data_paths[i])
                    list_occur.append(self.list_occur[i])
                self.data_paths = data_paths
                self.list_occur = list_occur

        if len(self.data_paths) == 0:
            raise Exception("0 file is loaded in the feature loader.")

    def __getitem__(self, index_long):

        index = index_long % len(self.data_paths)
        if self.use_shm:
            locs_in = SA.attach(
                "shm://%s_%s_%06d_locs_%08d"
                % (self.dataset_name, self.split, self.identifier, index)
            ).copy()
            feats_in = SA.attach(
                "shm://%s_%s_%06d_feats_%08d"
                % (self.dataset_name, self.split, self.identifier, index)
            ).copy()
            labels_in = SA.attach(
                "shm://%s_%s_%06d_labels_%08d"
                % (self.dataset_name, self.split, self.identifier, index)
            ).copy()
        else:
            locs_in, feats_in, labels_in = torch.load(self.data_paths[index])
            labels_in[labels_in == -100] = 255
            labels_in = labels_in.astype(np.uint8)
            if np.isscalar(feats_in) and feats_in == 0:
                # no color in the input point cloud, e.g nuscenes lidar
                feats_in = np.zeros_like(locs_in)
            else:
                feats_in = (feats_in + 1.0) * 127.5

        # load 3D features
        if self.dataset_name == "scannet_3d":
            scene_name = self.data_paths[index][:-15].split("/")[-1]
        else:
            scene_name = self.data_paths[index][:-4].split("/")[-1]

        if "nuscenes" not in self.dataset_name:
            n_occur = self.list_occur[index]
            if n_occur > 1:
                nn_occur = np.random.randint(n_occur)
            elif n_occur == 1:
                nn_occur = 0
            else:
                raise NotImplementedError

            processed_data = torch.load(
                join(self.datapath_feat, scene_name + "_%d.pt" % (nn_occur))
            )
        else:
            # no repeated file
            processed_data = torch.load(join(self.datapath_feat, scene_name + ".pt"))

        flag_mask_merge = False
        if len(processed_data.keys()) == 2:
            flag_mask_merge = True
            feat_3d, mask_chunk = processed_data["feat"], processed_data["mask_full"]
            if isinstance(
                mask_chunk, np.ndarray
            ):  # if the mask itself is a numpy array
                mask_chunk = torch.from_numpy(mask_chunk)
            mask = copy.deepcopy(mask_chunk)
            if self.split != "train":  # val or test set
                feat_3d_new = torch.zeros(
                    (locs_in.shape[0], feat_3d.shape[1]), dtype=feat_3d.dtype
                )
                feat_3d_new[mask] = feat_3d
                feat_3d = feat_3d_new
                mask_chunk = torch.ones_like(
                    mask_chunk
                )  # every point needs to be evaluted
        elif len(processed_data.keys()) > 2:  # legacy, for old processed features
            feat_3d, mask_visible, mask_chunk = (
                processed_data["feat"],
                processed_data["mask"],
                processed_data["mask_full"],
            )
            mask = torch.zeros(feat_3d.shape[0], dtype=torch.bool)
            mask[mask_visible] = True  # mask out points without feature assigned

        if len(feat_3d.shape) > 2:
            feat_3d = feat_3d[..., 0]

        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in

        # calculate the corresponding point features after voxelization
        if self.split == "train" and flag_mask_merge:
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs_in, feats_in, labels_in, return_ind=True
            )
            vox_ind = torch.from_numpy(vox_ind)
            mask = mask_chunk[vox_ind]  # voxelized visible mask for entire point cloud
            mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
            index1 = -torch.ones(mask_chunk.shape[0], dtype=int)
            index1[mask_ind] = mask_ind

            index1 = index1[vox_ind]
            chunk_ind = index1[index1 != -1]

            index2 = torch.zeros(mask_chunk.shape[0])
            index2[mask_ind] = 1
            index3 = torch.cumsum(index2, dim=0, dtype=int)
            # get the indices of corresponding masked point features after voxelization
            indices = index3[chunk_ind] - 1

            # get the corresponding features after voxelization
            feat_3d = feat_3d[indices]
        elif (
            self.split == "train" and not flag_mask_merge
        ):  # legacy, for old processed features
            feat_3d = feat_3d[mask]  # get features for visible points
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs_in, feats_in, labels_in, return_ind=True
            )
            mask_chunk[mask_chunk.clone()] = mask
            vox_ind = torch.from_numpy(vox_ind)
            mask = mask_chunk[vox_ind]  # voxelized visible mask for entire point clouds
            mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
            index1 = -torch.ones(mask_chunk.shape[0], dtype=int)
            index1[mask_ind] = mask_ind

            index1 = index1[vox_ind]
            chunk_ind = index1[index1 != -1]

            index2 = torch.zeros(mask_chunk.shape[0])
            index2[mask_ind] = 1
            index3 = torch.cumsum(index2, dim=0, dtype=int)
            # get the indices of corresponding masked point features after voxelization
            indices = index3[chunk_ind] - 1

            # get the corresponding features after voxelization
            feat_3d = feat_3d[indices]
        else:
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs[mask_chunk],
                feats_in[mask_chunk],
                labels_in[mask_chunk],
                return_ind=True,
            )
            vox_ind = torch.from_numpy(vox_ind)
            feat_3d = feat_3d[vox_ind]
            mask = mask[vox_ind]

        if self.eval_all:  # during evaluation, no voxelization for GT labels
            labels = labels_in
        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)
        coords = torch.from_numpy(locs).int()
        coords = torch.cat(
            (torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1
        )
        if self.input_color:
            feats = torch.from_numpy(feats).float() / 127.5 - 1.0
        else:
            # hack: directly use color=(1, 1, 1) for all points
            feats = torch.ones(coords.shape[0], 3)
        labels = torch.from_numpy(labels).long()

        if self.eval_all:
            return (
                coords,
                feats,
                labels,
                feat_3d,
                mask,
                torch.from_numpy(inds_reconstruct).long(),
            )
        return coords, feats, labels, feat_3d, mask


def collation_fn(batch):
    """
    :param batch:
    :return:    coords: N x 4 (batch,x,y,z)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)

    """
    coords, feats, labels, feat_3d, mask_chunk = list(zip(*batch))

    for i in range(len(coords)):
        coords[i][:, 0] *= i

    return (
        torch.cat(coords),
        torch.cat(feats),
        torch.cat(labels),
        torch.cat(feat_3d),
        torch.cat(mask_chunk),
    )


def collation_fn_eval_all(batch):
    """
    :param batch:
    :return:    coords: N x 4 (x,y,z,batch)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)
                inds_recons:ON

    """
    coords, feats, labels, feat_3d, mask, inds_recons = list(zip(*batch))
    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return (
        torch.cat(coords),
        torch.cat(feats),
        torch.cat(labels),
        torch.cat(feat_3d),
        torch.cat(mask),
        torch.cat(inds_recons),
    )


class FusedFeatureLoaderSMAP(FusedFeatureLoader):
    def __init__(
        self,
        datapath_prefix,
        datapath_prefix_feat,
        voxel_size=0.05,
        split="train",
        aug=False,
        memcache_init=False,
        identifier=7791,
        loop=1,
        eval_all=False,
        limited_files=None,
        smap_dir=None,
        max_num_views=6,
        debug=False,
    ):
        super(FusedFeatureLoaderSMAP, self).__init__(
            datapath_prefix=datapath_prefix,
            datapath_prefix_feat=datapath_prefix_feat,
            voxel_size=voxel_size,
            split=split,
            aug=aug,
            memcache_init=memcache_init,
            identifier=identifier,
            loop=loop,
            eval_all=eval_all,
            limited_files=limited_files,
            debug=debug,
        )

        self.max_num_views = max_num_views

        # read camera id mask and per-image feature
        self.smap_camera_id_mask = pickle_load(
            join(smap_dir, f"nuscenes_camera_id_mask_{self.split}.pkl")
        )
        self.smap_img_clip_feat = torch.load(
            join(smap_dir, f"cached_img_clip_feature_{self.split}.pt")
        )

    def get_smap_variables(self, index):
        # camera_id_masks: dict {token: camera_id_mask->(K,)}, K is the number of points of
        # key frame. camera_id_mask is an encoded uint8 value.
        camera_id_mask = self.smap_camera_id_mask[self.get_token(index)]
        # Decode it
        torch_camera_id_mask = torch.zeros(camera_id_mask.shape[0], 6, dtype=torch.bool)
        for i in range(len(cam_locs)):
            torch_camera_id_mask[:, i] = torch.from_numpy(camera_id_mask & (1 << i) > 0)

        # img_clip_feat: {
        #     'img_feats': a tensor of shape (N, 512), N is the number of images
        #     'token_names': a list of token names, can be used for verifying indexing
        # }
        token = self.smap_img_clip_feat["token_names"][index * 6].split("_")[0]
        if self.__len__() >= 100:
            # Non-debug mode, use the cached feature
            img_feats = copy.deepcopy(
                self.smap_img_clip_feat["img_feats"][index * 6 : index * 6 + 6]
            )
            assert token == self.get_token(index)
        else:
            # Debug mode, search the token
            tgt_token = self.get_token(index)
            for i, token in enumerate(self.smap_img_clip_feat["token_names"][::6]):
                if token.split("_")[0] == tgt_token:
                    img_feats = copy.deepcopy(
                        self.smap_img_clip_feat["img_feats"][i * 6 : i * 6 + 6]
                    )
                    img_feats = img_feats.unsqueeze(0)
                    break
            assert (
                tgt_token == self.smap_img_clip_feat["token_names"][i * 6].split("_")[0]
            )

        img_feats = img_feats.unsqueeze(0)  # (1, n_views, 512)
        return torch_camera_id_mask, img_feats

    def get_token(self, index):
        return self.data_paths[index].split("/")[-1][:-4]

    def __getitem__(self, index_long):

        index = index_long % len(self.data_paths)
        if self.use_shm:
            locs_in = SA.attach(
                "shm://%s_%s_%06d_locs_%08d"
                % (self.dataset_name, self.split, self.identifier, index)
            ).copy()
            feats_in = SA.attach(
                "shm://%s_%s_%06d_feats_%08d"
                % (self.dataset_name, self.split, self.identifier, index)
            ).copy()
            labels_in = SA.attach(
                "shm://%s_%s_%06d_labels_%08d"
                % (self.dataset_name, self.split, self.identifier, index)
            ).copy()
        else:
            locs_in, feats_in, labels_in = torch.load(self.data_paths[index])
            labels_in[labels_in == -100] = 255
            labels_in = labels_in.astype(np.uint8)
            if np.isscalar(feats_in) and feats_in == 0:
                # no color in the input point cloud, e.g nuscenes lidar
                feats_in = np.zeros_like(locs_in)
            else:
                feats_in = (feats_in + 1.0) * 127.5

            torch_camera_id_mask, img_feats = self.get_smap_variables(index)

        # load 3D features
        if self.dataset_name == "scannet_3d":
            scene_name = self.data_paths[index][:-15].split("/")[-1]
        else:
            scene_name = self.data_paths[index][:-4].split("/")[-1]

        if "nuscenes" not in self.dataset_name:
            n_occur = self.list_occur[index]
            if n_occur > 1:
                nn_occur = np.random.randint(n_occur)
            elif n_occur == 1:
                nn_occur = 0
            else:
                raise NotImplementedError

            processed_data = torch.load(
                join(self.datapath_feat, scene_name + "_%d.pt" % (nn_occur))
            )
        else:
            # no repeated file
            processed_data = torch.load(join(self.datapath_feat, scene_name + ".pt"))

        flag_mask_merge = False
        if len(processed_data.keys()) == 2:
            flag_mask_merge = True
            feat_3d, mask_chunk = processed_data["feat"], processed_data["mask_full"]
            if isinstance(
                mask_chunk, np.ndarray
            ):  # if the mask itself is a numpy array
                mask_chunk = torch.from_numpy(mask_chunk)
            mask = copy.deepcopy(mask_chunk)
        elif len(processed_data.keys()) > 2:  # legacy, for old processed features
            feat_3d, mask_visible, mask_chunk = (
                processed_data["feat"],
                processed_data["mask"],
                processed_data["mask_full"],
            )
            mask = torch.zeros(feat_3d.shape[0], dtype=torch.bool)
            mask[mask_visible] = True  # mask out points without feature assigned

        if len(feat_3d.shape) > 2:
            feat_3d = feat_3d[..., 0]

        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in

        # calculate the corresponding point features after voxelization
        if flag_mask_merge:
            # if self.split == "train" and flag_mask_merge:
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs_in, feats_in, labels_in, return_ind=True
            )
            vox_ind = torch.from_numpy(vox_ind)
            mask = mask_chunk[vox_ind]  # voxelized visible mask for entire point cloud
            mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
            index1 = -torch.ones(mask_chunk.shape[0], dtype=int)
            index1[mask_ind] = mask_ind

            index1 = index1[vox_ind]
            chunk_ind = index1[index1 != -1]

            index2 = torch.zeros(mask_chunk.shape[0])
            index2[mask_ind] = 1
            index3 = torch.cumsum(index2, dim=0, dtype=int)
            # get the indices of corresponding masked point features after voxelization
            indices = index3[chunk_ind] - 1

            # get the corresponding features after voxelization
            feat_3d = feat_3d[indices]
            voxel_float_coords = torch.from_numpy(locs_in[vox_ind]).float()
            torch_camera_id_mask = torch_camera_id_mask[indices]
            torch_camera_id_mask = torch_camera_id_mask.to(dtype=torch.bool)

        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)
        coords = torch.from_numpy(locs).int()
        coords = torch.cat(
            (torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1
        )
        if self.input_color:
            feats = torch.from_numpy(feats).float() / 127.5 - 1.0
        else:
            # hack: directly use color=(1, 1, 1) for all points
            feats = torch.ones(coords.shape[0], 3)
        labels = torch.from_numpy(labels).long()

        return (
            coords,
            feats,
            labels,
            feat_3d,
            mask,
            torch_camera_id_mask,
            img_feats,
            voxel_float_coords,
        )


def collation_fn_smap(batch):
    """
    :param batch:
    :return:    coords: N x 4 (batch,x,y,z)
                feats:  N x 3 (r,g,b, if available, otherwise all 1)
                labels: N
                feats_3d: N_v x 768  # target features. N_v is the number of voxels of last frame
                mask_chunk: N  # boolean, N -> N_v
                camera_id_mask: N_v x 6  # encoded camera id
                img_feats: (bsx6) x 512  # image features

    """
    (
        coords,
        feats,
        labels,
        feat_3d,
        mask_chunk,
        camera_id_mask,
        img_feats,
        voxel_float_coords,
    ) = list(zip(*batch))

    for i in range(len(coords)):
        coords[i][:, 0] *= i

    return (
        torch.cat(coords),
        torch.cat(feats),
        torch.cat(labels),
        torch.cat(feat_3d),
        torch.cat(mask_chunk),
        torch.cat(camera_id_mask),
        torch.cat(img_feats),
        torch.cat(voxel_float_coords),
    )
