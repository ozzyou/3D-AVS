#!/bin/sh

PYTHON_3DAVS="your_conda_path/envs/3davs/bin/python"
PYTHON_CLIPCAP="your_conda_path/envs/clipcap/bin/python"

set -x
export PYTHONPATH=.

tmp_dir_feat="./tmp/point_captioner_feat"
output_dir="./tmp/point_captioner_output"

mkdir -p ${tmp_dir_feat}
mkdir -p ${output_dir}

$PYTHON_3DAVS point_captioner/infer_features.py \
--checkpoint_file ./ckpt/smap_model_epoch_20.pth.tar \
--save_dir_feat ${tmp_dir_feat} \
--cfg_path ./config/nuscenes/ours_openseg_smap_pretrained.yaml

$PYTHON_CLIPCAP point_captioner/decode_features.py \
--output_path ${output_dir}
