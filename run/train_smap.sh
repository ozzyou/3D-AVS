#!/bin/sh

PYTHON="your_conda_path/envs/3d-avs/bin/python"
export OMP_NUM_THREADS=1

set -x

exp_root_dir="output"
exp_name="smap_nuscenes"
exp_dir=${exp_root_dir}/${exp_name}

config="config/nuscenes/ours_openseg_smap.yaml"

mkdir -p ${exp_dir}

export PYTHONPATH=.
$PYTHON -u run/train_smap.py \
  --config=${config} \
  --exp_name ${exp_name} \
  --wandb_project ${wandb_project} \
  save_path ${exp_dir} \
  2>&1 | tee -a ${exp_dir}/distill-$(date +"%Y%m%d_%H%M").log