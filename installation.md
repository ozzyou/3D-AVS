We test this environment with NVIDIA A100 GPUs and Linux RHEL 8 on 18 July 2025.

Note that, we use three separate environments for this project.

- `3davs` (Step 1-4): for segmentation and evaluation.
- `lavis` (Step 5): for image captioner, namely BLIP-related modules.
- `clipcap` (Step 6): for the decoder of point captioner.

Step 1: install PyTorch:

```
conda create -n 3davs python=3.8
conda activate 3davs
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

Step 2: install MinkowskiNet:

For any error in this step, please refer to
their [official installation page](https://github.com/NVIDIA/MinkowskiEngine#installation).

```bash
sudo apt install build-essential python3-dev libopenblas-dev
```

If you do not have sudo right, try the following:

```
conda install openblas-devel -c anaconda
```

And now install MinkowskiNet:

```
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
                           --install-option="--force_cuda" \
                           --install-option="--blas=openblas"
```

If both pip and conda installation **does not** work. You may compile locally. **Make sure you have GPU available**.

```bash
conda install openblas-devel -c anaconda
conda install ninja

cd ..
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine

python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

Step 3: install all the remaining dependencies:

```bash
cd ../3D-AVS
pip install -r requirements.txt
```

Step 4 (optional): if you need to run **multi-view feature fusion** with OpenSeg (especially for your
own dataset), **install tensorflow as follows**:

```bash
pip install tensorflow-gpu==2.5.0
pip install protobuf==3.20  # might be not useful
```

Install tensorflow only when using multi-view feature fusion.
After multi-view feature fusion is done, **uninstall it and recover numpy**:

```
pip uninstall tensorflow
pip uninstall tensorflow-gpu
pip install numpy==1.24.4
pip install typing_extensions==4.7.0
```

Step 5 (for BLIP3): create `lavis` env and install `lavis` library

```
cd ..
conda create -n lavis python=3.8
conda activate lavis
git clone git@github.com:salesforce/LAVIS.git
cd LAVIS
pip install -e .  # compile locally for later modification

# if there is an error in the above command, try the following:
pip install transformers==4.25
```

If it is still giving you error, please refer
to [LAVIS official installation page](https://github.com/salesforce/LAVIS?tab=readme-ov-file#installation).

Step 6 (for CLIPCap): create `clipcap` env and install `clipcap` library

```
cd ..
git clone https://github.com/rmokady/CLIP_prefix_caption && cd CLIP_prefix_caption
conda env create -f environment.yml
conda activate clip_prefix_caption
```

If it is still giving you error, please refer to [official page](https://github.com/rmokady/CLIP_prefix_caption).
