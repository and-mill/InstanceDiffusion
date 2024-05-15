
### Pipeline
Dataset preparation consists of three major steps:
1. Image-level label generation
2. Bounding-box and mask generation
3. Instance-level text prompt generation

### Requirements
- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 2.0 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. 
  Note, please check PyTorch version matches that is required by Detectron2.

### Example conda environment setup
Please begin by following the instructions in [INSTALL](https://github.com/frank-xwang/InstanceDiffusion/tree/main?tab=readme-ov-file#installation) to set up the conda environment. After that, proceed with the steps below to install the necessary packages for generating training data.

and-mill: Python 3.9
```bash
# install grounding-sam, ram and grounding-dino
#git clone git@github.com:IDEA-Research/Grounded-Segment-Anything.git
git subtree add git@github.com:IDEA-Research/Grounded-Segment-Anything.git main --prefix Grounded-Segment-Anything --squash  # Have to commit any changes beforehand
cd Grounded-Segment-Anything
#python -m pip install -e segment_anything
#PIP_CACHE_DIR=/home/host_mueller/cache/ XDG_CACHE_HOME=/home/host_mueller/cache/ TMPDIR=/home/host_mueller/tmp/ pip install --cache-dir=/home/host_mueller/cache/ -e segment_anything
pip install -e segment_anything


#git clone https://github.com/IDEA-Research/GroundingDINO.git
git subtree add https://github.com/IDEA-Research/GroundingDINO.git main --prefix GroundingDINO --squash  # Have to commit any changes beforehand
pip install -U openmim
mim install mmcv
#python -m pip install -e GroundingDINO
#PIP_CACHE_DIR=/home/host_mueller/cache/ XDG_CACHE_HOME=/home/host_mueller/cache/ TMPDIR=/home/host_mueller/tmp/ pip install --cache-dir=/home/host_mueller/cache/ -e GroundingDINO
pip install -e GroundingDINO
# and-mill: Will likely get "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. scikit-image 0.20.0 requires scipy<1.9.2,>=1.8; python_version <= "3.9", but you have scipy 1.10.0 which is incompatible.".


#git clone https://github.com/xinyu1205/recognize-anything.git
git subtree add https://github.com/xinyu1205/recognize-anything.git main --prefix Grounded-Segment-Anything/recognize-anything --squash  # Have to commit any changes beforehand

#PIP_CACHE_DIR=/home/host_mueller/cache/ XDG_CACHE_HOME=/home/host_mueller/cache/ TMPDIR=/home/host_mueller/tmp/ pip install --cache-dir=/home/host_mueller/cache/ -r ./recognize-anything/requirements.txt
pip install -r ./recognize-anything/requirements.txt

#PIP_CACHE_DIR=/home/host_mueller/cache/ XDG_CACHE_HOME=/home/host_mueller/cache/ TMPDIR=/home/host_mueller/tmp/ pip install --cache-dir=/home/host_mueller/cache/ --upgrade setuptools pip
pip install --upgrade setuptools pip  # and-mill
# and-mill: will likely get "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. openxlab 0.0.38 requires setuptools~=60.2.0, but you have setuptools 69.5.1 which is incompatible."

#PIP_CACHE_DIR=/home/host_mueller/cache/ XDG_CACHE_HOME=/home/host_mueller/cache/ TMPDIR=/home/host_mueller/tmp/ pip install --cache-dir=/home/host_mueller/cache/ -e ./recognize-anything/
pip install -e ./recognize-anything/

#PIP_CACHE_DIR=/home/host_mueller/cache/ XDG_CACHE_HOME=/home/host_mueller/cache/ TMPDIR=/home/host_mueller/tmp/ pip install --cache-dir=/home/host_mueller/cache/ --upgrade diffusers
pip install --upgrade diffusers

#PIP_CACHE_DIR=/home/host_mueller/cache/ XDG_CACHE_HOME=/home/host_mueller/cache/ TMPDIR=/home/host_mueller/tmp/ pip install --cache-dir=/home/host_mueller/cache/ submitit
pip install submitit

# install lavis
#PIP_CACHE_DIR=/home/host_mueller/cache/ XDG_CACHE_HOME=/home/host_mueller/cache/ TMPDIR=/home/host_mueller/tmp/ pip install --cache-dir=/home/host_mueller/cache/ salesforce-lavis
pip install salesforce-lavis
# and-mill: Will likely get "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. supervision 0.20.0 requires scipy==1.10.0; python_version < "3.9", but you have scipy 1.9.1 which is incompatible."

#PIP_CACHE_DIR=/home/host_mueller/cache/ XDG_CACHE_HOME=/home/host_mueller/cache/ TMPDIR=/home/host_mueller/tmp/ pip install --cache-dir=/home/host_mueller/cache/ webdataset
pip install webdataset

# download pretrained checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth

#PIP_CACHE_DIR=/home/host_mueller/cache/ XDG_CACHE_HOME=/home/host_mueller/cache/ TMPDIR=/home/host_mueller/tmp/ pip install --cache-dir=/home/host_mueller/cache/ transformers
pip install transformers

# For some reason torch and torchvision end up incompatible. You will see this when you see this when running dataset-generation script
# "ModuleNotFoundError: No module named 'torch._custom_ops'" or
# "RuntimeError: Detected that PyTorch and torchvision were compiled with different CUDA versions. PyTorch has CUDA Version=11.7 and torchvision has CUDA Version=11.8. Please reinstall the torchvision that matches your PyTorch install."
#PIP_CACHE_DIR=/home/host_mueller/cache/ XDG_CACHE_HOME=/home/host_mueller/cache/ TMPDIR=/home/host_mueller/tmp/ pip install --cache-dir=/home/host_mueller/cache/ -U torch torchvision
pip install -U torch torchvision
# and-mill: Will likely get "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. xformers 0.0.22 requires torch==2.0.1, but you have torch 2.3.0 which is incompatible."
```

### Expected structure for data generation
We should have a 'train_data.json' file with paired image path and image caption as follows. 
Utilizing a subset of the LAION-400M dataset (5~10M images) should enable the reproduction of the results. Additionally, incorporating datasets from GLIGEN could further enhance performance in adhering to bounding box conditions.
```
[
  {
    "image": /PATH/TO/IMAGE1,
    "caption": IMAGE1-CAPTION
  },
  {
    "image": /PATH/TO/IMAGE2,
    "caption": IMAGE2-CAPTION
  }
]
```

### Script for generating training data
We support training data generation with multiple GPUs and nodes. Executing the following commands will produce instance segmentation masks, detection bounding boxes, and instance-level captions for all images listed in `train_data_path`.
```bash
cd ..
cd dataset-generation/
PYTHONPATH="../Grounded-Segment-Anything:../Grounded-Segment-Anything/GroundingDINO" python run_with_submitit_generate_caption.py     --timeout 4000     --partition learn     --num_jobs 1     --config ../Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py     --ram_checkpoint ../Grounded-Segment-Anything/ram_swin_large_14m.pth     --grounded_checkpoint ../Grounded-Segment-Anything/groundingdino_swint_ogc.pth     --sam_checkpoint ../Grounded-Segment-Anything/sam_vit_h_4b8939.pth     --box_threshold 0.25     --text_threshold 0.2     --iou_threshold 0.5     --device "cuda"     --sam_hq_checkpoint ../Grounded-Segment-Anything/sam_hq_vit_h.pth     --use_sam_hq     --output_dir "/home/host_mueller/mueller/InstanceDiffusion/Grounded-Segment-Anything/sample-data-gen/"     --train_data_path /home/host_datasets/COCO/train2017/ --input_image /home/host_datasets/COCO/train2017/000000061048.jpg    --output_dir "train-data"  # and-mill

# For each image, a corresponding JSON file is created in the --output_dir. 
# To compile all file names into a list for model training, use
ls train-data/*.json > train.txt

# Or for handling a large number of files, it is recommended to run
python jsons2txt.py

```
`--num_jobs` specifies how many GPUs are employed for data generation, with jobs automatically distributed across GPUs on multiple machines.
