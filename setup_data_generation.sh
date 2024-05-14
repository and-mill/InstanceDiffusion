# install grounding-sam, ram and grounding-dino
#git clone git@github.com:IDEA-Research/Grounded-Segment-Anything.git
git subtree add git@github.com:IDEA-Research/Grounded-Segment-Anything.git main --prefix Grounded-Segment-Anything --squash
cd Grounded-Segment-Anything
python -m pip install -e segment_anything
#git clone https://github.com/IDEA-Research/GroundingDINO.git
git subtree add https://github.com/IDEA-Research/GroundingDINO.git main --prefix GroundingDINO --squash
pip install -U openmim
mim install mmcv
python -m pip install -e GroundingDINO
#git clone https://github.com/xinyu1205/recognize-anything.git
git subtree add https://github.com/xinyu1205/recognize-anything.git main --prefix recognize-anything --squash
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/
pip install --upgrade diffusers
pip install submitit
# install lavis
pip install salesforce-lavis
pip install webdataset

# download pretrained checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth