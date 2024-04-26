# pix2gestalt: Amodal Segmentation by Synthesizing Wholes
### CVPR 2024 (Highlight)
### [Project Page](https://gestalt.cs.columbia.edu/)  | [Paper](https://arxiv.org/pdf/2401.14398.pdf) | [arXiv](https://arxiv.org/abs/2401.14398) | [Weights](https://huggingface.co/cvlab/pix2gestalt-weights) | [Citation](https://github.com/cvlab-columbia/pix2gestalt#citation)

[pix2gestalt: Amodal Segmentation by Synthesizing Wholes](https://gestalt.cs.columbia.edu/)  
 [Ege Ozguroglu](https://egeozguroglu.github.io/)<sup>1</sup>, [Ruoshi Liu](https://ruoshiliu.github.io/)<sup>1</sup>, [Dídac Surís](https://www.didacsuris.com/)<sup>1</sup>, [Dian Chen](https://scholar.google.com/citations?user=zdAyna8AAAAJ&hl=en)<sup>2</sup>, [Achal Dave](https://www.achaldave.com/)<sup>2</sup>, [Pavel Tokmakov](https://pvtokmakov.github.io/home/)<sup>2</sup>, [Carl Vondrick](https://www.cs.columbia.edu/~vondrick/)<sup>1</sup> <br>
 <sup>1</sup>Columbia University, <sup>2</sup>Toyota Research Institute

![teaser](./assets/teaser.gif "Teaser")

## Updates
- We have released our [training script](https://github.com/cvlab-columbia/pix2gestalt#training), [dataset](https://github.com/cvlab-columbia/pix2gestalt#dataset), and [Gradio demo](https://github.com/cvlab-columbia/pix2gestalt#inference-and-weights) with inference instructions.
- Custom training & fine-tuning instructions coming soon. Beyond amodal perception, our repository can also be used to fine-tune Stable Diffusion in an image-conditioned manner with spatial prompts, such as binary masks.
- Pretrained models are released on [Huggingface](https://huggingface.co/cvlab/pix2gestalt-weights), more details provided [here](https://github.com/cvlab-columbia/pix2gestalt#inference-and-weights).  
- pix2gestalt was accepted to CVPR 2024, available on [arXiv](https://arxiv.org/abs/2401.14398)!

##  Installation
```
conda create -n pix2gestalt python=3.9
conda activate pix2gestalt
cd pix2gestalt
pip install -r requirements.txt
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
```
Note: We tested the installation processes on a system with Ubuntu 20.04 with NVIDIA GPUs using Ampere architecture. 

## Inference and Weights

First, download the pix2gestalt weights under `pix2gestalt/ckpt` through one of the following sources:

```
https://huggingface.co/cvlab/pix2gestalt-weights/tree/main

wget -c -P ./ckpt https://gestalt.cs.columbia.edu/assets/epoch=000005.ckpt
```
Note that we have released 2 model weights: epoch=000005.ckpt and epoch=000010.ckpt. By default, we use epoch=000005.ckpt which is the checkpoint after finetuning for 5 epochs on our [dataset](https://github.com/cvlab-columbia/pix2gestalt#dataset). We have also released epoch=000010.ckpt, trained for 10 epochs. This checkpoint can be desirable for synthetic occlusion settings (given our dataset approach), though it may naturally suffer in zero-shot generalization compared to our default model.

Download [SAM](https://segment-anything.com/) checkpoints:
```
wget -c -P ./ckpt https://gestalt.cs.columbia.edu/assets/sam_vit_{b,h,l}.pth
```

Run our Gradio demo for amodal completion and segmentation:

```
python app.py
```

Note that this app uses 22-28 GB of VRAM, so it may not be possible to run it on any GPU.

For inference without the Gradio demo, we provide standalone functionality for each component [here](./pix2gestalt/inference.py), encapsulated by the [run_pix2gestalt](./pix2gestalt/inference.py#L138) method. It supports both predicted modal masks from SAM (like our demo) or ground truth modal masks. 

### Training
Download the image-conditioned Stable Diffusion diffusion checkpoint released by Lambda Labs: 

```
wget -c -P ./ckpt https://gestalt.cs.columbia.edu/assets/sd-image-conditioned-v2.ckpt
```

Then, download our fine-tuning dataset via the instructions [here](https://github.com/cvlab-columbia/pix2gestalt#dataset) and update its path (see `data:params:root_dir`) in our [config](./pix2gestalt/configs/sd-finetune-pix2gestalt-c_concat-256.yaml).

Run training command:  
```
python main.py \
    -t \
    --base configs/sd-finetune-pix2gestalt-c_concat-256.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 2 \
    --finetune_from ckpt/sd-image-conditioned-v2.ckpt
```
Note that this training script is set for an 8-GPU system, each with 80GB of VRAM. Empirically, the large batch size is very important for "stably" fine-tuning Stable Diffusion in an image conditioned manner. If you have smaller GPUs, consider using smaller batch sizes with gradient accumulation to obtain a similar effective batch size.

### Dataset
Download and extract our dataset of occluded objects & their whole counterparts with:
```
wget https://gestalt.cs.columbia.edu/assets/pix2gestalt_occlusions_release.tar.gz

tar -xvf pix2gestalt_occlusions_release.tar.gz
```
Disclaimer: note that the source images are from the [Segment Anything-1B Dataset](https://segment-anything.com/dataset/index.html), which has faces and license plates de-identified. For amodal perception targeted specifically for such domains, we recommend re-training or fine-tuning pix2gestalt via our custom trainining instructions. 

The dataset is intended for research purposes only. The licenses for the source images are released under the same license that they are in SA-1B.

### Amodal Recognition and 3D Reconstruction
Since we synthesize RGB images of whole objects (amodal completion), our approach makes it straightforward to equip various computer vision methods with the ability to handle occlusions, beyond amodal segmentation.

For recognition, we use [CLIP](https://github.com/openai/CLIP) as the base open-vocabulary classifier. For novel view synthesis and  3D reconstruction, we use [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer). Refer to our [paper](https://gestalt.cs.columbia.edu/static/pix2gestalt.pdf) and [supplementary](https://gestalt.cs.columbia.edu/static/supplementary.pdf) for more details.


## Citation
If you use this code, please consider citing the paper as:
```
@article{ozguroglu2024pix2gestalt,
        title={pix2gestalt: Amodal Segmentation by Synthesizing Wholes},
        author={Ege Ozguroglu and Ruoshi Liu and D\'idac Sur\'s and Dian Chen and Achal Dave and Pavel Tokmakov and Carl Vondrick},
        journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2024}
}
```

##  Acknowledgement
This research is based on work partially supported by the Toyota Research Institute, the DARPA MCS program under Federal Agreement No. N660011924032, the NSF NRI Award \#1925157, and the NSF AI Institute for Artificial and Natural Intelligence Award \#2229929. DS is supported by the Microsoft PhD Fellowship.