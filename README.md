<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

Just a fork of [Detectron2](https://github.com/facebookresearch/detectron2) with some modifications for personal usages.

Extra libraries used:

* `imantics` for converting inference masks into shapes


All setup steps:

```bash
conda create -n detectron2 python=3.10 -y
conda activate detectron2
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install imantics Flask -y

```
