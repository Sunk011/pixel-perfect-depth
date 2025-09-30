<div align="center">
<h1>Pixel-Perfect Depth</h1>

<a href=""><img src='https://img.shields.io/badge/arXiv-Pixel Perfect Depth-red' alt='Paper PDF'></a>
<a href='https://pixel-perfect-depth.github.io/'><img src='https://img.shields.io/badge/Project_Page-Pixel Perfect Depth-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/gangweix/Pixel-Perfect-Depth'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
</div>

This work presents Pixel-Perfect Depth, a monocular depth estimation model with pixel-space diffusion transformers. Compared to existing discriminative and generative models, 
its estimated depth maps can produce high-quality, flying-pixel-free point clouds.

![teaser](assets/teaser.png)

## News
- **2024-10-01:** Paper, project page, code, models, and demo are all released.

## Pre-trained Models

Our pretrained models are available on the huggingface hub:

| Model | Params | Checkpoint | Training Resolution |
|:-|-:|:-:|:-:|
| PPD-Large | 500M | [Download](https://huggingface.co/gangweix/Pixel-Perfect-Depth/resolve/main/ppd.pth) | 1024x768 |
