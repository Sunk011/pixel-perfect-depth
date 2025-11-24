<div align="center">
<h1>Pixel-Perfect Depth (像素级完美深度估计)</h1>

[**中文**](README_zh-CN.md) | [**English**](README.md)

[**Gangwei Xu**](https://gangweix.github.io/)<sup>1,2,&ast;</sup> · [**Haotong Lin**](https://haotongl.github.io/)<sup>3,&ast;</sup> · Hongcheng Luo<sup>2</sup> · [**Xianqi Wang**](https://scholar.google.com/citations?user=1GCLBNAAAAAJ&hl=zh-CN&oi=ao)<sup>1</sup> · [**Jingfeng Yao**](https://jingfengyao.github.io/)<sup>1</sup>
<br>
[**Lianghui Zhu**](https://scholar.google.com/citations?user=NvMHcs0AAAAJ&hl=zh-CN&oi=ao)<sup>1</sup> · Yuechuan Pu<sup>2</sup> · Cheng Chi<sup>2</sup> · Haiyang Sun<sup>2,&dagger;</sup> · Bing Wang<sup>2</sup> 
<br>
Guang Chen<sup>2</sup> · Hangjun Ye<sup>2</sup> · [**Sida Peng**](https://pengsida.net/)<sup>3</sup> · [**Xin Yang**](https://sites.google.com/view/xinyang/home)<sup>1,&dagger;,✉️</sup>

<sup>1</sup>华中科技大学&emsp; <sup>2</sup>小米汽车&emsp; <sup>3</sup>浙江大学  
<br>
&ast;共同第一作者 &emsp; &dagger;项目负责人 &emsp; ✉️ 通讯作者

<a href="https://arxiv.org/pdf/2510.07316"><img src='https://img.shields.io/badge/arXiv-Pixel Perfect Depth-red' alt='Paper PDF'></a>
<a href='https://pixel-perfect-depth.github.io/'><img src='https://img.shields.io/badge/Project_Page-Pixel Perfect Depth-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/gangweix/Pixel-Perfect-Depth'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
</div>

本项目提出了 Pixel-Perfect Depth，这是一种基于像素空间扩散 Transformer 的单目深度估计模型。与现有的判别式和生成式模型相比，其估计的深度图可以生成高质量、无飞点的点云。

![teaser](assets/teaser.png)

![overview](assets/overview.png)  
*Pixel-Perfect Depth 概览。我们直接在像素空间执行扩散生成，而不使用任何 VAE。* 

## 🌟 特性

*   **像素空间扩散生成**：直接在图像空间操作，无需 VAE 或潜在表示，能够从估计的深度图生成无飞点的点云。
*   **融合架构**：我们的模型将判别式表示 (ViT) 集成到生成式建模 (DiT) 中，充分利用了两种范式的优势。
*   **纯 Transformer 架构**：网络架构完全基于 Transformer，不包含任何卷积层。
*   **灵活的分辨率支持**：虽然模型是在 1024×768 的固定分辨率下训练的，但在推理过程中可以灵活支持各种输入分辨率和纵横比。

## 新闻
- **2025-10-01:** 论文、项目主页、代码、模型和演示均已发布。

## 预训练模型

我们的预训练模型可在 Hugging Face Hub 上获取：

| 模型 | 参数量 | 权重文件 | 训练分辨率 |
|:-|-:|:-:|:-:|
| PPD-Large | 500M | [下载](https://huggingface.co/gangweix/Pixel-Perfect-Depth/resolve/main/ppd.pth) | 1024×768 |

## 使用方法

### 准备工作

```bash
git clone https://github.com/gangweix/pixel-perfect-depth
cd pixel-perfect-depth
pip install -r requirements.txt
```

下载我们的预训练模型 [ppd.pth](https://huggingface.co/gangweix/Pixel-Perfect-Depth/resolve/main/ppd.pth) 并将其放在 `checkpoints/` 目录下。
此外，您还需要下载预训练模型 [depth_anything_v2_vitl.pth](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) 并将其放在 `checkpoints/` 目录下。

### 在*图像*上运行深度估计

```bash
python run.py 
```

### 在*图像*上运行点云生成

生成点云需要来自 MoGe 的度量深度和相机内参。
请下载预训练模型 [moge2.pt](https://huggingface.co/Ruicheng/moge-2-vitl-normal/resolve/main/model.pt?download=true) 并将其放在 `checkpoints/` 文件夹下。

```bash
python run_point_cloud.py --save_pcd
```

## 🖥️ Web 演示与 Docker 支持

我们提供了一个基于 Gradio 的本地 Web 演示，支持：
- **单张图片推理**：上传图片，调整采样步数，可视化并下载结果。
- **批量处理**：上传包含图片的文件夹，批量处理并下载 ZIP 格式的结果。
- **多 GPU 支持**：自动检测可用 GPU 并在界面中允许动态切换。

### 使用 Docker (推荐)

我们提供了 `Dockerfile` 和辅助脚本以便轻松部署。

**先决条件：**
- 已安装 Docker
- NVIDIA Container Toolkit (用于 GPU 支持)
- 权重文件已放置在 `checkpoints/` 目录下

**一键运行：**

```bash
chmod +x docker-run.sh
./docker-run.sh
```

该脚本将：
1. 构建 Docker 镜像。
2. 运行带有 GPU 访问权限 (`--gpus all`) 的容器。
3. 挂载本地的 `checkpoints/` 和 `assets/` 目录。
4. 在 `http://localhost:7860` 暴露 Web UI。

**手动构建与运行：**

```bash
# 构建镜像
docker build -t pixel-perfect-depth .

# 运行容器
docker run -it --rm \
    --gpus all \
    -p 7860:7860 \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/assets:/app/assets" \
    pixel-perfect-depth
```

### 本地运行 Web 演示 (不使用 Docker)

如果您已在本地设置好环境：

```bash
pip install gradio
python app.py
```

## 与先前方法的定性比较

与 Depth Anything v2 和 MoGe 2 相比，我们的模型保留了更细粒度的细节，同时与 Depth Pro 相比表现出显著更高的鲁棒性。

![teaser](assets/vis_comp.png)

## 致谢

感谢 [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)、[MoGe](https://github.com/microsoft/MoGe) 和 [DiT](https://github.com/facebookresearch/DiT) 团队发布的代码和模型。我们也衷心感谢 NeurIPS 审稿人对这项工作的认可（评分：5, 5, 5, 5）。

## 引用

如果您觉得本项目有用，请考虑引用：

```bibtex
@article{xu2025pixel,
  title={Pixel-Perfect Depth with Semantics-Prompted Diffusion Transformers},
  author={Xu, Gangwei and Lin, Haotong and Luo, Hongcheng and Wang, Xianqi and Yao, Jingfeng and Zhu, Lianghui and Pu, Yuechuan and Chi, Cheng and Sun, Haiyang and Wang, Bing and others},
  journal={arXiv preprint arXiv:2510.07316},
  year={2025}
}
```
