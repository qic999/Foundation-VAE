<h1 align="center">Foundation-VAE</h1>

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=qic999/Foundation-VAE)
[![GitHub Repo stars](https://img.shields.io/github/stars/qic999/Foundation-VAE?style=social)](https://github.com/qic999/Foundation-VAE/stargazers)
[![arXiv](https://img.shields.io/badge/arXiv-red)]() &ensp; [![Project Page](https://img.shields.io/badge/Project%20Page-green
)](https://qic999.github.io/projects/Foundation-VAE/)

</div>

We make a progressive stride toward training-free medical VAEs by leveraging a critical observation: a single Foundation VAE, pretrained at scale on natural images and videos, can serve as a unified interface for CT Reconstruction, Augmentation, and Generation. (1) CT Reconstruction: A Foundation VAE pretrained at scale on natural images and videos reconstructs a 3D CT volume via its frozen encoder $E$ and decoder $D$.
(2) CT Augmentation: Through zero shot transfer to CT, the reconstructed volumes provide a boundary enhanced training view, improving downstream segmentation especially on surface accuracy.
(3) CT Generation: In the fixed latent space of the same \textit{Foundation VAE}, we train a conditional latent diffusion model to synthesize anatomically consistent healthy and abnormal CT volumes, controlled by organ masks and clinical findings.

<p align="center"><img width="100%" src="figures/fig_teaser.png" /></p>

## Paper

<b>Foundation VAE for CT Reconstruction, Augmentation, and Generation</b> <br/>
[Qi Chen](https://scholar.google.com/citations?user=4Q5gs2MAAAAJ&hl=en)<sup>1,*</sup>, [Shuhan Ding](https://scholar.google.com/citations?user=NluKVTAAAAAJ&hl=en&oi=ao)<sup>2,*</sup>, [Yu Gu](https://scholar.google.com/citations?user=1PoaURIAAAAJ&hl=en)<sup>3</sup>, [Nan Liu](https://scholar.google.com/citations?user=ceF698kAAAAJ&hl=en)<sup>2</sup>, [Jiang Bian](https://scholar.google.com/citations?user=pZBEnY8AAAAJ&hl=en)<sup>3</sup>, [Alan L. Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>1</sup>, [Zongwei Zhou](https://www.zongweiz.com/)<sup>1</sup>, and [Jingjing Fu](https://scholar.google.com/citations?user=w-6C7LkAAAAJ&hl=zh-CN)<sup>3</sup> <br/>
<sup>1</sup> Johns Hopkins University <br/>
<sup>2</sup> Duke-NUS Medical School <br/>
<sup>3</sup> Microsoft Research <br/>
<sup>*</sup> Equal contribution
[paper](https://www.cs.jhu.edu/~alanlab/Pubs24/chen2024towards.pdf) | [code](https://github.com/qic999/Foundation-VAE) | [huggingface](https://huggingface.co/qicq1c/Foundation-VAE)

**We have summarized publications related to Medical VAE in [Awesome Medical VAE](https://github.com/qic999/Foundation-VAE/blob/main/AWESOME.md) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re).**


## 0. Installation

```bash
git clone https://github.com/qic999/Foundation-VAE.git
cd Foundation-VAE
```

## 1. Reconstruction
We transfer a Foundation VAE pretrained on natural images/videos to 3D CT reconstruction with both encoder and decoder frozen.
This reconstruction operator suppresses acquisition noise while preserving clinically relevant anatomical boundaries, making it suitable as a stable CT interface across heterogeneous scanners and protocols.

### Demo
<p align="center">
  <img width="100%" src="figures/lung_066_grid_allz_loop_small.gif" alt="Foundation-VAE lung reconstruction demo" />
</p>
<p align="center"><em>MSD Task06 Lung: reconstruction and segmentation comparison.</em></p>
<p align="center">
  <img width="100%" src="figures/pancreas_095_grid_allz_loop.gif" alt="Foundation-VAE pancreas reconstruction demo" />
</p>
<p align="center"><em>MSD Task07 Pancreas: reconstruction and segmentation comparison.</em></p>

### Data
- You can evaluate on [MSD dataset](http://medicaldecathlon.com/)
- Our released reconstruction assets:
  [https://huggingface.co/qicq1c/Foundation-VAE/tree/main/Reconstruction](https://huggingface.co/qicq1c/Foundation-VAE/tree/main/Reconstruction)


## 2. Augmentation
Our augmentation strategy is **reconstruction-based augmentation**: use reconstructed CT volumes as an additional training view for downstream tasks.
Because reconstruction is boundary-stable and largely preserves label-defining geometry, segmentation trained on reconstructed CTs is comparable to, and often better than, training on raw CTs, with clear gains on boundary-sensitive metrics.

This stage is annotation-free with respect to reconstruction itself and can be directly plugged into standard segmentation pipelines.

Released augmentation models/assets:
[https://huggingface.co/qicq1c/Foundation-VAE/tree/main/Augmentation](https://huggingface.co/qicq1c/Foundation-VAE/tree/main/Augmentation)


## 3. Generation
In the same fixed Foundation VAE latent space, we train a **conditional latent diffusion model** for controllable 3D CT generation.

### Conditioning
- Anatomy masks for spatial grounding
- Disease masks for pathology control
- Radiology report embeddings for semantic control

### Key design
- Frozen Foundation VAE encoder/decoder as latent interface
- Mask latents concatenated during denoising for structure consistency
- Lightweight 3D consistency attention across slices for coherent volumetric anatomy/pathology

This enables controllable synthesis of healthy and abnormal CT volumes under unified latent modeling.

### Demo
<p align="center">
  <img width="100%" src="figures/three_view_valid_991_a_1_loop_small.gif" alt="Foundation-VAE controllable CT generation demo" />
</p>
<p align="center"><em>Three-view (axial/coronal/sagittal) generated CT with report conditioning.</em></p>
<p align="center">
  <img width="100%" src="figures/loop_small.gif" alt="Foundation-VAE anatomical and pathological grounding demo" />
</p>
<p align="center"><em>Anatomical and pathological grounding comparison.</em></p>

Released generation models/assets:
[https://huggingface.co/qicq1c/Foundation-VAE/tree/main/Generation](https://huggingface.co/qicq1c/Foundation-VAE/tree/main/Generation)



## Citation

```
@article{chen2026foundationvae,
  title         = {Foundation VAE for CT Reconstruction, Augmentation, and Generation},
  author        = {Chen, Qi and Ding, Shuhan and Gu, Yu and Liu, Nan and Bian, Jiang and Yuille, Alan L. and Zhou, Zongwei and Fu, Jingjing},
  journal       = {arXiv preprint arXiv:2602.12345},
  year          = {2026},
  archivePrefix = {arXiv},
  eprint        = {2602.12345},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2602.12345}
}
```
