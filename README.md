[![License](https://img.shields.io/badge/license-red.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv.2312.06709-blue.svg)](https://arxiv.org/abs/2308.06332)
[![Paper](https://img.shields.io/badge/paper-MICCAI.2023-green.svg)](https://link.springer.com/chapter/10.1007/978-3-031-43990-2_65)

# \[MICCAI 2023\] SwinFSR: Advancing Super-Resolution of Fundus Images for SANS Visual Assessment Technology


Official PyTorch implementation of \[MICCAI 2023\] [**Revolutionizing Space Health (Swin-FSR): Advancing Super-Resolution of Fundus Images for SANS Visual Assessment Technology**](https://arxiv.org/abs/2201.01266).

\[[Paper](https://arxiv.org/abs/2312.06709)\]\[[BibTex](#citing-SwinFSR)\]

<br clear="left"/>

---

## Abstract
Swin-FSR is a groundbreaking model designed to address the challenges of super-resolution imaging in remote and constrained environments. Leveraging Swin Transformer with spatial and depth-wise attention mechanisms, Swin-FSR achieves remarkable results in enhancing the resolution of fundus images, crucial for accurate disease identification. With a Peak signal-to-noise-ratio (PSNR) of 47.89, 49.00, and 45.32 on prominent datasets such as iChallenge-AMD, iChallenge-PM, and G1020, respectively, Swin-FSR demonstrates its efficacy in improving image quality across diverse visual domains. Additionally, when applied to the analysis of images related to Spaceflight Associated Neuro-Ocular Syndrome (SANS), Swin-FSR yields comparable results to previous architectures, showcasing its versatility and robustness in various medical imaging applications.


<div align="left">
  <img src="Figure/Fig1(4).png" width="1000"/>
</div>

## Training

_Detail_Coming Soon_


## Results
### Visulaze-Output:
Qualitative comparison of (×2) image reconstruction using different SR methods on AMD, PALM, G1020 and SANS dataset. The green rectangle is the zoomed-in region. The rows are for the AMD, PALM and SANS datasets. Whereas, the column is for each different models: SwinFSR, SwinIR, RCAN and ELAN.
<div align="left">
  <img src="Figure/Fig2(4).png" width="700"/>
</div>

### Model stats:
(a) and (b) Effects of the numbers of iRSTB Blocks on the PSNR and SSIM, and (c) and (d) the numbers of DCA Blocks on the PSNR and SSIM for *2 images.
<div align="left">
  <img src="Figure/Fig4(2).png" width="700"/>
</div>


### Clinical Assessment:
We carried out a diagnostic assessment with two expert ophthalmologists and test samples of 80 fundus images (20 fundus images per disease classes: AMD, Glaucoma, Pathological Myopia and SANS for both original x2 and x4 images, and super-resolution enhanced images). Half of the 20 fundus images were control patients without disease pathologies; the other half contained disease pathologies. The clinical experts were not provided any prior pathology information regarding the images. Each of the experts was given 10 images with equally distributed control and diseased images for each disease category.
<div align="left">
  <img src="Figure/clinicalAssessment.jpg" width="900"/>
</div>

## Citing SwinFSR

If you find this repository useful, please consider giving a star and citation:

#### MICCAI 2023 Reference:
```bibtex
@inproceedings{hossain2023revolutionizing,
  title={Revolutionizing space health (Swin-FSR): advancing super-resolution of fundus images for SANS visual assessment technology},
  author={Hossain, Khondker Fariha and Kamran, Sharif Amit and Ong, Joshua and Lee, Andrew G and Tavakkoli, Alireza},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={693--703},
  year={2023},
  organization={Springer}
}
```

#### ArXiv Reference:
```bibtex
@article{hossain2023revolutionizing,
  title={Revolutionizing Space Health (Swin-FSR): Advancing Super-Resolution of Fundus Images for SANS Visual Assessment Technology},
  author={Hossain, Khondker Fariha and Kamran, Sharif Amit and Ong, Joshua and Lee, Andrew G and Tavakkoli, Alireza},
  journal={arXiv preprint arXiv:2308.06332},
  year={2023}
}
```
