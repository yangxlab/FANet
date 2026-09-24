# FA-Net: A Feature Alignment Network for Video-Based Visible-Infrared Person Re-Identification

**Xi Yang**, Wenjiao Dong, Xian Wang, De Cheng\* (corresponding author), Nannan Wang

*IEEE Transactions on Image Processing (TIP)*, vol. 34, pp. 8406–8420, 2025

- **Paper (DOI):** <https://doi.org/10.1109/TIP.2025.3642633>
- **IEEE Xplore:** <https://ieeexplore.ieee.org/document/11301926>

---

## Method

### Overall Architecture

<p align="center">
  <img src="assets/fig2_architecture.png" width="100%"
       alt="Architecture of FA-Net">
</p>

<p align="center"><sub><i>Fig. 2 — Architecture of FA-Net. Given a sequence, the SAM-GA module is first applied to remove domain shift noise and enhance each frame representation. Apart from the CNN backbone, FA-Net mainly consists of STAM and MDC. STAM aligns the spatial representation using global and local features and employs inter-frame feature mining to explore temporal clues; MDC applies a symmetric distribution loss to constrain the feature distributions of the two modalities.</i></sub></p>

FA-Net comprises three components:

- **SAM-Guided Augmentation (SAM-GA)** — leverages the Segment Anything Model to transform the image space of RGB and IR frames, performing training-only stochastic convex mixing between raw and mask-filtered frames to suppress domain-shift noise.
- **Spatio-Temporal Alignment Module (STAM)** — a three-branch design (spatial / temporal / spatio-temporal alignment) that aligns spatial representations from global and local perspectives and establishes temporal relationships across frames.
- **Modality Distribution Constraint (MDC)** — a symmetric distribution loss built on Optimal Transport theory, which minimizes the cost of transforming one modality's feature distribution into the other.

### SAM Guidance Augmentation (SAM-GA)

<p align="center">
  <img src="assets/fig3_sam_ga.png" width="100%"
       alt="Architecture of SAM-GA">
</p>

<p align="center"><sub><i>Fig. 3 — Architecture of the SAM Guidance Augmentation (SAM-GA). The image branch is on the left and the video branch on the right.</i></sub></p>

### Spatio-Temporal Alignment Module (STAM)

<p align="center">
  <img src="assets/fig4_stam.png" width="100%"
       alt="Architecture of STAM">
</p>

<p align="center"><sub><i>Fig. 4 — Architecture of the Spatio-Temporal Alignment Module (STAM). In practical implementation, STAM comprises three branches: the Spatial Feature Alignment branch, the Temporal Feature Alignment branch, and the Spatio-Temporal Feature Alignment branch.</i></sub></p>

## Environment

The code is implemented with PyTorch. Main dependencies:

```
python >= 3.7
torch, torchvision
numpy, scipy
tensorboardX
Pillow
mobile_sam  (MobileSAM, for the SAM-GA module)
```

## Data Preparation

Please download the **HITSZ-VCM** dataset and set its root path in `data_manager.py`:

```python
class VCM(object):
    root = '/path/to/VCM-HITSZ/'
```

## Training

```bash
python train.py --dataset VCM --arch resnet50 --gpu 0
```

## Testing

```bash
python test.py --dataset VCM --mode all --gpu 0 -r save_model/<your_checkpoint>.t
```

The result of both retrieval modes (infrared to visible and visible to infrared) is printed as Rank-1 / Rank-5 / Rank-10 / Rank-20 / mAP.

## Acknowledgements

The SAM-GA strategy builds upon [MobileSAM](https://github.com/ChaoningZhang/MobileSAM).

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{yang2025fanet,
  author    = {Xi Yang and Wenjiao Dong and Xian Wang and De Cheng and Nannan Wang},
  title     = {FA-Net: A Feature Alignment Network for Video-Based Visible-Infrared Person Re-Identification},
  journal   = {IEEE Transactions on Image Processing},
  volume    = {34},
  pages     = {8406--8420},
  year      = {2025},
  doi       = {10.1109/TIP.2025.3642633}
}
```
