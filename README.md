# You Only Look Yourself: Unsupervised and Untrained Single Image Dehazing Neural Network (YOLY) ![](https://visitor-badge.glitch.me/badge?page_id=XLearning-SCU.2021-IJCV-YOLY)

Pytorch implementation for YOLY (IJCV 2021) [[paper](https://link.springer.com/article/10.1007/s11263-021-01431-5)]

## Dependencies

* Python == 3.6.10
* Pytorch == 1.1.0 
* opencv-python == 3.4.2.16 
* opencv-contrib-python == 3.4.2.16 

We also export our conda virtual environment as YOLY.yaml. You can use the following command to create the environment.

```bash
conda env create -f YOLY.yaml
```

## Demo

You can use the following command to dehaze test images in ./data:

```bash
python dehazing.py
```

If you want to test YOLY on a real world image which does not have ground truth. You can use the following command:

```bash
python RW_dehazing.py
```

The only difference between two command is whether the program calculates PSNR and SSIM.

## Citation

If you find YOLY useful in your research, please consider citing:

```
@article{Li:2021kt,
author = {Li, Boyun and Gou, Yuanbiao and Gu, Shuhang and Liu, Jerry Zitao and Zhou, Joey Tianyi and Peng, Xi},
title = {{You Only Look Yourself: Unsupervised and Untrained Single Image Dehazing Neural Network}},
journal = {International Journal of Computer Vision},
year = {2021},
pages = {1--14},
month = mar
}
```
