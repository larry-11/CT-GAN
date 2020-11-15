# CT-GAN

## Method

We propose an unconditioned CT-GAN based on a multi-scale pyramid of GAN to generate various CT scan

![](https://github.com/larry-11/CT-GAN/blob/master/imgs/network.jpg)

We propose two generators based on the type of convolution filter: 2D and 3D. 

Specifically, the 3D G and D modules are structured as follows:

![](https://github.com/larry-11/CT-GAN/blob/master/imgs/GD.jpg)

The 2D G and D modules are structured as follows:

![](https://github.com/larry-11/CT-GAN/blob/master/imgs/2D_GD.jpg)

## DataSetUsed

- LUNA16

## Requirement

- python 3.6

- pytorch 1.0.0 or 1.1.0

- torchvision 0.2.2 or 0.3.0

- tqdm

- scipy

- PIL

- opencv-python (cv2)

  for **Nodule Detection**:

- SimpleITK

- tensorflow-gpu

- pandas

- scikit-learn

## Usage

Executing the *train.sh* to train the network

```
sh train.sh
```

## Result

During the experiment,  training loss is comprised of an adversarial term, a reconstruction term, a projection term, and a segmentation term.

### 3D CT-GAN experiment result:

| **Setting** | **Lsgan_loss** | **Rec_loss** | **Pr_loss** | **Seg_loss** | **MSE** | **PSNR** | **SSIM** |
| ----------- | -------------- | ------------ | ----------- | ------------ | ------- | -------- | -------- |
| A           | √              |              |             |              | 0.0054  | 22.639   | 0.7514   |
| B           | √              | √            |             |              | 0.0025  | 25.991   | 0.8359   |
| C           | √              | √            | √           |              | 0.0024  | 26.070   | 0.8427   |
| D           | √              | √            | √           | √            | 0.0017  | 27.925   | 0.8769   |

Qualitative samples of our 3D generated CT sample are shown as follows:

![](https://github.com/larry-11/CT-GAN/blob/master/imgs/result.jpg)

| **Setting** | **Lsgan_loss** | **Rec_loss** | **Pr_loss** | **Seg_loss** | **MSE** | **PSNR** | **SSIM** |
| ----------- | -------------- | ------------ | ----------- | ------------ | ------- | -------- | -------- |
| A           | √              | √            |             |              | 0.00011 | 39.487   | 0.9878   |
| B           | √              | √            | √           |              | 0.00049 | 33.094   | 0.9554   |
| C           | √              | √            | √           | √            | 0.00032 | 34.931   | 0.9674   |
| D           | √              | √            |             | √            | 0.00006 | 41.967   | 0.9956   |

### 2D CT-GAN experiment result:

Qualitative samples of our 2D generated CT sample are shown as follows:

![](https://github.com/larry-11/CT-GAN/blob/master/imgs/result_2D.png)

## References

Paper:

1. [SinGAN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.pdf)

Code:

1. [SinGAN](https://github.com/FriedRonaldo/SinGAN)
2. [LUNA16-LUng-Nodule-Analysis-2016-Challenge](https://github.com/junqiangchen/LUNA16-Lung-Nodule-Analysis-2016-Challenge)
