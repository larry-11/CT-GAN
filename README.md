# CT-GAN

## Method

We propose an unconditioned CT-GAN based on a multi-scale pyramid of GAN to generate various CT scan

![](https://github.com/larry-11/CT-GAN/tree/master/imgs/network.jpg)

Specifically, the G and D modules are structured as follows:

![](https://github.com/larry-11/CT-GAN/tree/master/imgs/GD.jpg)

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

| **Setting** | **Lsgan_loss** | **Rec_loss** | **Pr_loss** | **Seg_loss** | **MSE** | **PSNR** | **SSIM** |
| ----------- | -------------- | ------------ | ----------- | ------------ | ------- | -------- | -------- |
| A           | √              |              |             |              | 0.0054  | 22.639   | 0.7514   |
| B           | √              | √            |             |              | 0.0025  | 25.991   | 0.8359   |
| C           | √              | √            | √           |              | 0.0024  | 26.070   | 0.8427   |
| D           | √              | √            | √           | √            | 0.0017  | 27.925   | 0.8769   |

Qualitative samples of our generated CT sample are shown as follows:

![](https://github.com/larry-11/CT-GAN/tree/master/imgs/result.jpg)