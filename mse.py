import torchvision.transforms as transforms
from torch.nn import functional as F
import SimpleITK as sitk
import numpy as np
import torch

# PATH1 IS THE PATH OF GENERATED CT (gen_stagex_iterxxxx.npy)
path1 = ''
# PATH1 IS THE PATH OF INPUT CT (inp_stagex_iterxxxx.npy)
path2 = ''

test   = np.load(path1)[0][0]
target = np.load(path2)[0][0]

test   = torch.from_numpy(test)
target = torch.from_numpy(target)

loss = F.mse_loss(test, target)
psnr = 10 * np.log10(1 / loss)
print('Loss: {d_losses:.8f} PSNR: {psnr:.8f}'.format(d_losses=loss, psnr=psnr))