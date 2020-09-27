from mhd import MHDData
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
import cv2

path = '/data/shanyx/larry/SinGAN/data'

transforms_train = transforms.Compose([transforms.ToTensor()])

train_dataset = MHDData(path, True, transform=transforms_train)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=1, pin_memory=True)

for img in train_loader:
    img = img.mul(255).cpu().numpy().squeeze(0)
    cv2.imwrite("/data/shanyx/larry/SinGAN/data/SinGANdata/loader0.png", img[0,:,:])
    cv2.imwrite("/data/shanyx/larry/SinGAN/data/SinGANdata/loader1.png", img[:,0,:])
    cv2.imwrite("/data/shanyx/larry/SinGAN/data/SinGANdata/loader2.png", img[:,:,0])
    print(img.shape)

