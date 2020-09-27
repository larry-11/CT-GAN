from mhd import MHDData
import torch
import torchvision.transforms as transforms

path = 'data/'

transforms_train = transforms.Compose([transforms.ToTensor()])

train_dataset = MHDData(path, True, transform=transforms_train)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=1, pin_memory=True)

for img in train_loader:
    print(img.shape)

