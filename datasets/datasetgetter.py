# from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import os
import torchvision.transforms as transforms
from datasets import photoimage
from datasets import mhd


def get_dataset(dataset, args):
    if dataset.lower() == 'photo':
        print('USE PHOTO DATASET')
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        transforms_train = transforms.Compose([transforms.Resize((args.img_size_max, args.img_size_max)),
                                               transforms.ToTensor(),
                                               normalize])
        transforms_val = transforms.Compose([transforms.Resize((args.img_size_max, args.img_size_max)),
                                             transforms.ToTensor(),
                                             normalize])

        train_dataset = photoimage.PhotoData(args.data_dir, True, transform=transforms_train, img_to_use=args.img_to_use)
        val_dataset = photoimage.PhotoData(args.data_dir, False, transform=transforms_val, img_to_use=args.img_to_use)

        if train_dataset.randidx != -999:
            args.img_to_use = train_dataset.randidx
    elif dataset.lower() == 'mhd':
        print('USE MHD DATASET')
        transforms_train = transforms.Compose([transforms.ToTensor()])

        train_dataset = mhd.MHDData(args.data_dir, True, transform=transforms_train)
        val_dataset = mhd.MHDData(args.data_dir, True, transform=transforms_train)
    else:
        print('NOT IMPLEMENTED DATASET :', dataset)
        exit(-3)

    return train_dataset, val_dataset
