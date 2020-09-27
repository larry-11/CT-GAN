from torch import nn
import torch


class Discriminator(nn.Module):
    def __init__(self, img_size_min, num_scale, scale_factor=4/3):
        torch.set_default_tensor_type(torch.DoubleTensor)
        super(Discriminator, self).__init__()
        self.img_size_min = img_size_min
        self.scale_factor = scale_factor
        self.num_scale = num_scale
        self.nf = 16
        self.current_scale = 0

        self.size_list = [int(self.img_size_min * scale_factor**i) for i in range(num_scale + 1)]

        self.sub_discriminators = nn.ModuleList()

        first_discriminator = nn.ModuleList()

        first_discriminator.append(nn.Sequential(nn.Conv3d(1, self.nf, 3, 1, 1),
                                             nn.LeakyReLU(2e-1)))
        for _ in range(3):
            first_discriminator.append(nn.Sequential(nn.Conv3d(self.nf, self.nf, 3, 1, 1),
                                                 nn.BatchNorm3d(self.nf),
                                                 nn.LeakyReLU(2e-1)))

        first_discriminator.append(nn.Sequential(nn.Conv3d(self.nf, 1, 3, 1, 1)))

        first_discriminator = nn.Sequential(*first_discriminator)

        self.sub_discriminators.append(first_discriminator)

    def forward(self, x):
        out = self.sub_discriminators[self.current_scale](x)
        return out

    def progress(self):
        self.current_scale += 1
        # Lower scale discriminators are not used in later ... replace append to assign?
        # if self.current_scale % 2 == 0:
        #     self.nf *= 2

        tmp_discriminator = nn.ModuleList()
        tmp_discriminator.append(nn.Sequential(nn.Conv3d(1, self.nf, 3, 1, 1),
                                               nn.LeakyReLU(2e-1)))

        for _ in range(3):
            tmp_discriminator.append(nn.Sequential(nn.Conv3d(self.nf, self.nf, 3, 1, 1),
                                                   nn.BatchNorm3d(self.nf),
                                                   nn.LeakyReLU(2e-1)))

        tmp_discriminator.append(nn.Sequential(nn.Conv3d(self.nf, 1, 3, 1, 1)))

        tmp_discriminator = nn.Sequential(*tmp_discriminator)

        # if self.current_scale % 2 != 0:
        prev_discriminator = self.sub_discriminators[-1]

        # Initialize layers via copy
        if self.current_scale >= 1:
            tmp_discriminator.load_state_dict(prev_discriminator.state_dict())

        self.sub_discriminators.append(tmp_discriminator)
        print("DISCRIMINATOR PROGRESSION DONE")
