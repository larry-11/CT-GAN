from torch import nn
from torch.nn import functional as F
import torch


class Generator(nn.Module):
    def __init__(self, img_size_min, num_scale, scale_factor=4/3):
        torch.set_default_tensor_type(torch.DoubleTensor)
        super(Generator, self).__init__()
        self.img_size_min = img_size_min
        self.scale_factor = scale_factor
        self.num_scale = num_scale
        self.current_scale = 0

        self.size_list = [int(self.img_size_min * scale_factor**i) for i in range(num_scale + 1)]

        self.sub_generators = nn.ModuleList()
        
        first_generator = nn.ModuleList()

        first_generator.append(nn.Sequential(nn.Conv2d(int(self.size_list[self.current_scale]/6), 256, 3, 1, 1)))

        for _ in range(3):
            for __ in range(2):
                first_generator.append(nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1),
                                                    nn.BatchNorm2d(256),
                                                    nn.ReLU()))
            first_generator.append(nn.Sequential(nn.MaxPool2d(2)))
            
        for _ in range(3):
            first_generator.append(nn.Sequential(nn.ConvTranspose2d(256, 256, 3, 2, 1, 1)))
            for __ in range(2):
                first_generator.append(nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1),
                                                    nn.BatchNorm2d(256),
                                                    nn.ReLU()))

        first_generator.append(nn.Sequential(nn.Conv2d(256, int(self.size_list[self.current_scale]/6), 3, 1, 1)))

        first_generator = nn.Sequential(*first_generator)

        self.sub_generators.append(first_generator)


    def forward(self, z, img=None):
        x_list = []
        x_first = self.sub_generators[0](z[0].squeeze(0))
        x_first = F.interpolate(x_first.unsqueeze(0), (int(self.size_list[0]/6), self.size_list[0], self.size_list[0]), mode='trilinear', align_corners=True)
        x_list.append(x_first)

        if img is not None:
            x_inter = img
        else:
            x_inter = x_first

        for i in range(1, self.current_scale + 1):
            x_inter = F.interpolate(x_inter, (int(self.size_list[i]/6), self.size_list[i], self.size_list[i]), mode='trilinear', align_corners=True)
            x_prev = x_inter           
            # x_inter = F.pad(x_inter, [5, 5, 5, 5, 5, 5], value=0)
            # x_inter = x_inter + z[i]
            x_res = self.sub_generators[i](x_inter.squeeze(0))
            x_res = F.interpolate(x_res.unsqueeze(0), (int(self.size_list[i]/6), self.size_list[i], self.size_list[i]), mode='trilinear', align_corners=True)
            x_inter = x_res + x_prev
            x_list.append(x_inter)

        return x_list

    def progress(self):
        self.current_scale += 1

        tmp_generator = nn.ModuleList()

        tmp_generator.append(nn.Sequential(nn.Conv2d(int(self.size_list[self.current_scale]/6), 256, 3, 1, 1)))

        for _ in range(3):
            for __ in range(2):
                tmp_generator.append(nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1),
                                                    nn.BatchNorm2d(256),
                                                    nn.ReLU()))
            tmp_generator.append(nn.Sequential(nn.MaxPool2d(2)))
            
        for _ in range(3):
            tmp_generator.append(nn.Sequential(nn.ConvTranspose2d(256, 256, 3, 2, 1, 1)))
            for __ in range(2):
                tmp_generator.append(nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1),
                                                    nn.BatchNorm2d(256),
                                                    nn.ReLU()))

        tmp_generator.append(nn.Sequential(nn.Conv2d(256, int(self.size_list[self.current_scale]/6), 3, 1, 1)))

        tmp_generator = nn.Sequential(*tmp_generator)

        # if self.current_scale % 2 != 0:
        prev_generator = self.sub_generators[-1]

        # Initialize layers via copy
        if self.current_scale >= 1:
            cnnt = 1
            for net in prev_generator[1:-1]:    
                tmp_generator[cnnt].load_state_dict(net.state_dict())
                cnnt += 1

        self.sub_generators.append(tmp_generator)
        print("GENERATOR PROGRESSION DONE")
