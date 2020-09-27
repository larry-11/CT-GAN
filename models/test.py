import torch
import torchsummary
import numpy as np
from torch.nn import functional as F
from generator import Generator
from discriminator import Discriminator

discriminator = Discriminator()
generator = Generator(25, 8, 4/3)

networks = [discriminator, generator]

scale_factor = 4/3
tmp_scale = 250 / 25
num_scale = int(np.round(np.log(tmp_scale) / np.log(scale_factor)))
size_list = [int(25 * scale_factor**i) for i in range(num_scale + 1)]

# print(size_list)

z_fix_list = [F.pad(torch.randn(1, 3, size_list[0], size_list[0]), [5, 5, 5, 5], value=0)]
zero_list = [F.pad(torch.zeros(1, 3, size_list[zeros_idx], size_list[zeros_idx]), [5, 5, 5, 5], value=0) for zeros_idx in range(1, num_scale + 1)]

# z_fix_list = [t.numpy() for t in z_fix_list]
# zero_list = [t.numpy() for t in zero_list]
z_fix_list = z_fix_list + zero_list
# # print(len(z_fix_list))
# # print(len(zero_list))

# stage = 1
# x_in = torch.randn(1, 159, 512, 512)
# x_in = F.interpolate(x_in, (size_list[stage], size_list[stage]), mode='bilinear', align_corners=True)
# print('size stage:', size_list[stage])
# print(x_in.shape)


# print(z_fix_list[0].shape)
# print(zero_list[0].shape)

# print(discriminator)
# print(generator)
# print(generator.sub_generators[0])
# print(discriminator.sub_generators[0])
z = torch.from_numpy(np.random.randn(1,3,25,25))
out = discriminator.sub_discriminators[0](z)
print(out.shape)

# torchsummary.summary(generator.sub_generators[0].cpu(), (3,32,32))