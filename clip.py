import numpy as np
from PIL import Image

path = '/data/shanyx/larry/SinGAN/code/logs/11_rec_pro/gen_stage4_iter4999.npy'
target = '/data/shanyx/larry/SinGAN/clip'
test   = np.load(path)[0][0] * 255
print(test.shape)
cnt = 0

for clip in test:
    im = Image.fromarray(clip)
    im = im.convert("L")
    im.save('/data/shanyx/larry/SinGAN/clip/{}.jpg'.format(str(cnt)))
    cnt += 1
