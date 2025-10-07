from PIL import Image
import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
img_path = "D:\\Deeplearning\\hymenoptera_data\\train\\ants\\0013035.jpg"
img_PIL = Image.open(img_path)##使用PIL打开图像
img_array = np.array(img_PIL)##将PIL格式的图像变量用numpy转换为nparray类型
print(img_array.shape)#查看图片的大小格式


writer = SummaryWriter("logs")#一个变量
writer.add_image("test", img_array, 1, dataformats="HWC")###因为add_image默认图片的shape参数是CHW，这里需要用改参数。
writer.close()
for i in range(100):
    writer.add_scalar("y = 3x", 3*i, i)##
###在Terminal中使用tensorboard --logdir="*你的writer的参数"在浏览器中进行查看结果，类似Jupyter.

writer.close()