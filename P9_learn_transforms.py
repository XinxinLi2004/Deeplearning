import numpy as np
import matplotlib.pyplot as plt
import torchvision
import cv2#opencv这个package
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


#Tensor数据类型，张量。 To tensor,Transforms 适用于图像数据的处理。

img_path = "D:\\Deeplearning\\hymenoptera_data\\train\\ants\\0013035.jpg"
img = Image.open(img_path)
#img.show()
print(img)
tensor = transforms.ToTensor()##
tensor.img = tensor(img)

#tensor.img = transforms.ToPILImage(img)
print(tensor.img)
cv_img = cv2.imread(img_path)##读取图片后就是numpyarray形式的图片格式，不需要在转化。
##Totensor 这个函数无论是PIL格式还是numpy格式的图片都可以使用。

writer = SummaryWriter("logs")
writer.add_image("img1", tensor.img, 1)##img只为tensor or nparray格式
writer.close()
