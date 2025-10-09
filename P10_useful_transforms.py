from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np


trans_totensor = transforms.ToTensor()###这里ToTensor是一个class，有__call__魔法方法，这里利用trans_totensor使得其可以像功能函数一样调用。
trans_toPILImage = transforms.ToPILImage()##转化为PIL格式
normalize = transforms.Normalize([1, 0.5, 3], [2, 1, 5])###需要给出标准差，RGB格式的图片按这个参数.

writer = SummaryWriter("logs")
img = Image.open("./hymenoptera_data/train/ants/0013035.jpg")
img_array = np.array(img)

## Normalize 函数学习
img_tensor = trans_totensor(img)
img_norm = normalize(img_tensor)
#img_norm = normalize(img_tensor)
#img_norm2 = normalize.forward(img_tensor)###似乎是一样的
print(img_norm[0][0][0])
print(img_tensor[0][0][0])#writer.add_image("img_ten", img_tensor, global_step=1)
writer.add_image("img_norm", img_norm, global_step=2)
#writer.add_image("img_norm2", img_norm2, global_step=1)

###Resize 函数学习
trans_resize = transforms.Resize((512, 512))##更新后的Resize类，可以处理PIL，nbarray，tensor等多种图片格式，但是如果是tensor类型需要[..., H, W]
###Resize 函数主要是用来将图片的大小进行缩放。
print(img.size)
img_resize = trans_resize(img)
print(img_resize)
print(img_tensor)
img_resize = trans_totensor(img_resize)##新版直接处理tensor就好了
print(img_resize)
writer.add_image("img_resize", img_resize, global_step=1)
## Compose Resize 2,实际上，compose就是一个工作流。
trans_resize_2 = transforms.Resize(512)##当参数只有一个时，代表你图片的最短边的长度。
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])###compose的参数为列表，且元素需要为Transforms格式。
img_resize_2 = trans_compose(img)##因为现在Resize可以传入tensor，所以可以直接用，不需要compose函数了，但是如果有其他的组合操作，依旧方便。

###RandomCrop 随机裁剪,用随机裁取的方式将图片变为指定大小，而resize是缩放
trans_Random = transforms.RandomCrop(512)###图片为PIL 或者 Tensor， 我的建议：善用Ctrl+左键，help()等查询使用方法。
trans_compose2 = transforms.Compose([trans_Random, trans_totensor])
img_compose = trans_compose2(img)
writer.add_image("img_compose", img_compose, global_step=1)
for i in range(10):
    img_compose = trans_compose2(img)
    writer.add_image("img_compose2", img_compose, global_step=i)



writer.close()