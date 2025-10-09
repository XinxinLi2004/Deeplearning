##利用 https://pytorch.org/domains/ 查询不同package所内置的数据集datasets。
##这里我们一直学习的是pytorch，这是一个主体，而用到torchvision，这是用于处理图像学习的，还有torchtext 用于文本（现在是torchtune了）以及***audio等。
##接下来会使用https://www.cs.toronto.edu/~kriz/cifar.html CIFAR10 这个数据集合，这是一个基础的物品识别的dataset。
from PIL import Image
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


writer = SummaryWriter("logs")

train_set = torchvision.datasets.CIFAR10(root='./dataset', transform=transforms.ToTensor(), train=True, download=True)
##torchvision package name, datasets, 调用内置的datasets，调用其中的CIFAR10这个dataset
# Train 参数： If True, creates dataset from training set, otherwise
#creates from test se
#download : 是否下载这个数据集到本地.
##transform 执行transforms的操作。这里是将数据集里面本身PIL的格式转换为summary add image可以识别的tensor格式
test_set  = torchvision.datasets.CIFAR10(root='./dataset', transform=transforms.ToTensor(), train=False, download=True)



# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()
# print(test_set.classes)

for i in range(10):
    img, target = test_set[i]
    writer.add_image("img_test", img, i)

writer.close()
