from PIL import Image
from torchvision import transforms

trans_totensor = transforms.ToTensor()###这里ToTensor是一个class，有__call__魔法方法，这里利用trans_totensor使得其可以像功能函数一样调用。

