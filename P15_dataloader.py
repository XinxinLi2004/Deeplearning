##前面dataset就是一个数据集，这里面的Dataloader顾名思义就是从数据集中选取数据的方式
# dataset (Dataset) – dataset from which to load the data.
#
# batch_size (int, optional) – how many samples per batch to load (default: 1).
#
# shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
# shuffle 每次完整训练一次数据集后，再次训练的时候数据是否打乱，默认是false，一般设置为True
# sampler (Sampler or Iterable, optional) – defines the strategy to draw samples from the dataset. Can be any Iterable with __len__ implemented. If specified, shuffle must not be specified.
#
# batch_sampler (Sampler or Iterable, optional) – like sampler, but returns a batch of indices at a time. Mutually exclusive with batch_size, shuffle, sampler, and drop_last.
#
# num_workers (int, optional) – how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
#设置进程数，默认0，即一个main process，根据自己电脑的性能来设置。
# drop_last (bool, optional) – set to True to drop the last incomplete batch, if the dataset size is not divisible
# by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: False)
# 数据总量和batchsize相除，这个参数决定没除尽的情况下是否舍弃余下部分，true为舍弃，false即为处理所有。

import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

##准备测试集
test_set = torchvision.datasets.CIFAR10("./dataset",train=False, transform=torchvision.transforms.ToTensor(), download=True)
###随机采样
test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

img, target = test_set[0]
print(img.shape)
print(target)
print(len(test_loader))
# step = 1
# for data in test_loader:
#     imgs, targets = data
#     print(imgs.shape)
#     print(targets)
#     writer.add_images("img_test_64_drop_last", imgs, step)###因为这里面的data是几个图片的集合，所有用images参数
#     step = step + 1

for epoch in range(2):####我们可以把完整的训练完一次数据集，称为一个epoch
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("img_test_64_Epoch: {}".format(epoch), imgs, step)
        step = step + 1





writer.close()


