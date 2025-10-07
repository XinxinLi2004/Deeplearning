import os
import torch
# from torchvision import transforms
from PIL import Image
from torch.utils.data import dataloader, Dataset


class mydata(Dataset):

    def __init__(self, root_dir, label):
        self.root_dir = root_dir
        self.label = label
        self.dir_path = os.path.join(root_dir, label)
        self.img_list = os.listdir(self.dir_path)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_item_path = os.path.join(self.root_dir, self.label, img_name)
        img = Image.open(img_item_path)
        return img, self.label
    
    def __len__(self):
        return len(self.img_list)

    def txtoutput(self):
        out_dir = root_dir + "\\" + label +"_label"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for i in self.img_list:
            filename = i.split('.')[0]
            with open(os.path.join(out_dir, filename+'.txt'), 'w', encoding='utf-8') as f:
                f.write(label)

root_dir = "D:\\Deeplearning\\hymenoptera_data\\train"
label = "ants"
ant_data = mydata(root_dir, label)
ant_data.txtoutput()
label = "bees"
bee_data = mydata(root_dir, label)
##合并训练数据集
train_data = ant_data + bee_data

bee_data.txtoutput()


print(ant_data[0])


