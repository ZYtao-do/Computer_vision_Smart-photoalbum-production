import argparse
import torch
from torchvision.models import resnet18

import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image, ImageEnhance
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np


class ClassifyDpModel(nn.Module):  # 搭建网络模型
    def __init__(self):
        super(ClassifyWpModel, self).__init__()

        # 加载预训练模型
        structure = resnet18(pretrained=True)

        # 取掉model的后1层
        self.resnet_layer = nn.Sequential(*list(structure.children())[:-1])
        self.DATASET_PIC_lb = nn.Linear(in_features=512, out_features=5, bias=True)  # 五分类

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        classify = self.DATASET_PIC_lb(x)

        return classify


class DataSetDP(torch.utils.data.Dataset):  # 建立数据集
    def __init__(self, img_label, img_tranform, img_path, train_flag):
        super(DataSetDP, self).__init__()
        self.Database_label = img_label
        self.img_path = img_path.copy()
        self.img_tranform = img_tranform

        # print(images)

    def __len__(self):  # 获取index范围
        return len(self.img_path)

    def __getitem__(self, index):
        # 初始化相片的路径、类别的属性
        img_path = self.img_path[0]
        print(img_path)
        img_lb = np.array(int(self.Database_label[index]))
        img_label = torch.from_numpy(img_lb).long()

        img_all = Image.open(img_path).convert('RGB')
        # trick：图像亮度、饱和度、对比度增强
        img_all = ImageEnhance.Brightness(img_all)
        img_all = img_all.enhance(2)

        img_all = ImageEnhance.Color(img_all)
        img_all = img_all.enhance(2)

        img_all = ImageEnhance.Contrast(img_all)
        img_all = img_all.enhance(2)

        # Image Transform
        if self.img_tranform:
            img_all = self.img_tranform(img_all)

        return img_all, img_label


def evalute(model, test_loader, device):
    model.eval()

    with torch.no_grad():
        for i, (images, pic_label) in enumerate(test_loader):
            images, pic_label = images.to(device), pic_label.to(device)
            pred1 = model(images)

    return pred1


parse = argparse.ArgumentParser()
parse.add_argument("-f", "--filepath", help="file path")
args = parse.parse_args()
pic_path = args.filepath


# 转换图片格式以满足模型输入
img_tranform = transforms.Compose([
    transforms.Resize((340, 340)),
    transforms.RandomCrop((256, 256)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)
])

test_label, pic_path1 = [], []
# pic_path1.append("D:\\ProgramFiles\\Pycharm\\PycharmProjects\\computer_vision\\PictureSet_Classify\\Dataset_picture\\animals\\0001.jpg")
pic_path1.append(pic_path)
a = 1
test_label.append(a)
# print(type(test_label))
# print(type(pic_path1))
# print(len(test_label))
# print(len(pic_path1))
train_Dataset = DataSetDP(img_label=test_label, img_tranform=img_tranform, img_path=pic_path1,
                          train_flag=0)
img_all = DataLoader(train_Dataset, batch_size=64, shuffle=True)

model = torch.load("D:\\ProgramFiles\\Pycharm\\PycharmProjects\\computer_vision\\PictureSet_Classify\\model\\model_10.pth")
model.to("cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

model.eval()

for i, (images, labels) in enumerate(img_all):
    images = images.to("cpu")
    outputs = model(images)
    pre_label = outputs.argmax(1).numpy()

    print("各个类别的预测概率：", outputs)
    # print(torch.max(outputs, 1)) # 输出最大的预测值的索引 -> 即类别
    print("预测最终类别为：\n", int(pre_label))


# model = ClassifyDpModel()



# print(args.filepath)

# print(args.port)
