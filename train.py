import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset import CityScapeDataset
from model import FCNs, VGGNet
from torchvision import transforms, utils
from torch import Tensor
from labels import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size, n_class, h, w = 1, 19, 480, 320
vgg_model = VGGNet(requires_grad=True)
fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(fcn_model.parameters(), lr=1e-3, momentum=0.9)


preprocess = transforms.Compose([
    # transforms.Scale(256),
    # transforms.ToTensor(),
    # normalize
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
dataset = CityScapeDataset('.\\data', (h, w), transform=preprocess, target_transform=transforms.RandomHorizontalFlip())


img, label = dataset[0]

def OneHot(batch_size, H, W, n_class, y):

    y_onehot = torch.FloatTensor(batch_size, n_class, H, W).to(device)
    y_onehot.zero_()
    ones = torch.ones(y.size()[2:]).to(device)
    print(ones.shape)
    print(ones)
    y_onehot.scatter_(1, y, ones)

    return y_onehot


for iter, (img, label) in enumerate(dataset):
    img = Tensor(img).to(device)
    label = label.cuda()
    img = img.reshape((1,) + img.size())
    label = label.reshape((1,) + label.size())
    print(label.shape)
    optimizer.zero_grad()
    output = fcn_model(img)
    print(output.shape)
    # output = nn.functional.sigmoid(output)
    print( OneHot(batch_size=batch_size, H=h, W=w, n_class=n_class, y=label) )
    break
    # loss = criterion(output, label)
    # print(loss)
    # loss.backward()
    # print("iter{}, loss {}".format(iter, loss.data[0]))
    # optimizer.step()
