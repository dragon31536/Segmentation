import numpy as np
import os
import scipy.misc as sm
import torch
from torch.utils.data import Dataset
import PIL
from PIL import Image, ImageOps
import torchvision.transforms.functional as F
import random
from labels import labels
# for test
import matplotlib.pyplot as plt
from torchvision import transforms, utils


class CityScapeDataset(Dataset):
    def __init__(self, root, img_WH, train=True, transform=None, target_transform=None):
        self.root = root
        self.img_WH = img_WH
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        if self.train:
            self.path = os.path.join(self.root, "training")
        else:
            self.path = os.path.join(self.root, "validation")

        self.filenames = []
        for img in os.listdir(self.image_dir):
            self.filenames.append(img)

        if len(self.filenames) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

    @property
    def image_dir(self):
        return os.path.join(self.path, "image")

    @property
    def label_dir(self):
        return os.path.join(self.path, "semantic_rgb")

    def encode_label(self, label):
        mask = np.zeros(label.shape[0:2])
        for _label in labels:
            mask[np.sum(label == np.array(_label.color),axis=2)==3] = _label.trainId  
        return mask
    
    def TensorFromPIL(self, im):
        if isinstance(im, PIL.Image.Image):
            im_np = np.array(im, dtype=np.long)
            return torch.from_numpy(im_np).long()
            
        elif isinstance(im, torch.Tensor):
            return im


    def __getitem__(self, index):
        img_path, label_path = os.path.join(self.image_dir, self.filenames[index]), os.path.join(self.label_dir, self.filenames[index])
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.img_WH, Image.NEAREST)
        print(type(self.transform))
        seed = np.random.randint(31536) # make a seed with numpy generator
        random.seed(seed)
        if self.transform is not None:
            img = self.transform(img)

        label = Image.open(label_path).convert('RGB')
        label = label.resize(self.img_WH, Image.NEAREST)
        label = np.array(label, dtype=np.uint8)
        label = self.encode_label(label)
        label = Image.fromarray(label)
        random.seed(seed)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, self.TensorFromPIL(label)
    
    def __len__(self):
        return len(self.filenames)




if __name__ == "__main__":
    
    preprocess = transforms.Compose([
        # transforms.Scale(256),
        transforms.Pad(8),
        transforms.RandomCrop((128, 256)),
        # transforms.ToTensor(),
        # normalize
        transforms.RandomHorizontalFlip(),
    ])
    dataset = CityScapeDataset('.\\data', (256, 128), transform=preprocess, target_transform=preprocess)
    img, label = dataset[2]

    fig, axes = plt.subplots(1,2)
    ax1=axes[0] 
    ax1.imshow(img)
    
    ax2=axes[1] 
    ax2.imshow(label)
    
    plt.show()