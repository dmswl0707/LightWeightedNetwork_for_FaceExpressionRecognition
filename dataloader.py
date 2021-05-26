import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from preprocessing import *
import matplotlib.pyplot as plt

train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)
val_sampler = SubsetRandomSampler(val_idx)


dataset_sizes = {
    'train' : len(train_idx),
    'test' : len(test_idx),
    'val' : len(val_idx)
}

batch_size = 8

loaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, sampler=train_sampler),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, sampler=test_sampler),
    'val': torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, sampler=val_sampler),
}

import numpy as np


def imshow(img):
    img = img / 2 + 0.5  
    plt.imshow(np.transpose(img, (1, 2, 0)))

dataiter = iter(loaders['train'])
images, labels = dataiter.next()
print(images.shape,labels.shape)
images = images.numpy() 
fig = plt.figure(figsize=(25, 16))

for idx in np.arange(8):
    ax = fig.add_subplot(2, 4, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(categories[int(labels[idx])],fontsize=20,color='white')