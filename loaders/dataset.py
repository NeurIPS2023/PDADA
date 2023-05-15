import numpy as np
from PIL import Image
import torch


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        images = [(x.split(' ')[0], int(x.split(' ')[1])) for x in f.readlines()]
    return images


class Imagelists_VISDA(object):
    def __init__(self, image_list=None, root="./data/", transform=None, target_transform=None, empty=False):
        self.empty = empty
        if self.empty:
            self.imgs = np.empty((1, 2), dtype='<U1000')
        else:
            self.imgs = make_dataset_fromlist(image_list)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root

    def __getitem__(self, index):
        assert (self.transform is not None)
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(image=np.array(img))['image']
        if self.target_transform is not None:
            target = self.target_transform(target)

        index = torch.tensor(index) if isinstance(index, int) else index
        return img, target, index, path

    def __len__(self):
        return len(self.imgs)

    def add_item(self, addition):
        for i in range(len(addition)):
            addition[i, 1] = int(addition[i, 1].item())
        if self.empty:
            self.imgs = addition
            self.empty = False
        else:
            self.imgs = np.concatenate((self.imgs, addition), axis=0)
        return self.imgs

    def remove_item(self, reduced):
        self.imgs = np.delete(self.imgs, reduced, axis=0)
        return self.imgs
