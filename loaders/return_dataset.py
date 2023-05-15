import os
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A

from loaders.dataset import Imagelists_VISDA


def get_data_transforms(args, transform='train'):
    crop_size = 224
    load_size = int(crop_size * 1.15)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': A.Compose([
            A.augmentations.geometric.resize.Resize(width=load_size, height=load_size),
            A.HorizontalFlip(),
            A.RandomCrop(width=crop_size, height=crop_size),
            A.Normalize(mean, std),
            ToTensorV2(),
        ]),
        'test': A.Compose([
            A.augmentations.geometric.resize.Resize(width=load_size, height=load_size),
            A.CenterCrop(width=crop_size, height=crop_size),
            A.Normalize(mean, std),
            ToTensorV2(),
        ])
    }

    if isinstance(transform, str):
        return data_transforms[transform]
    else:
        return transform


def get_loader(args, dataset, shuffle=True, bs=None, nw=None, dl=True, sampler=None):
    if (bs is None):
        bs = args.bs

    if (nw is None):
        nw = args.num_workers

    loader = torch.utils.data.DataLoader(dataset, batch_size=min(bs, len(dataset)), sampler=sampler, num_workers=nw,
                                         shuffle=shuffle, drop_last=dl)
    return loader


def get_dataset(args, split=None, transform='train', empty=False):
    if empty:
        dataset = Imagelists_VISDA(transform=get_data_transforms(args, transform), empty=True)
    else:
        base_path = './data/{}'.format(args.dataset)
        root = './data/{}/'.format(args.dataset)
        domain = args.source if 'source' in split else args.target
        appendix = '{}.txt'.format(split.split('_')[-1])
        image_set_file = "{}/{}_{}".format(base_path, domain, appendix)
        assert (os.path.exists(image_set_file)), "file \'{}\' not exist".format(image_set_file)
        dataset = Imagelists_VISDA(image_set_file, root=root, transform=get_data_transforms(args, transform))

    return dataset
