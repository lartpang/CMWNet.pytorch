# -*- coding: utf-8 -*-
# @Time    : 2021/6/3
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
import random
from functools import partial
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from utils import customized_worker_init_fn, get_datasets_info_with_keys


class JointCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask, depth):
        for t in self.transforms:
            image, mask, depth = t(image, mask, depth)
        return image, mask, depth


class JointResize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise RuntimeError("size参数请设置为int或者tuple")

    def __call__(self, image, mask, depth):
        image = image.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        depth = depth.resize(self.size, Image.BILINEAR)
        return image, mask, depth


class JointRandomHorizontallyFlip(object):
    def __call__(self, image, mask, depth):
        if random.random() < 0.5:
            return (
                image.transpose(Image.FLIP_LEFT_RIGHT),
                mask.transpose(Image.FLIP_LEFT_RIGHT),
                depth.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return image, mask, depth


class JointRandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, image, mask, depth):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            image.rotate(rotate_degree, Image.BILINEAR),
            mask.rotate(rotate_degree, Image.NEAREST),
            depth.rotate(rotate_degree, Image.BILINEAR),
        )


class RGBDTestDataset(Dataset):
    def __init__(self, root: Tuple[str, dict], shape: Dict[str, int]):
        super().__init__()
        self.datasets = get_datasets_info_with_keys(
            dataset_infos=[root], extra_keys=["mask", "depth"]
        )
        self.total_image_paths = self.datasets["image"]
        self.total_mask_paths = self.datasets["mask"]
        self.total_depth_paths = self.datasets["depth"]
        self.image_trans = transforms.Compose(
            [
                transforms.Resize(
                    (shape["h"], shape["w"]), interpolation=Image.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.depth_trans = transforms.Compose(
            [
                transforms.Resize(
                    (shape["h"], shape["w"]), interpolation=Image.BILINEAR
                ),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        curr_image_path = self.total_image_paths[index]
        curr_mask_path = self.total_mask_paths[index]
        curr_depth_path = self.total_depth_paths[index]

        curr_image = Image.open(curr_image_path)
        curr_depth = Image.open(curr_depth_path)
        if len(curr_image.split()) != 3:
            curr_image = curr_image.convert("RGB")
        if len(curr_depth.split()) == 3:
            curr_depth = curr_depth.convert("L")

        curr_size = curr_image.size[::-1]  # h, w

        curr_image_tensor = self.image_trans(curr_image)
        curr_depth_tensor = self.depth_trans(curr_depth)

        return dict(
            data=dict(
                image=curr_image_tensor,
                depth=curr_depth_tensor,
            ),
            info=dict(
                ori_shape=dict(h=curr_size[0], w=curr_size[1]),
                mask_path=curr_mask_path,
                mask_name=os.path.basename(curr_mask_path),
            ),
        )

    def __len__(self):
        return len(self.total_image_paths)


class RGBDTrainDataset(Dataset):
    def __init__(self, root: List[Tuple[str, dict]], shape: Dict[str, int]):
        super().__init__()
        self.datasets = get_datasets_info_with_keys(
            dataset_infos=root, extra_keys=["mask", "depth"]
        )
        self.total_image_paths = self.datasets["image"]
        self.total_mask_paths = self.datasets["mask"]
        self.total_depth_paths = self.datasets["depth"]

        self.Joint_trans = JointCompose(
            [
                JointResize(size=(shape["h"], shape["w"])),
                JointRandomHorizontallyFlip(),
                JointRandomRotate(10),
            ]
        )
        self.image_trans = transforms.Compose(
            [
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # 处理的是Tensor
            ]
        )
        self.mask_trans = transforms.ToTensor()
        self.depth_trans = transforms.ToTensor()

    def __getitem__(self, index):
        curr_image_path = self.total_image_paths[index]
        curr_mask_path = self.total_mask_paths[index]
        curr_depth_path = self.total_depth_paths[index]

        curr_image = Image.open(curr_image_path)
        curr_mask = Image.open(curr_mask_path)
        curr_depth = Image.open(curr_depth_path)
        if len(curr_image.split()) != 3:
            curr_image = curr_image.convert("RGB")
        if len(curr_mask.split()) == 3:
            curr_mask = curr_mask.convert("L")
        if len(curr_depth.split()) == 3:
            curr_depth = curr_depth.convert("L")

        curr_image, curr_mask, curr_depth = self.Joint_trans(
            image=curr_image, mask=curr_mask, depth=curr_depth
        )

        curr_image_tensor = self.image_trans(curr_image)
        curr_depth_tensor = self.depth_trans(curr_depth)
        curr_mask_tensor = self.mask_trans(curr_mask)

        return dict(
            data=dict(
                image=curr_image_tensor,
                mask=curr_mask_tensor,
                depth=curr_depth_tensor,
            )
        )

    def __len__(self):
        return len(self.total_image_paths)


def get_tr_loader(dataset_info, batch_size, num_workers, shape):
    dataset = RGBDTrainDataset(root=dataset_info, shape=shape)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=partial(customized_worker_init_fn, base_seed=0),
    )
    print(f"Length of Trainset: {len(dataset)}")
    return loader


def get_te_loader(dataset_info, shape, batch_size, num_workers) -> list:
    loaders = []
    for i, (te_data_name, te_data_path) in enumerate(dataset_info):
        dataset = RGBDTestDataset(root=(te_data_name, te_data_path), shape=shape)
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=partial(customized_worker_init_fn, base_seed=0),
        )
        print(f"Testing with testset: {te_data_name}: {len(dataset)}")
        loaders.append((te_data_name, te_data_path, loader))
    return loaders
