# -*- coding: utf-8 -*-
# @Time    : 2021/6/3
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
import random
from collections import abc, defaultdict
from numbers import Number
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def load_params_for_new_conv(conv_layer, new_conv_layer, in_dim):
    o, i, k_h, k_w = new_conv_layer.weight.shape
    ori_weight = conv_layer.weight
    if in_dim < 3:
        new_weight = ori_weight[:, :in_dim]
    else:
        new_weight = torch.repeat_interleave(
            ori_weight, repeats=in_dim // i + 1, dim=1
        )[:, :in_dim]
    new_conv_layer.weight = nn.Parameter(new_weight)
    new_conv_layer.bias = conv_layer.bias


def cus_sample(
    feat: torch.Tensor,
    mode=None,
    factors=None,
    *,
    interpolation="bilinear",
    align_corners=False,
) -> torch.Tensor:
    """
    :param feat: 输入特征
    :param mode: size/scale
    :param factors: shape list for mode=size or scale list for mode=scale
    :param interpolation:
    :param align_corners: 具体差异可见https://www.yuque.com/lart/idh721/ugwn46
    :return: the resized tensor
    """
    if mode is None:
        return feat
    else:
        if factors is None:
            raise ValueError(
                f"factors should be valid data when mode is not None, but it is {factors} now."
            )

    interp_cfg = {}
    if mode == "size":
        if isinstance(factors, Number):
            factors = (factors, factors)
        assert isinstance(factors, (list, tuple)) and len(factors) == 2
        factors = [int(x) for x in factors]
        if factors == list(feat.shape[2:]):
            return feat
        interp_cfg["size"] = factors
    elif mode == "scale":
        assert isinstance(factors, (int, float))
        if factors == 1:
            return feat
        recompute_scale_factor = None
        if isinstance(factors, float):
            recompute_scale_factor = False
        interp_cfg["scale_factor"] = factors
        interp_cfg["recompute_scale_factor"] = recompute_scale_factor
    else:
        raise NotImplementedError(f"mode can not be {mode}")

    if interpolation == "nearest":
        if align_corners is False:
            align_corners = None
        assert align_corners is None, (
            "align_corners option can only be set with the interpolating modes: "
            "linear | bilinear | bicubic | trilinear, so we will set it to None"
        )
    try:
        result = F.interpolate(
            feat, mode=interpolation, align_corners=align_corners, **interp_cfg
        )
    except NotImplementedError as e:
        print(
            f"shape: {feat.shape}\n"
            f"mode={mode}\n"
            f"factors={factors}\n"
            f"interpolation={interpolation}\n"
            f"align_corners={align_corners}"
        )
        raise e
    except Exception as e:
        raise e
    return result


def read_gray_array(
    path, div_255=False, to_normalize=False, thr=-1, dtype=np.float32
) -> np.ndarray:
    """
    1. read the binary image with the suffix `.jpg` or `.png`
        into a grayscale ndarray
    2. (to_normalize=True) rescale the ndarray to [0, 1]
    3. (thr >= 0) binarize the ndarray with `thr`
    4. return a gray ndarray (np.float32)
    """
    assert path.endswith(".jpg") or path.endswith(".png")
    assert not div_255 or not to_normalize
    gray_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if div_255:
        gray_array = gray_array / 255

    if to_normalize:
        gray_array = gray_array / 255
        gray_array_min = gray_array.min()
        gray_array_max = gray_array.max()
        if gray_array_max != gray_array_min:
            gray_array = (gray_array - gray_array_min) / (
                gray_array_max - gray_array_min
            )

    if thr >= 0:
        gray_array = gray_array > thr

    return gray_array.astype(dtype)


def read_color_array(path: str):
    assert path.endswith(".jpg") or path.endswith(".png")
    bgr_array = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    return rgb_array


def to_device(data, device):
    """
    :param data:
    :param device:
    :return:
    """
    if isinstance(data, (tuple, list)):
        return to_device(data, device)
    elif isinstance(data, dict):
        return {name: to_device(item, device) for name, item in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=True)
    else:
        raise TypeError(
            f"Unsupported type {type(data)}. Only support Tensor or tuple/list/dict containing Tensors."
        )


def save_array_as_image(data_array: np.ndarray, save_name: str, save_dir: str):
    """
    save the ndarray as a image

    Args:
        data_array: np.float32 the max value is less than or equal to 1
        save_name: with special suffix
        save_dir: the dirname of the image path
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    if data_array.dtype != np.uint8:
        if data_array.max() > 1:
            raise Exception("the range of data_array has smoe errors")
        data_array = (data_array * 255).astype(np.uint8)
    cv2.imwrite(save_path, data_array)


def imresize(image_array: np.ndarray, target_h, target_w, interp="linear"):
    _interp_mapping = dict(
        linear=cv2.INTER_LINEAR,
        cubic=cv2.INTER_CUBIC,
        nearst=cv2.INTER_NEAREST,
    )
    assert interp in _interp_mapping, f"Only support interp: {_interp_mapping.keys()}"
    resized_image_array = cv2.resize(
        image_array, dsize=(target_w, target_h), interpolation=_interp_mapping[interp]
    )
    return resized_image_array


def get_data_from_txt(path: str) -> list:
    """
    读取文件中各行数据，存放到列表中
    """
    lines = []
    with open(path, encoding="utf-8", mode="r") as f:
        line = f.readline().strip()
        while line:
            lines.append(line)
            line = f.readline().strip()
    return lines


def get_name_list_from_dir(path: str) -> list:
    """直接从文件夹中读取所有文件不包含扩展名的名字"""
    return [os.path.splitext(x)[0] for x in os.listdir(path)]


def get_datasets_info_with_keys(dataset_infos: List[tuple], extra_keys: list) -> dict:
    """
    从给定的包含数据信息字典的列表中，依据给定的extra_kers和固定获取的key='image'来获取相应的路径
    Args:
        dataset_infos: 数据集列表
        extra_keys: 除了'image'之外的需要获取的信息名字

    Returns:
        包含指定信息的绝对路径列表
    """

    # total_keys = tuple(set(extra_keys + ["image"]))
    # e.g. ('image', 'mask')
    def _get_intersection(list_a: list, list_b: list, to_sort: bool = True):
        """返回两个列表的交集，并可以随之排序"""
        intersection_list = list(set(list_a).intersection(set(list_b)))
        if to_sort:
            return sorted(intersection_list)
        return intersection_list

    def _get_info(dataset_info: dict, extra_keys: list, path_collection: defaultdict):
        """
        配合get_datasets_info_with_keys使用，针对特定的数据集的信息进行路径获取

        Args:
            dataset_info: 数据集信息字典
            extra_keys: 除了'image'之外的需要获取的信息名字
            path_collection: 存放收集到的路径信息
        """
        total_keys = tuple(set(extra_keys + ["image"]))
        # e.g. ('image', 'mask')

        infos = {}
        for k in total_keys:
            assert k in dataset_info, f"{k} is not in {dataset_info}"
            infos[k] = dict(dir=dataset_info[k]["path"], ext=dataset_info[k]["suffix"])

        if (index_file_path := dataset_info.get("index_file", None)) is not None:
            image_names = get_data_from_txt(index_file_path)
        else:
            image_names = get_name_list_from_dir(infos["image"]["dir"])

        if "mask" in total_keys:
            mask_names = get_name_list_from_dir(infos["mask"]["dir"])
            image_names = _get_intersection(image_names, mask_names)

        for i, name in enumerate(image_names):
            for k in total_keys:
                path_collection[k].append(
                    os.path.join(infos[k]["dir"], name + infos[k]["ext"])
                )

    path_collection = defaultdict(list)
    for dataset_name, dataset_info in dataset_infos:
        print(f"Loading data from {dataset_name}: {dataset_info['root']}")

        _get_info(
            dataset_info=dataset_info,
            extra_keys=extra_keys,
            path_collection=path_collection,
        )
    return path_collection


def customized_worker_init_fn(worker_id, base_seed):
    set_seed_for_lib(base_seed + worker_id)


def set_seed_for_lib(seed):
    random.seed(seed)
    np.random.seed(seed)
    # 为了禁止hash随机化，使得实验可复现。
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def initialize_seed_cudnn(seed, deterministic):
    assert isinstance(deterministic, bool) and isinstance(seed, int)
    set_seed_for_lib(seed)
    if not deterministic:
        print("We will use `torch.backends.cudnn.benchmark`")
    else:
        print("We will not use `torch.backends.cudnn.benchmark`")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def mapping_to_str(
    mapping: abc.Mapping, *, prefix: str = "    ", lvl: int = 0, max_lvl: int = 1
) -> str:
    """
    Print the structural information of the dict.
    """
    sub_lvl = lvl + 1
    cur_prefix = prefix * lvl
    sub_prefix = prefix * sub_lvl

    if lvl == max_lvl:
        sub_items = str(mapping)
    else:
        sub_items = ["{"]
        for k, v in mapping.items():
            sub_item = sub_prefix + k + ": "
            if isinstance(v, abc.Mapping):
                sub_item += mapping_to_str(
                    v, prefix=prefix, lvl=sub_lvl, max_lvl=max_lvl
                )
            else:
                sub_item += str(v)
            sub_items.append(sub_item)
        sub_items.append(cur_prefix + "}")
        sub_items = "\n".join(sub_items)
    return sub_items


def change_lr(optimizer, curr_idx, total_num, lr_decay):
    ratio = pow((1 - float(curr_idx) / total_num), lr_decay)
    optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * ratio
    optimizer.param_groups[1]["lr"] = optimizer.param_groups[0]["lr"]


def mkdir_if_not_exist(path_list: list):
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)
