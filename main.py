# -*- coding: utf-8 -*-
# @Time    : 2021/6/3
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

import recorder
from dataset import get_te_loader, get_tr_loader
from dataset_config import rgbd_sod_data
from engine import TrainIterator
from model import CMWNet_V16
from utils import (
    change_lr,
    imresize,
    initialize_seed_cudnn,
    mapping_to_str,
    mkdir_if_not_exist,
    read_gray_array,
    save_array_as_image,
    to_device,
)


@torch.no_grad()
def test_model():
    model.eval()
    for data_name, data_path, te_loader in get_te_loader(
        dataset_info=dataset_info["test"]["path"],
        shape=dataset_info["test"]["shape"],
        batch_size=batch_size,
        num_workers=num_workers,
    ):
        pred_save_path = os.path.join(save_path, data_name)
        cal_total_seg_metrics = recorder.CalTotalMetric()
        for batch in tqdm(te_loader, total=len(te_loader), desc="TE", ncols=79):
            batch_images = to_device(batch["data"], device=model.device)
            logits = model(data=batch_images)
            probs = logits.sigmoid().squeeze(1).cpu().detach().numpy()

            for i, seg_pred in enumerate(probs):
                mask_path = batch["info"]["mask_path"][i]
                mask_array = read_gray_array(mask_path, dtype=np.uint8)
                mask_h, mask_w = mask_array.shape
                seg_pred = imresize(
                    seg_pred, target_h=mask_h, target_w=mask_w, interp="linear"
                )

                if pred_save_path:  # 这里的pred_save_path包含了数据集名字
                    pred_name = batch["info"]["mask_name"][i]
                    save_array_as_image(
                        data_array=seg_pred,
                        save_name=pred_name,
                        save_dir=pred_save_path,
                    )

                seg_pred = (seg_pred * 255).astype(np.uint8)
                cal_total_seg_metrics.step(seg_pred, mask_array, mask_path)
        seg_results = cal_total_seg_metrics.get_results()
        te_txt(
            msg=f"Results on the testset({data_name}): {mapping_to_str(data_path)}\n{seg_results}",
            show=True,
        )


def train_model():
    tr_loader = get_tr_loader(
        dataset_info=dataset_info["train"]["path"],
        shape=dataset_info["train"]["shape"],
        batch_size=batch_size,
        num_workers=num_workers,
    )
    epoch_length = len(tr_loader)

    model.to(model.device)

    data_iterator = TrainIterator(
        start_epoch=0, num_iters=num_iters, data_iterable=tr_loader
    )
    print(f"TrainIterator:\n{data_iterator}")

    time_logger = recorder.TimeRecoder()
    loss_recorder = recorder.AvgMeter()
    for curr_iter, iter_in_epoch, batch in data_iterator:
        curr_epoch = data_iterator.curr_epoch

        if data_iterator.is_first_iter_in_epoch:
            print(f"Exp_Name: {exp_name}")
            time_logger.start(msg="An Epoch Start...")
            loss_recorder.reset()
            model.train()

        change_lr(
            optimizer=optim, curr_idx=curr_iter, total_num=num_iters, lr_decay=lr_decay
        )

        batch_data = to_device(data=batch["data"], device=model.device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            probs, loss, loss_str = model(data=batch_data)
            loss = loss / grad_acc_step
        scaler.scale(loss).backward()

        # Accumulates scaled gradients.
        if (curr_iter + 1) % grad_acc_step == 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

        item_loss = loss.item()
        data_shape = batch_data["image"].shape
        loss_recorder.update(value=item_loss, num=data_shape[0])

        if log_interval > 0:
            if (
                curr_iter % log_interval == 0
                or data_iterator.is_last_iter_in_epoch
                or data_iterator.is_last_iter
            ):
                msg = (
                    f"[I:{curr_iter}:{num_iters} ({iter_in_epoch}/{epoch_length}/{curr_epoch})]"
                    f"[Lr:{[round(x['lr'], 5) for x in optim.param_groups]}]"
                    f"[M:{loss_recorder.avg:.5f},C:{item_loss:.5f}]"
                    f"{list(data_shape)}"
                    f"\n{loss_str}"
                )
                tr_txt(msg, show=True)

        if data_iterator.is_last_iter_in_epoch:
            torch.save(model.state_dict(), final_state_path)
            time_logger.now(pre_msg="An Epoch End...")


load_from = ""
use_amp = True
num_iters = 22500
batch_size = 4
lr = 5e-3
lr_decay = 0.9
momentum = 0.9
weight_decay = 5e-4
num_workers = 2
shape = dict(h=288, w=288)
grad_acc_step = 1
log_interval = 20
exp_name = (
    f"cmwnet_amp{'Y' if use_amp else 'N'}_i{num_iters}_bs{batch_size}_lr{lr}_ld{lr_decay}_wd{weight_decay}"
    f"_h{shape['h']}w{shape['w']}_ga{grad_acc_step}"
)

proj_root = os.path.dirname(os.path.abspath(__file__))
ckpt_path = os.path.join(proj_root, "output")
pth_log_path = os.path.join(ckpt_path, exp_name)
save_path = os.path.join(pth_log_path, "pre")
pth_path = os.path.join(pth_log_path, "pth")
mkdir_if_not_exist(path_list=[ckpt_path, pth_path, save_path, pth_path])

final_state_path = os.path.join(pth_path, "state_final.pth")
tr_log_path = os.path.join(pth_log_path, f"tr_{str(datetime.now())[:10]}.txt")
te_log_path = os.path.join(pth_log_path, f"te_{str(datetime.now())[:10]}.txt")

dataset_info = dict(
    train=dict(
        shape=dict(h=288, w=288),
        path=[
            ("njudtr", rgbd_sod_data["njudtr"]),
            ("nlprtr", rgbd_sod_data["nlprtr"]),
        ],
    ),
    test=dict(
        shape=dict(h=288, w=288),
        path=[
            ("njudte", rgbd_sod_data["njudte"]),
            ("nlprte", rgbd_sod_data["nlprte"]),
            ("lfsd", rgbd_sod_data["lfsd"]),
            ("rgbd135", rgbd_sod_data["rgbd135"]),
            ("sip", rgbd_sod_data["sip"]),
            ("ssd", rgbd_sod_data["ssd"]),
            ("stere", rgbd_sod_data["stere"]),
        ],
    ),
)

initialize_seed_cudnn(seed=0, deterministic=True)
model = CMWNet_V16()
model.device = "cuda:0"

tr_txt = recorder.TxtLogger(path=tr_log_path)
te_txt = recorder.TxtLogger(path=te_log_path)

if load_from:
    model.load_state_dict(torch.load(load_from, map_location="cpu"))
else:
    optim = torch.optim.SGD(
        params=[
            # 不对bias参数执行weight decay操作，weight decay主要的作用就是通过对网络
            # 层的参数（包括weight和bias）做约束（L2正则化会使得网络层的参数更加平滑）达
            # 到减少模型过拟合的效果。
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if name[-4:] == "bias"
                ],
                "lr": 2 * lr,
                "weight_decay": 0,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if name[-4:] != "bias"
                ],
                "lr": lr,
                "weight_decay": weight_decay,
            },
        ],
        lr=lr,
        momentum=momentum,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    train_model()

test_model()
print(f"{datetime.now()}: End training...")
