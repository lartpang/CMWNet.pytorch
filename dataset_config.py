# -*- coding: utf-8 -*-
# @Time    : 2020/12/20
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os

_RGBD_SOD_ROOT = "/home/lart/Datasets/Saliency/RGBDSOD"

LFSD = dict(
    root=os.path.join(_RGBD_SOD_ROOT, "LFSD"),
    image=dict(path=os.path.join(_RGBD_SOD_ROOT, "LFSD", "Image"), suffix=".jpg"),
    depth=dict(path=os.path.join(_RGBD_SOD_ROOT, "LFSD", "Depth"), suffix=".png"),
    mask=dict(path=os.path.join(_RGBD_SOD_ROOT, "LFSD", "Mask"), suffix=".png"),
)
NLPR_TR = dict(
    root=os.path.join(_RGBD_SOD_ROOT, "NLPR_FULL"),
    image=dict(path=os.path.join(_RGBD_SOD_ROOT, "NLPR_FULL", "Image"), suffix=".jpg"),
    depth=dict(path=os.path.join(_RGBD_SOD_ROOT, "NLPR_FULL", "Depth"), suffix=".png"),
    mask=dict(path=os.path.join(_RGBD_SOD_ROOT, "NLPR_FULL", "Mask"), suffix=".png"),
    index_file=os.path.join(_RGBD_SOD_ROOT, "nlpr_train_jw_name_list.lst"),
)
NJUD_TR = dict(
    root=os.path.join(_RGBD_SOD_ROOT, "NJUD_FULL"),
    image=dict(path=os.path.join(_RGBD_SOD_ROOT, "NJUD_FULL", "Image"), suffix=".jpg"),
    depth=dict(path=os.path.join(_RGBD_SOD_ROOT, "NJUD_FULL", "Depth"), suffix=".png"),
    mask=dict(path=os.path.join(_RGBD_SOD_ROOT, "NJUD_FULL", "Mask"), suffix=".png"),
    index_file=os.path.join(_RGBD_SOD_ROOT, "njud_train_jw_name_list.lst"),
)
NLPR_TE = dict(
    root=os.path.join(_RGBD_SOD_ROOT, "NLPR_FULL"),
    image=dict(path=os.path.join(_RGBD_SOD_ROOT, "NLPR_FULL", "Image"), suffix=".jpg"),
    depth=dict(path=os.path.join(_RGBD_SOD_ROOT, "NLPR_FULL", "Depth"), suffix=".png"),
    mask=dict(path=os.path.join(_RGBD_SOD_ROOT, "NLPR_FULL", "Mask"), suffix=".png"),
    index_file=os.path.join(_RGBD_SOD_ROOT, "nlpr_test_jw_name_list.lst"),
)
NJUD_TE = dict(
    root=os.path.join(_RGBD_SOD_ROOT, "NJUD_FULL"),
    image=dict(path=os.path.join(_RGBD_SOD_ROOT, "NJUD_FULL", "Image"), suffix=".jpg"),
    depth=dict(path=os.path.join(_RGBD_SOD_ROOT, "NJUD_FULL", "Depth"), suffix=".png"),
    mask=dict(path=os.path.join(_RGBD_SOD_ROOT, "NJUD_FULL", "Mask"), suffix=".png"),
    index_file=os.path.join(_RGBD_SOD_ROOT, "njud_test_jw_name_list.lst"),
)
RGBD135 = dict(
    root=os.path.join(_RGBD_SOD_ROOT, "RGBD135"),
    image=dict(path=os.path.join(_RGBD_SOD_ROOT, "RGBD135", "Image"), suffix=".jpg"),
    depth=dict(path=os.path.join(_RGBD_SOD_ROOT, "RGBD135", "Depth"), suffix=".png"),
    mask=dict(path=os.path.join(_RGBD_SOD_ROOT, "RGBD135", "Mask"), suffix=".png"),
)
SIP = dict(
    root=os.path.join(_RGBD_SOD_ROOT, "SIP"),
    image=dict(path=os.path.join(_RGBD_SOD_ROOT, "SIP", "Image"), suffix=".jpg"),
    depth=dict(path=os.path.join(_RGBD_SOD_ROOT, "SIP", "Depth"), suffix=".png"),
    mask=dict(path=os.path.join(_RGBD_SOD_ROOT, "SIP", "Mask"), suffix=".png"),
)
SSD = dict(
    root=os.path.join(_RGBD_SOD_ROOT, "SSD"),
    image=dict(path=os.path.join(_RGBD_SOD_ROOT, "SSD", "Image"), suffix=".jpg"),
    depth=dict(path=os.path.join(_RGBD_SOD_ROOT, "SSD", "Depth"), suffix=".png"),
    mask=dict(path=os.path.join(_RGBD_SOD_ROOT, "SSD", "Mask"), suffix=".png"),
)
STERE = dict(
    root=os.path.join(_RGBD_SOD_ROOT, "STERE"),
    image=dict(path=os.path.join(_RGBD_SOD_ROOT, "STERE", "Image"), suffix=".jpg"),
    depth=dict(path=os.path.join(_RGBD_SOD_ROOT, "STERE", "Depth"), suffix=".png"),
    mask=dict(path=os.path.join(_RGBD_SOD_ROOT, "STERE", "Mask"), suffix=".png"),
)
DUTRGBD_TE = dict(
    root=os.path.join(_RGBD_SOD_ROOT, "DUT-RGBD/Test"),
    image=dict(
        path=os.path.join(_RGBD_SOD_ROOT, "DUT-RGBD/Test", "Image"), suffix=".jpg"
    ),
    depth=dict(
        path=os.path.join(_RGBD_SOD_ROOT, "DUT-RGBD/Test", "Depth"), suffix=".png"
    ),
    mask=dict(
        path=os.path.join(_RGBD_SOD_ROOT, "DUT-RGBD/Test", "Mask"), suffix=".png"
    ),
)
DUTRGBD_TR = dict(
    root=os.path.join(_RGBD_SOD_ROOT, "DUT-RGBD/Train"),
    image=dict(
        path=os.path.join(_RGBD_SOD_ROOT, "DUT-RGBD/Train", "Image"), suffix=".jpg"
    ),
    depth=dict(
        path=os.path.join(_RGBD_SOD_ROOT, "DUT-RGBD/Train", "Depth"), suffix=".png"
    ),
    mask=dict(
        path=os.path.join(_RGBD_SOD_ROOT, "DUT-RGBD/Train", "Mask"), suffix=".png"
    ),
)

rgbd_sod_data = dict(
    nlprtrdmra=NLPR_TR,
    njudtrdmra=NJUD_TR,
    nlprtedmra=NLPR_TE,
    njudtedmra=NJUD_TE,
    lfsd=LFSD,
    rgbd135=RGBD135,
    sip=SIP,
    ssd=SSD,
    stere=STERE,
    dutrgbdtr=DUTRGBD_TR,
    dutrgbdte=DUTRGBD_TE,
)
