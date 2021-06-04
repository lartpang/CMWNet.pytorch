# -*- coding: utf-8 -*-
# @Time    : 2021/6/3
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import numpy as np
from py_sod_metrics import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure

import functools
from datetime import datetime


class TimeRecoder:
    __slots__ = ["_start_time", "_has_start"]

    def __init__(self):
        self._start_time = datetime.now()
        self._has_start = False

    def start(self, msg=""):
        self._start_time = datetime.now()
        self._has_start = True
        if msg:
            print(f"[{self._start_time}] {msg}")

    def now_and_reset(self, pre_msg=""):
        if not self._has_start:
            raise AttributeError("You must call the `.start` method before the `.now_and_reset`!")
        self._has_start = False
        end_time = datetime.now()
        print(f"[{end_time}] {pre_msg} {end_time - self._start_time}")
        self.start()

    def now(self, pre_msg=""):
        if not self._has_start:
            raise AttributeError("You must call the `.start` method before the `.now`!")
        self._has_start = False
        end_time = datetime.now()
        print(f"[{end_time}] {pre_msg} {end_time - self._start_time}")

    @staticmethod
    def decorator(start_msg="", end_pre_msg=""):
        """as a decorator"""
        _temp_obj = TimeRecoder()

        def _timer(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                _temp_obj.start(start_msg)
                results = func(*args, **kwargs)
                _temp_obj.now(end_pre_msg)
                return results

            return wrapper

        return _timer


class CalTotalMetric(object):
    __slots__ = ["cal_mae", "cal_fm", "cal_sm", "cal_em", "cal_wfm"]

    def __init__(self):
        self.cal_mae = MAE()
        self.cal_fm = Fmeasure()
        self.cal_sm = Smeasure()
        self.cal_em = Emeasure()
        self.cal_wfm = WeightedFmeasure()

    def step(self, pred: np.ndarray, gt: np.ndarray, gt_path: str):
        assert pred.ndim == gt.ndim and pred.shape == gt.shape, (pred.shape, gt.shape, gt_path)
        assert pred.dtype == np.uint8, pred.dtype
        assert gt.dtype == np.uint8, gt.dtype

        self.cal_mae.step(pred, gt)
        self.cal_fm.step(pred, gt)
        self.cal_sm.step(pred, gt)
        self.cal_em.step(pred, gt)
        self.cal_wfm.step(pred, gt)

    def get_results(self, bit_width: int = 3) -> dict:
        fm = self.cal_fm.get_results()["fm"]
        wfm = self.cal_wfm.get_results()["wfm"]
        sm = self.cal_sm.get_results()["sm"]
        em = self.cal_em.get_results()["em"]
        mae = self.cal_mae.get_results()["mae"]
        results = {
            "Smeasure": sm,
            "wFmeasure": wfm,
            "MAE": mae,
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max(),
            "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max(),
        }

        def _round_w_zero_padding(_x):
            _x = str(_x.round(bit_width))
            _x += "0" * (bit_width - len(_x.split(".")[-1]))
            return _x

        results = {name: _round_w_zero_padding(metric) for name, metric in results.items()}
        return results


class AvgMeter(object):
    __slots__ = ["value", "avg", "sum", "count"]

    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, num=1):
        self.value = value
        self.sum += value * num
        self.count += num
        self.avg = self.sum / self.count


class TxtLogger(object):
    __slots__ = ["_path"]

    def __init__(self, path):
        self._path = path

    def __call__(self, msg, show=True):
        with open(self._path, "a") as f:
            f.write(f"{msg}")

        if show:
            print(msg)
