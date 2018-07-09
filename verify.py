#!/usr/bin/env python
# -*- coding: utf-8 -*-


# @Time    : 2018/6/12 9:08
# @Author  : Allen
# @File    : verify.py
import os
import numpy as np

import model_build
import train_data_generate
import fileUtils


def net_value(pred_y, true_y):
    """
    净值计算
    :param pred_y:
    :param true_y:
    :return:
    """
    data = np.column_stack((true_y, pred_y))
    fee_rate = 0.0014
    day_1_exp = 0.5
    day_2_exp = 0.5
    dates = np.unique(true_y[:, 1])
    days = len(dates)
    for day in range(days):
        date = dates[day]
        cell = data[data[:, 1] == date]
        cell = cell[np.argsort(-cell[:, true_y.shape[1]].astype(float))]
        top_mean = np.mean(cell[0:50, true_y.shape[1] - 1].astype(float))
        # print('mean===========', top_mean)
        if day % 2 == 1:
            day_1_exp = (top_mean - fee_rate + 1) * day_1_exp
        else:
            day_2_exp = (top_mean - fee_rate + 1) * day_2_exp
        daily_exp = day_1_exp + day_2_exp
    print(date, "日净值:", daily_exp, "当日股数: ", len(cell))
    return daily_exp

def itto_ryu_net_value(pred_y, true_y):
    """
    净值计算(一刀流)
    :param pred_y:
    :param true_y:
    :return:
    """
    data = np.column_stack((true_y, pred_y))
    fee_rate = 0.0014
    day_1_exp = 0.5
    day_2_exp = 0.5
    dates = np.unique(true_y[:, 1])
    days = len(dates)
    for day in range(days):
        date = dates[day]
        cell = data[data[:, 1] == date]
        cell = cell[np.argsort(-cell[:, 2].astype(float))]
        cell = cell[: 1000]
        cell = cell[np.argsort(-cell[:, true_y.shape[1]].astype(float))]
        top_mean = np.mean(cell[0:50, true_y.shape[1] - 1].astype(float))
        # print('mean===========', top_mean)
        if day % 2 == 1:
            day_1_exp = (top_mean - fee_rate + 1) * day_1_exp
        else:
            day_2_exp = (top_mean - fee_rate + 1) * day_2_exp
        daily_exp = day_1_exp + day_2_exp
    print(date, "日净值:", daily_exp, "当日股数: ", len(cell))
    return daily_exp
