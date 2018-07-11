#!/usr/bin/env python
# -*- coding: utf-8 -*-


# @Time    : 2018/6/6 13:02
# @Author  : Allen
# @File    : train_data_generate.py

"""
训练数据生成
"""
import numpy as np
import os
from sklearn.preprocessing import scale
import original_data_generate
import fileUtils
from datetime import datetime

data_row = 20


#           #       0  # 'S_INFO_WINDCODE',                  # 'Wind代码'
#           #       1  # 'TRADE_DT',                         # '交易日期'

#     price    #       2  # 'S_DQ_ADJOPEN',                     # '复权开盘价(元)'
#     price    #       3  # 'S_DQ_ADJHIGH',                     # '复权最高价(元)'
#     price    #       4  # 'S_DQ_ADJLOW',                      # '复权最低价(元)'
#     price    #       5  # 'S_DQ_ADJCLOSE',                    # '复权收盘价(元)'
#     price    #       6  # 'S_DQ_AVGPRICE',                    # '均价(VWAP)'
#     price    #       7  # 'S_DQ_MA5',                         #  均线5天
#     price    #       8  # 'S_DQ_MA10',                       -#  均线10天
#     price    #       9  # 'S_DQ_MA20',                       -#  均线20天
#     price    #      10  # 'S_DQ_MA30',                       -#  均线30天
#     price    #      11  # 'S_DQ_MA60',                       -#  均线60天


#      rate    #      12  # 'S_DQ_FREETURNOVER',                # '换手率(基准.自由流通股本)'

#      rate    #      13  # 'DQ_CHANGE',                        # '一天涨跌幅'
#      rate    #      14  # 'DQ_2_CHANGE',                      # '两天涨跌幅'
#      rate    #      15  # 'DQ_MARKET_CHANGE',                 # '市场一天涨跌幅'

#       sum    #      16  # 'S_DQ_AMOUNT',                     -# '成交金额(千元)'
#       sum    #      17  # 'S_DQ_5_AMOUNT',                   -# '5天平均成交金额(元)'

#     count    #      18  # 'S_DQ_VOLUME',                      # '成交量(手)'
#     count    #      19  # 'S_DQ_5_VOLUME',                    # '5天平均成交量(手)'


#  idicator    #      20  # 'S_DQ_RSV',                         #
#  idicator    #      21  # 'S_DQ_RATIO',                       #  量比
#  idicator    #      22  # 'S_DQ_14_RSI',                      #  强弱指标
#  idicator    #      23  # 'S_DQ_14_RSY',                      #  心理线
#  idicator    #      24  # 'S_DQ_NLSR',                        #  多空比率净额

#  idicator    #      25  # 'S_K_ENTITY',                       #  K线实体
#  idicator    #      26  # 'S_K_UP_SHADOW',                    #  K线上影线
#  idicator    #      27  # 'S_K_DOWN_SHADOW',                  #  K线下影线
#  idicator    #      28  # 'S_DQ_PVT',                         #  价量趋势(PVT)指标
#  idicator    #      29  # 'S_DQ_60_LOW',                      #  60日最低价
#  idicator    #      30  # 'S_DQ_60_HIGH',                     #  60日最高价


def train_prepare(start_date=19900101, end_date=None, predict_split=None):
    """
    生成训练数据
    :param start_date:
    :param end_date:
    :param predict_split:
    :return:
    """
    data = original_data_generate.load_original_datas()
    print(data.dtype)
    klc = original_data_generate.read_klc()
    trade_dates = klc['TRADE_DT'].values
    data = data[(data[:, 1].astype('int64') >= start_date)]
    if end_date:
        data = data[(data[:, 1].astype('int64') <= end_date)]
    stocks = np.unique(data[:, 0])
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    stock_num = len(stocks)
    num = 0
    count = 0
    batch = 0
    start_time = datetime.now()
    for code in stocks:
        num = num + 1
        print(stock_num, '----------', num)
        cell = data[(data[:, 0] == code)]
        row_length = cell.shape[0]
        for i in np.arange(data_row, row_length - 2):
            start_date = cell[i - data_row, 1]
            end_date = cell[i + 1, 1]
            start_index = np.argwhere(trade_dates == start_date)
            klc_end_date = trade_dates[start_index + data_row + 1]
            k_line_scale_index = np.arange(25, 28)
            idicator_scale_index = [12, 20, 21, 22, 23, 24, 25, 28]
            price_scale_index = [2, 3, 4, 5, 6, 7, 29, 30]
            rate_scale_index = np.arange(13, 16)
            count_scale_index = np.arange(18, 20)
            idicator_name = '12-20-21-22-23-24-25~27-2~8-13~16-18~20'
            if end_date == klc_end_date:
                count = count + 1
                x_arr = cell[i - data_row: i]
                if ~(x_arr[:, 6] == x_arr[:, 7]).any():
                    x_arr_scale, y_arr = idicator_cell(cell, i, x_arr, price_scale_index, rate_scale_index, count_scale_index, idicator_scale_index, k_line_scale_index)
                    if int(cell[i - 1, 1]) < predict_split:
                        train_x.append(x_arr_scale)
                        train_y.append(y_arr)
                    else:
                        test_x.append(x_arr_scale)
                        test_y.append(y_arr)

        if (num == stock_num) | (num % 500 == 0):
            batch = batch + 1
            batch_dir = os.path.abspath(os.path.join(fileUtils.batch_data_dir, 'batch_data_' + str(batch)))
            np.savez(batch_dir,
                     train_x=np.array(train_x),
                     train_y=np.array(train_y),
                     test_x=np.array(test_x),
                     test_y=np.array(test_y))
            train_x.clear()
            train_y.clear()
            test_x.clear()
            test_y.clear()
            print(batch_dir, "save success")

    end_time = datetime.now()
    print('总计： ', count, start_time, end_time, '耗时: ', end_time - start_time)
    return idicator_name


def idicator_cell(cell, i, x_arr, price_scale_index, rate_scale_index, count_scale_index, idicator_scale_index, k_line_scale_index):
    price_scale = scale(x_arr[:, price_scale_index].flatten()).reshape(-1, price_scale_index.shape[0])
    rate_scale = x_arr[:, rate_scale_index]
    k_line_scale = scale(x_arr[:, k_line_scale_index].flatten()).reshape(-1, 3)
    count_scale = scale(x_arr[:, count_scale_index].flatten()).reshape(-1, 2)
    idicator_scale = scale(x_arr[:, idicator_scale_index])  # idicator_scale = scale(x_arr[:, [20, 21, 22, 23, 24]])
    x_arr_scale = np.column_stack((x_arr[:, [0, 1]]
                                   , idicator_scale
                                   , price_scale
                                   , rate_scale
                                   , count_scale
                                   , k_line_scale
                                   ))
    y_index = cell[i - 1, [0, 1, 17]]
    y_change = cell[i + 1, 14]
    y_arr = np.append(y_index, y_change)
    return x_arr_scale, y_arr


def save(idicator_name=None):
    """
    保存训练数据
    :return:
    """
    train_x, train_y, test_x, test_y = load_train_batch_datas()
    train_path = os.path.abspath(os.path.join(fileUtils.train_datas_dir, 'train_data_%s_.npz' % idicator_name))
    np.savez(train_path, train_x=train_x, train_y=train_y)
    print('train_data save success')
    test_path = os.path.abspath(os.path.join(fileUtils.test_datas_dir, 'test_data_%s_.npz' % idicator_name))
    np.savez(test_path, test_x=test_x, test_y=test_y)
    print('All file save success')


def load_train_datas(file_name=None):
    """
     加载训练数据
     :return:
     """
    print('loading start...')
    train_dir = os.path.abspath(os.path.join(fileUtils.train_datas_dir, file_name))
    train_data = np.load(train_dir)
    train_x = train_data['train_x']
    train_y = train_data['train_y']
    print('loading success...')
    print('train_x.shape', train_x.shape)
    print('train_y.shape', train_y.shape)
    return train_x, train_y


def load_test_datas(file_name=None):
    """
     加载测试数据
     :return:
     """
    print('loading start...')
    # 'test_data_%s_.npz' % idicator_name
    test_dir = os.path.abspath(os.path.join(fileUtils.test_datas_dir, file_name))
    test_data = np.load(test_dir)
    test_x = test_data['test_x']
    test_y = test_data['test_y']
    print('loading success...')
    print('test_x.shape', test_x.shape)
    print('test_y.shape', test_y.shape)
    return test_x, test_y


def load_train_batch_datas():
    """
     加载训练数据
     :return:
     """
    batch_data_dir = fileUtils.batch_data_dir
    file_names = sorted(os.listdir(batch_data_dir))
    file_num = len(file_names)
    for i in range(file_num):
        fname = file_names[i]
        print(fname, 'loading ...')
        path = os.path.abspath(os.path.join(batch_data_dir, fname))
        batch_data = np.load(path)
        if i == 0:
            train_x = batch_data['train_x']
            train_y = batch_data['train_y']
            test_x = batch_data['test_x']
            test_y = batch_data['test_y']
        else:
            train_x = np.concatenate((train_x, batch_data['train_x']), axis=0)
            train_y = np.concatenate((train_y, batch_data['train_y']), axis=0)
            test_x = np.concatenate((test_x, batch_data['test_x']), axis=0)
            test_y = np.concatenate((test_y, batch_data['test_y']), axis=0)
    return train_x, train_y, test_x, test_y

