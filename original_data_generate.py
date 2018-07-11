#!/usr/bin/env python
# -*- coding: utf-8 -*-


# @Time    : 2018/6/5 11:31
# @Author  : Allen
# @File    : original_data_generate.py

import os
import pandas as pd
import numpy as np
import pymysql
import fileUtils


def read(start_date=19900101, end_date=None):
    """
    读取股票数据
    :return: dataframe
    """
    file_name = 'ashareeodprices'
    path = os.path.abspath(os.path.join(fileUtils.csv_dir, file_name))

    names = ['S_INFO_WINDCODE',  # 'Wind代码'
             'TRADE_DT',  # '交易日期'
             'S_DQ_ADJFACTOR',  # 复权因子
             'S_DQ_ADJOPEN',  # '复权开盘价(元)'
             'S_DQ_ADJHIGH',  # '复权最高价(元)'
             'S_DQ_ADJLOW',  # '复权最低价(元)'
             'S_DQ_ADJCLOSE',  # '复权收盘价(元)'
             'S_DQ_AVGPRICE',  # '均价(VWAP)'
             'S_DQ_FREETURNOVER',  # '换手率(基准.自由流通股本)'
             'S_DQ_PCTCHANGE',  # 涨跌幅
             'S_DQ_AMOUNT',  # 成交金额(千元)
             'S_DQ_VOLUME',  # 成交量
             ]

    conn = pymysql.connect(host='115.238.110.42', user='ai_user', passwd='123456', db='daily_data',
                           port=3307)  # 连接

    cursor = conn.cursor()
    sql = 'SELECT' \
          '	prices.S_INFO_WINDCODE,' \
          '	prices.TRADE_DT,' \
          ' prices.S_DQ_ADJFACTOR,' \
          '	prices.S_DQ_ADJOPEN,' \
          '	prices.S_DQ_ADJHIGH,' \
          '	prices.S_DQ_ADJLOW,' \
          '	prices.S_DQ_ADJCLOSE,' \
          '	prices.S_DQ_AVGPRICE,' \
          '	indicator.S_DQ_FREETURNOVER,' \
          ' prices.S_DQ_PCTCHANGE,' \
          '	prices.S_DQ_AMOUNT,' \
          '	prices.S_DQ_VOLUME ' \
          'FROM' \
          '	ashareeodprices AS prices' \
          '	LEFT JOIN ashareeodderivativeindicator AS indicator ON prices.S_INFO_WINDCODE = indicator.S_INFO_WINDCODE ' \
          '	AND prices.TRADE_DT = indicator.TRADE_DT ' \
          'WHERE' \
          '	1 = 1 ' \
          '	AND prices.TRADE_DT BETWEEN (%s) ' \
          '	AND (%s) ' \
          'ORDER BY' \
          '	prices.S_INFO_WINDCODE,' \
          '	prices.TRADE_DT' % (start_date, end_date)
    cursor.execute(sql)
    nRet = cursor.fetchall()
    df = pd.DataFrame(list(nRet), columns=names)
    conn.commit()
    cursor.close()
    conn.close()

    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'], axis=0, ascending=True)
    df = df.fillna(value=0)
    df["S_INFO_WINDCODE"] = df["S_INFO_WINDCODE"].str.split(".", n=1).str[0]  # 按第一个空格分割
    df = df.astype(np.float64)
    return df


def read_klc():
    """
    读取上证指数数据
    :return: klc dataframe
    """
    file_name = '999999'
    path = os.path.abspath(os.path.join(fileUtils.csv_dir, file_name))
    names = ['TRADE_DT',  # '交易日期'
             'S_DQ_OPEN',  # '开盘价(元)'
             'S_DQ_HIGH',  # '最高价(元)'
             'S_DQ_LOW',  # '最低价(元)'
             'S_DQ_CLOSE',  # '收盘价(元)'
             'S_DQ_VOLUME',  # '成交量(手)'
             'S_DQ_AMOUNT',  # '成交金额(千元)'
             ]
    df = pd.read_csv(path + '.csv',
                     encoding='utf8',
                     names=names,
                     low_memory=False)
    df = df.dropna()
    df = df.astype(np.float64)
    return df


def filter_st(df, start_date=19900101, end_date=None):
    """
    过滤 ST 交易数据
    :param df:
    :param start_date:
    :param end_date:
    :return:
    """
    st_dir = fileUtils.st_dir
    file_names = os.listdir(st_dir)
    # names = ['TRADE_DT',  # '交易日期'
    #          'S_DQ_OPEN',  # '开盘价(元)'
    #          'S_DQ_HIGH',  # '最高价(元)'
    #          'S_DQ_LOW',  # '最低价(元)'
    #          'S_DQ_CLOSE',  # '收盘价(元)'
    #          'S_DQ_VOLUME',  # '成交量(手)'
    #          'S_DQ_AMOUNT',  # '成交金额(千元)'
    #          ]
    for name in file_names:
        windcode = float(os.path.splitext(name)[0])
        df = df.loc[~(df['S_INFO_WINDCODE'] == windcode)]
        # path = os.path.abspath(os.path.join(fileUtils.st_dir, name))
        # st_df = pd.read_csv(path, encoding='gbk', names=names, low_memory=False)
        # st_df = st_df.dropna()
        # st_df = st_df.astype(np.float64)
        # if start_date:
        #     st_df = st_df[st_df['TRADE_DT'].astype('int64') >= start_date]
        # if end_date:
        #     st_df = st_df[st_df['TRADE_DT'].astype('int64') <= end_date]

        # st_arr = st_df['TRADE_DT'].values
        # df = df.loc[~((df['S_INFO_WINDCODE'] == windcode) & (df['TRADE_DT'].isin(st_arr)))]
    return df


def before_clean(df, start_date=19900101, end_date=None):
    klc = read_klc()
    if start_date:
        klc = klc[klc['TRADE_DT'].astype('int64') >= start_date]
    if end_date:
        klc = klc[klc['TRADE_DT'].astype('int64') <= end_date]
    # 交易日取交集
    k_l_c_arr = klc['TRADE_DT'].values
    df = df[df['TRADE_DT'].astype('int64').isin(k_l_c_arr)]
    # ST取差集合
    df = filter_st(df, start_date, end_date)
    # 原始数据中剔除涨跌幅异常（>10.5或<-10.5）的数据
    df = df[(df['S_DQ_PCTCHANGE'] > -10.5) & (df['S_DQ_PCTCHANGE'] < 10.5)]
    df = df[(df['S_DQ_VOLUME'] > 0)]
    return df


def idicator(df):
    """
    创造特征
    :param df:
    :return:
    """
    idicator_names = ['S_INFO_WINDCODE',  # 'Wind代码'

                      'TRADE_DT',  # '交易日期'
                      'S_DQ_ADJOPEN',  # '复权开盘价(元)'
                      'S_DQ_ADJHIGH',  # '复权最高价(元)'
                      'S_DQ_ADJLOW',  # '复权最低价(元)'
                      'S_DQ_ADJCLOSE',  # '复权收盘价(元)'

                      'S_DQ_AVGPRICE',  # '均价(VWAP)'
                      'S_DQ_MA5',  # 均线
                      'S_DQ_MA10',
                      'S_DQ_MA20',
                      'S_DQ_MA30',

                      'S_DQ_MA60',
                      'S_DQ_FREETURNOVER',  # '换手率(基准.自由流通股本)'
                      'DQ_CHANGE',  # 涨跌幅
                      'DQ_2_CHANGE',  # '两天涨跌幅'
                      # 'DQ_MARKET_CHANGE', # 市场涨跌幅
                      'S_DQ_AMOUNT',  # 成交金额(千元)

                      'S_DQ_3_AMOUNT',
                      'S_DQ_VOLUME',  # 成交量
                      'S_DQ_5_VOLUME',
                      'S_DQ_RSV',
                      'S_DQ_RATIO',  # 量比

                      'S_DQ_14_RSI',  # 强弱指标
                      'S_DQ_14_RSY',  # 心理线
                      'S_DQ_NLSR',  # 多空比率净额
                      'S_K_ENTITY',
                      'S_K_UP_SHADOW',

                      'S_K_DOWN_SHADOW',
                      'S_DQ_PVT',   # 价量趋势(PVT)指标
                      'S_DQ_60_LOW',
                      'S_DQ_60_HIGH',
                      ]
    num = 0
    data = []
    for name, group in df.groupby('S_INFO_WINDCODE'):
        num = num + 1
        print(name, '-------------', num)
        s_dq_60_low = group['S_DQ_ADJLOW'].rolling(window=60).min()
        s_dq_60_high = group['S_DQ_ADJHIGH'].rolling(window=60).max()
        s_dq_avgprice = group['S_DQ_AVGPRICE'] * group['S_DQ_ADJFACTOR']
        s_dq_5_volume = group['S_DQ_VOLUME'].rolling(window=5).mean()
        s_dq_ratio = group['S_DQ_VOLUME'] / s_dq_5_volume

        dq_change = (group['S_DQ_ADJCLOSE'] - group['S_DQ_ADJOPEN']) / group['S_DQ_ADJOPEN']
        dq_2_change = (group['S_DQ_ADJCLOSE'] - group['S_DQ_ADJOPEN'].shift(1)) / group['S_DQ_ADJOPEN'].shift(2)
        s_dq_3_amount = group['S_DQ_AMOUNT'].rolling(window=3).mean()

        s_dq_9_min_close = group['S_DQ_ADJCLOSE'].rolling(window=9).min()
        s_dq_9_max_close = group['S_DQ_ADJCLOSE'].rolling(window=9).max()
        s_dq_rsv = (group['S_DQ_ADJCLOSE'] - s_dq_9_min_close) / (s_dq_9_max_close - s_dq_9_min_close)
        s_dq_14_rsi = dq_change.rolling(window=14).apply(lambda x: x[x > 0].sum()) / dq_change.rolling(window=14).sum()
        # PSY=N日内上涨天数/N*100
        s_dq_14_rsy = dq_change.rolling(window=14).apply(lambda x: x[x > 0].shape[0])/14
        # 均线
        s_dq_ma5 = group['S_DQ_ADJCLOSE'].rolling(window=5).mean()
        s_dq_ma10 = group['S_DQ_ADJCLOSE'].rolling(window=10).mean()
        s_dq_ma20 = group['S_DQ_ADJCLOSE'].rolling(window=20).mean()
        s_dq_ma30 = group['S_DQ_ADJCLOSE'].rolling(window=30).mean()
        s_dq_ma60 = group['S_DQ_ADJCLOSE'].rolling(window=60).mean()

        s_k_entity = group['S_DQ_ADJCLOSE'] - group['S_DQ_ADJOPEN']
        s_k_up_shadow = group['S_DQ_ADJHIGH'] - group[['S_DQ_ADJCLOSE', 'S_DQ_ADJOPEN']].apply(lambda row: row['S_DQ_ADJCLOSE'] if row['S_DQ_ADJCLOSE'] > row['S_DQ_ADJOPEN'] else row['S_DQ_ADJOPEN'], axis=1)
        s_k_down_shadow = group[['S_DQ_ADJCLOSE', 'S_DQ_ADJOPEN']].apply(lambda row: row['S_DQ_ADJCLOSE'] if row['S_DQ_ADJCLOSE'] < row['S_DQ_ADJOPEN'] else row['S_DQ_ADJOPEN'], axis=1) - group['S_DQ_ADJLOW']
        s_dq_pvt = (group['S_DQ_ADJCLOSE'] - group['S_DQ_ADJCLOSE'].shift(1))/group['S_DQ_ADJCLOSE'] * group['S_DQ_VOLUME']


        # ------------------------------------------------------------------------
        # # 乖离率 N日BIAS=（当日收盘价—N日移动平均价）÷N日移动平均价×100
        # s_dq_5_bias = (group['S_DQ_ADJCLOSE'] - s_dq_ma5)/s_dq_ma5
        # #  威廉指标 n日WMS=[(Hn—Ct)/(Hn—Ln)] ×100 Cn——当天的收盘价 Hn和Ln——最近N日内（包括当天）出现的最高价和最低价。
        # s_dq_30_wr = (group['S_DQ_ADJHIGH'].rolling(window=3).max() - group['S_DQ_ADJCLOSE'])/(group['S_DQ_ADJHIGH'].rolling(window=3).max() - group['S_DQ_ADJLOW'].rolling(window=3).min())
        # # TRIX 三重指数平滑平均线
        # s_dq_ma20_trix = s_dq_trix.rolling(window=20).mean()
        # s_dq_trix = s_dq_12_3_ema - s_dq_12_3_ema.shift(1)/ s_dq_12_3_ema.shift(1)
        # s_dq_12_3_ema = s_dq_12_2_ema.ewm(span=12).mean()
        # s_dq_12_2_ema = s_dq_12_ema.ewm(span=12).mean()
        # s_dq_12_ema = group['S_DQ_ADJCLOSE'].ewm(span=12).mean()

        #  多空比率净额= [（收盘价－最低价）－（最高价-收盘价）] ÷（ 最高价－最低价）×成交量
        s_dq_nlsr = ((group['S_DQ_ADJCLOSE'] - group['S_DQ_ADJLOW']) - (group['S_DQ_ADJHIGH'] - group['S_DQ_ADJCLOSE']))/ (group['S_DQ_ADJHIGH'] - group['S_DQ_ADJLOW']) * group['S_DQ_VOLUME']
        cell = group.copy()
        cell = cell.drop(['S_DQ_ADJFACTOR'], axis=1)
        cell = cell.drop(['S_DQ_PCTCHANGE'], axis=1)
        cell['S_DQ_AVGPRICE'] = s_dq_avgprice
        cell.insert(7, 'S_DQ_MA5', s_dq_ma5)
        cell.insert(8, 'S_DQ_MA10', s_dq_ma10)
        cell.insert(9, 'S_DQ_MA20', s_dq_ma20)
        cell.insert(10, 'S_DQ_MA30', s_dq_ma30)
        cell.insert(11, 'S_DQ_MA60', s_dq_ma60)
        cell.insert(13, 'DQ_CHANGE', dq_change)
        cell.insert(14, 'DQ_2_CHANGE', dq_2_change)
        cell.insert(16, 'S_DQ_3_AMOUNT', s_dq_3_amount)
        cell.insert(18, 'S_DQ_5_VOLUME', s_dq_5_volume)
        cell.insert(19, 'S_DQ_RSV', s_dq_rsv)
        cell.insert(20, 'S_DQ_RATIO', s_dq_ratio)
        cell.insert(21, 'S_DQ_14_RSI', s_dq_14_rsi)
        cell.insert(22, 'S_DQ_14_RSY', s_dq_14_rsy)
        cell.insert(23, 'S_DQ_NLSR', s_dq_nlsr)
        cell.insert(24, 'S_K_ENTITY', s_k_entity)
        cell.insert(25, 'S_K_UP_SHADOW', s_k_up_shadow)
        cell.insert(26, 'S_K_DOWN_SHADOW', s_k_down_shadow)
        cell.insert(27, 'S_DQ_PVT', s_dq_pvt)
        cell.insert(28, 'S_DQ_60_LOW', s_dq_60_low)
        cell.insert(29, 'S_DQ_60_HIGH', s_dq_60_high)
        len_d = 50 if cell.shape[0] > 50 else cell.shape[0]
        cell.drop(cell.index[range(len_d)], inplace=True)
        data.extend(cell.values)
    data = pd.DataFrame(np.asarray(data), columns=idicator_names)
    df_data = []

    for date, group in data.groupby('TRADE_DT'):
        dq_market_change = group['DQ_CHANGE'].mean()
        cell = group.copy()
        cell.insert(15, 'DQ_MARKET_CHANGE', dq_market_change)
        df_data.extend(cell.values)
    idicator_names.insert(15, 'DQ_MARKET_CHANGE')
    df_data = pd.DataFrame(np.asarray(df_data), columns=idicator_names)
    df_data = df_data.dropna(axis=0, how="any")
    return df_data


def save(start_date=19900101, end_date=None):
    """
    保存原始数据
    :param start_date:
    :param end_date:
    :return: original_datas
    """
    df = read(start_date, end_date)
    print(df.shape)
    # before
    print('df=========', df.shape)
    df = before_clean(df, start_date, end_date)
    print('before clean=========', df.shape)

    # 创造特征
    df = idicator(df)
    # 原始数据中剔除涨跌幅异常（>10.5或<-10.5）的数据
    df = df[(df['DQ_CHANGE'] > -0.105) & (df['DQ_CHANGE'] < 0.105)]
    df = df[(df['DQ_2_CHANGE'] > -0.199) & (df['DQ_2_CHANGE'] < 0.221)]

    print('end=========', df.shape)
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'], axis=0, ascending=True)
    df['S_INFO_WINDCODE'] = df['S_INFO_WINDCODE']
    df['TRADE_DT'] = df['TRADE_DT']
    print(df.shape)
    print(df.head(10))
    original_datas = df.values
    data_dir = fileUtils.data_dir
    data_dir = os.path.abspath(os.path.join(data_dir, "original_datas.npz"))
    np.savez(data_dir, original_datas=original_datas)
    print('finished ...')
    print(original_datas)
    return original_datas


def load_original_datas():
    """
    加载原始数据
    :return: original_datas
    """
    print('load original_datas start')
    data_dir = fileUtils.data_dir
    data_dir = os.path.abspath(os.path.join(data_dir, "original_datas.npz"))
    original_datas = np.load(data_dir)

    data = original_datas['original_datas']
    print('original_datas load finish')
    return data

