# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
#
#
# # @Time    : 2018/5/9 22:16
# # @Author  : Allen
# # @File    : test.py
#
import numpy as np
import pandas as pd
import os
import fileUtils
import datetime
import time
import math

print(datetime.datetime.now().strftime("%Y%m%d%H%S"))

# df = pd.DataFrame([
#                    ['000001', 20160103, 3, 3, 0.111, 10, 6666.6, 7777.7, 8888.8, 9999.9],
#                    ['000001', 20160104, 4, 3, 0.111, 21, 6666.6, 7777.7, 8888.8, 9999.9],
#                    ['000001', 20160105, 2, 3, 0.111, 21, 6666.6, 7777.7, 8888.8, 9999.9],
#                    ['000001', 20160106, 1, 3, 0.111, 21, 6666.6, 7777.7, 8888.8, 9999.9],
#                    ['000001', 20160107, 7, 3, 0.111, 1, 6666.6, 7777.7, 8888.8, 9999.9]],
#                   # ['000002', 20160102, 2, 3, 0.111, 1, 6666.6, 7777.7, 8888.8, 9999.9],
#                   # ['000002', 20160103, 3, 3, 0.111, 1, 6666.6, 7777.7, 8888.8, 9999.9],
#                   # ['000002', 20160104, -4,3, 0.111, 1.0, 6666.6, 7777.7, 8888.8, 9999.9],
#                   # ['000002', 20160105, 5, 3, 0.111, 2.0, 6666.6, 7777.7, 8888.8, 9999.9],
#                   # ['000002', 20160106, 6, 3, 0.111, 3.0, 6666.6, 7777.7, 8888.8, 9999.9]],
#                   columns=list('qwertasdfg'))
#
# data = []
# for name, group in df.groupby('q'):
#     f = group['f'] * group['g']
#     a = group['r'] - group['r'].shift(-1)
#     b = group['e'].rolling(window=3).min()
#     u = group['e'].cumsum()
#     c = (group['e'] - group['a'].rolling(window=3).min()) / (group['a'].rolling(window=3).max() - group['a'].rolling(window=3).min()) * 100
#     ewma_df = group['e'].ewm(span=5).mean()
#     h = group[['e', 'r']].apply(lambda row: row['e'] if row['e'] > row['r'] else row['r'], axis=1)
#     print(b)
#     cell = group.copy()
#     cell['f'] = f
#     cell.insert(10, 'x', b)
#     cell.insert(11, 'z', a)
#     cell.insert(12, 'c', c)
#     cell.insert(12, 'u', u)
#     data.extend(cell.values)
# data = pd.DataFrame(np.asarray(data), columns=list('qwertasdfgzxcu'))
#
# data1 = []
# for name, group in data.groupby('w'):
#     y = group['e'].mean()
#     cell = group.copy()
#     cell.insert(5, 'y', y)
#     data1.extend(cell.values)
# arr = list('qwertasdfgzxcu')
# arr.insert(5, 'y')
# data1 = pd.DataFrame(np.asarray(data1), columns=arr)


