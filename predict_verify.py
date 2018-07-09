#!/usr/bin/env python
# -*- coding: utf-8 -*-


# @Time    : 2018/7/4 16:37
# @Author  : Allen
# @File    : predict_verify.py.py
"""
1. 读取训练数据和测试数据，并加载，并用模型得到预测结果
2. 随机抽取某一天，生成当天的预测结果
3. 对比预测结果和生成的预测结果有何不同（可能会存在停牌的差异）
"""


