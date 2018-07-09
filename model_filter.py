#!/usr/bin/env python
# -*- coding: utf-8 -*-


# @Time    : 2018/7/4 16:05
# @Author  : Allen
# @File    : model_filter.py


"""
1. 设定训练要求的净值水平limit，如limit设定为1.8
2. 加载训练好的temp模型，用测试集数据进行测试，测试出相应的净值
3. 如果得到的净值大于limit，则将模型文件重命名为模型名-日期时间-净值.h5并保存至filtered_model目录
"""



import keras


def filter(limit=1.4):
    pass
