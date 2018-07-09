#!/usr/bin/env python
# -*- coding: utf-8 -*-


# @Time    : 2018/6/5 15:05
# @Author  : Allen
# @File    : fileUtils.py

import os

parent_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(parent_dir, "data_file"))
temp_dir = os.path.abspath(os.path.join(parent_dir, "temp"))
csv_dir = os.path.abspath(os.path.join(data_dir, "csv"))
test_datas_dir = os.path.abspath(os.path.join(data_dir, "test_datas"))
train_datas_dir = os.path.abspath(os.path.join(data_dir, "train_datas"))
st_dir = os.path.abspath(os.path.join(csv_dir, "ST"))
batch_data_dir = os.path.abspath(os.path.join(data_dir, "batch"))