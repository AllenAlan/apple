#!/usr/bin/env python
# -*- coding: utf-8 -*-


# @Time    : 2018/6/12 9:22
# @Author  : Allen
# @File    : model_train.py
import model_build
import train_data_generate
import fileUtils
import os
import verify
import datetime
import original_data_generate


def single(idicator_name=None, model_temp=None):
    train_x, train_y = train_data_generate.load_train_datas('train_data_%s_.npz' % idicator_name)
    input_shape = (train_x.shape[1], train_x.shape[2] - 2)
    models = model_build.build_models(input_shape)
    model_name = model_build.train(model_temp, models[model_temp], train_x, train_y, idicator_name=idicator_name)
    m_name_list = model_name.split('_')
    test_x, test_y = train_data_generate.load_test_datas('test_data_%s_.npz' % idicator_name)
    model_path = os.path.abspath(os.path.join(fileUtils.temp_dir, model_name))
    pred_y = model_build.predict(model_path, test_x)
    verify.net_value(pred_y, test_y)
    print('----------------------------------------------')
    nv = verify.itto_ryu_net_value(pred_y, test_y)
    if float(m_name_list[3]) < nv:
        m_name_list[3] = str(nv)
        new_model_name = '_'.join(m_name_list)
        new_path = os.path.abspath(os.path.join(fileUtils.temp_dir, new_model_name))
        os.rename(model_path, new_path)


def batch():
    start_date = datetime.datetime.now()
    end_date = start_date + datetime.timedelta(days=2)
    while datetime.datetime.now() < end_date:
        file_names = sorted(os.listdir(fileUtils.train_datas_dir))
        for name in file_names:
            print("current train_datas is ", name)
            name_list = name.split('_')
            idicator_name = name_list[2]
            train_x, train_y = train_data_generate.load_train_datas(name)
            input_shape = (train_x.shape[1], train_x.shape[2] - 2)
            models = model_build.build_models(input_shape)
            for (temp_name, model) in models.items():
                model_name = model_build.train(temp_name, model, train_x, train_y, idicator_name=idicator_name)
                m_name_list = model_name.split('_')
                test_x, test_y = train_data_generate.load_test_datas('test_data_%s_.npz' % idicator_name)
                model_path = os.path.abspath(os.path.join(fileUtils.temp_dir, model_name))
                pred_y = model_build.predict(model_path, test_x)
                verify.net_value(pred_y, test_y)
                print('%s ----------------------------------------------' % temp_name)
                nv = verify.itto_ryu_net_value(pred_y, test_y)
                if float(m_name_list[3]) < nv:
                    m_name_list[3] = str(nv)
                    new_model_name = '_'.join(m_name_list)
                    new_path = os.path.abspath(os.path.join(fileUtils.temp_dir, new_model_name))
                    os.rename(model_path, new_path)


original_data_generate.save(start_date=20130101, end_date=20180103)
idicator_name = train_data_generate.train_prepare(20130101, 20180103, 20161228)
train_data_generate.save(idicator_name)
# single('12-20-21-22-23-24-25~27-2~8-13~16-18~20', 'model_2cnnI2cnnC2lstm')
batch()

