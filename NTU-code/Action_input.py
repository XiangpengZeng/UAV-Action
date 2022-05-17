# 数据读取与迭代
import tensorflow as tf
import numpy as np
import os
import random
import csv
import cv2

def read_ntu(DATA_PATH, VIEW_NUMBER, config=None):
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), DATA_PATH)):
        file_list = files

    sklt_input = []
    sklt_label = []    

    for fileNo in range(len(file_list)):

        f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
        csv_reader = csv.reader(f, delimiter=',')
        temp_sklt = []
        for index,row in enumerate(csv_reader):  
            row = list(map(float, row))           # 转整数，因为可能出现小数
            temp_sklt.append(row)

        sklt_input.append(temp_sklt)
        sklt_label.append(int(file_list[fileNo][17:20]))  
        f.close()

    return sklt_input, sklt_label


def NTU_iterator(raw_data,cos_data, label, batch_size, input_size, num_steps, is_training):

    data_len = len(raw_data)
    batch_len = data_len // batch_size

    batch_index = np.arange(batch_size)

    epoch_size = (batch_len - 1) // num_steps

    for i in range(batch_len):
        if is_training == True:
            for randNo in range(batch_size):
                batch_index[randNo] = random.randint(0, data_len-1)
            x = raw_data[batch_index, :, :]
            y = label[batch_index, :]
            z = cos_data[batch_index, :, :]
        else:   
            x = raw_data[i*batch_size:(i+1)*batch_size, :, :]
            y = label[i*batch_size:(i+1)*batch_size, :]
            z = cos_data[i*batch_size:(i+1)*batch_size, :, :]
        yield (x, y, z)

