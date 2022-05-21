# ==============================================================================

"""Routine for decoding the MSR Action3D text file format."""
import tensorflow as tf
import numpy as np
import os
import random
import csv
import cv2

# 多视角融合
def read_test_by_sequence(DATA_PATH, config=None):                          
    file_list = os.listdir(os.path.join(os.getcwd(), DATA_PATH))
    file_list.sort()     

    sklt_input = []
    sklt_label = []

    s = []   
    l = []

    for fileNo in range(len(file_list)):
        s_mean = 0  
        l_mean = 0
        f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
        csv_reader = csv.reader(f, delimiter=',')
        temp_sklt = []
        cut_temp_sklt = []
        for index,row in enumerate(csv_reader):  
            if index == 0:
                continue
            temp_sklt.append(row)
            
        for x in range(len(temp_sklt)):        
            cut_temp_sklt.append(temp_sklt[x][0:28])

            list_init = temp_sklt[x][0:28]    # 读进来是字符串，要转成整型
            list_init = list(map(int, list_init))  
            list_del0 = []
            # 数据除0
            for i in range(14):
                if list_init[2*i] == 0 and list_init[2*i + 1] == 0:
                    continue
                else:
                    list_del0.append(list_init[2*i])
                    list_del0.append(list_init[2*i + 1])
            s_rect = 0
            if len(list_del0) != 0: 
                list_np = np.array(list_del0)  
                rect = cv2.minAreaRect(list_np.reshape(-1,2))
                s_rect = rect[1][0] * rect[1][1]

            s_mean = s_rect + s_mean     #所有帧的平均外接矩形大小          

            # 长度
            l0 = 0
            if list_init[10] != 0 and list_init[4] != 0:
                l0 = abs(list_init[10]-list_init[4]) / s_rect  #横坐标,表征相对面积来说的肩膀宽度
            l_mean = l0 + l_mean                    
             
        s_mean = s_mean/len(temp_sklt)
        l_mean = l_mean/len(temp_sklt)

        s.append([s_mean])   # 二维列表，每行一个元素，每个元素代表一个序列平均面积,在最后转成ndarray并返回  
        l.append([l_mean])

        sklt_input.append(cut_temp_sklt)

        sklt_label.append(int(file_list[fileNo][1:2]))
        f.close()
    else:
        pass

    return sklt_input, sklt_label, np.array(s), np.array(l)

def Sim_iterator(raw_data,cos_data, label, batch_size, input_size, num_steps, is_training):

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
