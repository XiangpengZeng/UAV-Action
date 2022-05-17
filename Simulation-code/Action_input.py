# Copyright 2016 Inwoong Lee All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the MSR Action3D text file format."""
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
    sklt_label = []     #所有的train或test文件分别读取后都存储在list中

    for fileNo in range(len(file_list)):

        # print os.path.join(os.getcwd(), 'sklt_data', file_list[fileNo])
        f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
        csv_reader = csv.reader(f, delimiter=',')
        temp_sklt = []
        for index,row in enumerate(csv_reader):   #zeng
            row = list(map(float, row))           # 转整数，因为可能出现小数
            temp_sklt.append(row)

        sklt_input.append(temp_sklt)
        sklt_label.append(int(file_list[fileNo][17:20]))   #zeng
        f.close()

    return sklt_input, sklt_label

def read_msra(DATA_PATH, VIEW_NUMBER, config=None):
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), DATA_PATH)):
        file_list = files

    sklt_input = []
    sklt_label = []     #所有的train或test文件分别读取后都存储在list中

    for fileNo in range(len(file_list)):
        for camNo in VIEW_NUMBER:

            if int(file_list[fileNo][5:7]) == camNo:          #different person
                # print os.path.join(os.getcwd(), 'sklt_data', file_list[fileNo])
                f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
                csv_reader = csv.reader(f, delimiter=',')
                temp_sklt = []
                for index,row in enumerate(csv_reader):   #zeng
                    row = list(map(float, row)) 
                    temp_sklt.append(row)

                sklt_input.append(temp_sklt)
                sklt_label.append(int(file_list[fileNo][1:3]))   #zeng
                f.close()
            else:
                pass

    return sklt_input, sklt_label

def read(DATA_PATH, VIEW_NUMBER, config=None):                             #for segment dataset by code
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), DATA_PATH)):
        file_list = files

    sklt_input = []
    sklt_label = []     #所有的train或test文件分别读取后都存储在list中

    for fileNo in range(len(file_list)):
        for camNo in VIEW_NUMBER:

            # print file_list[fileNo][1:4]
            #if int(file_list[fileNo][13:15]) == camNo:        #zeng  for e00-e05
            #if int(file_list[fileNo][9:11]) == camNo:         #sequence
            if int(file_list[fileNo][4:5]) == camNo:          #different person
                # print os.path.join(os.getcwd(), 'sklt_data', file_list[fileNo])
                f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
                csv_reader = csv.reader(f, delimiter=',')
                temp_sklt = []
                cut_temp_sklt = []
                for index,row in enumerate(csv_reader):   #zeng
                    if index == 0:
                        continue
                    temp_sklt.append(row)
                    
                    #temp_sklt = temp_sklt[0:28]     #zeng!!!
                    #print("len:",len(temp_sklt[0]))
                for x in range(len(temp_sklt)):          #zeng!!!
                    #print("len:",len(temp_sklt[x]))
                    cut_temp_sklt.append(temp_sklt[x][0:28])
                    #print("len:",len(cut_temp_sklt[x]))

                #print("sequence_length:",len(temp_sklt))

                sklt_input.append(cut_temp_sklt)
                # sklt_label.append(int(file_list[fileNo][1:3]))
                sklt_label.append(int(file_list[fileNo][1:2]))   #zeng
                f.close()
            else:
                pass

    # if CAMERA_OR_SUBJECT == 1:
    #     # CAMERA
    #     for fileNo in range(len(file_list)):
    #         for camNo in cv_set:
    #             # print file_list[fileNo][1:4]
    #             if int(file_list[fileNo][5:8]) == camNo:
    #                 # print int(file_list[fileNo][5:7])
    #                 # print os.path.join(os.getcwd(), 'sklt_data', file_list[fileNo])
    #                 f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
    #                 csv_reader = csv.reader(f, delimiter=',')
    #                 temp_sklt = []
    #                 for row in csv_reader:
    #                     temp_sklt.append(row)
    #                 sklt_input.append(temp_sklt)
    #                 sklt_label.append(int(file_list[fileNo][17:20]))
    #                 f.close()
    #             else:
    #                 pass
    # else:
    #     # SUBJECT
    #     for fileNo in range(len(file_list)):
    #         for subjNo in cv_set:
    #             # print file_list[fileNo][1:4]
    #             if int(file_PATHos.path.join(os.getcwd(), 'sklt_data', file_list[fileNo])
    #                 f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
    #                 csv_reader = csv.reader(f, delimiter=',')
    #                 temp_sklt = []
    #                 for row in csv_reader:
    #                     temp_sklt.append(row)
    #                 sklt_input.append(temp_sklt)
    #                 sklt_label.append(int(file_list[fileNo][17:20]))
    #                 f.close()
    #             else:
    #                 pass

    return sklt_input, sklt_label

def read_single(DATA_PATH, file_name, config=None):                             #for single file
    # for root, dirs, files in os.walk(os.path.join(os.getcwd(), DATA_PATH)):
    #     file_list = files

    sklt_input = []
    sklt_label = []

    # for fileNo in range(len(file_list)):
    #     for camNo in VIEW_NUMBER:

            # print file_list[fileNo][1:4]
            #if int(file_list[fileNo][13:15]) == camNo:        #zeng  for e00-e05
            #if int(file_list[fileNo][9:11]) == camNo:         #sequence
            # if int(file_list[fileNo][9:11]) == camNo:          #different person
                #print(file_list[fileNo][9:11])
                # print int(file_list[fileNo][5:7])
                # print os.path.join(os.getcwd(), 'sklt_data', file_list[fileNo])
    f = open(DATA_PATH)
    csv_reader = csv.reader(f, delimiter=',')
    temp_sklt = []
    cut_temp_sklt = []
    for row in csv_reader:
        temp_sklt.append(row)
        
        #temp_sklt = temp_sklt[0:28]     #zeng!!!
        #print("len:",len(temp_sklt[0]))
    for x in range(len(temp_sklt)):          #zeng!!!
        #print("len:",len(temp_sklt[x]))
        cut_temp_sklt.append(temp_sklt[x][0:28])
        #print("len:",len(cut_temp_sklt[x]))

    #print("sequence_length:",len(temp_sklt))

    sklt_input.append(cut_temp_sklt)
    sklt_label.append(int(file_name[1:3]))
    f.close()
            # else:
            #     pass

    # if CAMERA_OR_SUBJECT == 1:
    #     # CAMERA
    #     for fileNo in range(len(file_list)):
    #         for camNo in cv_set:
    #             # print file_list[fileNo][1:4]
    #             if int(file_list[fileNo][5:8]) == camNo:
    #                 # print int(file_list[fileNo][5:7])
    #                 # print os.path.join(os.getcwd(), 'sklt_data', file_list[fileNo])
    #                 f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
    #                 csv_reader = csv.reader(f, delimiter=',')
    #                 temp_sklt = []
    #                 for row in csv_reader:
    #                     temp_sklt.append(row)
    #                 sklt_input.append(temp_sklt)
    #                 sklt_label.append(int(file_list[fileNo][17:20]))
    #                 f.close()
    #             else:
    #                 pass
    # else:
    #     # SUBJECT
    #     for fileNo in range(len(file_list)):
    #         for subjNo in cv_set:
    #             # print file_list[fileNo][1:4]
    #             if int(file_PATHos.path.join(os.getcwd(), 'sklt_data', file_list[fileNo])
    #                 f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
    #                 csv_reader = csv.reader(f, delimiter=',')
    #                 temp_sklt = []
    #                 for row in csv_reader:
    #                     temp_sklt.append(row)
    #                 sklt_input.append(temp_sklt)
    #                 sklt_label.append(int(file_list[fileNo][17:20]))
    #                 f.close()
    #             else:
    #                 pass

    return sklt_input, sklt_label

def read_test(DATA_PATH, config=None):                                      #for all files in a single folder
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), DATA_PATH)):
        file_list = files

    sklt_input = []
    sklt_label = []

    for fileNo in range(len(file_list)):
        f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
        csv_reader = csv.reader(f, delimiter=',')
        temp_sklt = []
        cut_temp_sklt = []
        for row in csv_reader:
            temp_sklt.append(row)
            
            #temp_sklt = temp_sklt[0:28]     #zeng!!!
            #print("len:",len(temp_sklt[0]))
        for x in range(len(temp_sklt)):          #zeng!!!
            #print("len:",len(temp_sklt[x]))
            cut_temp_sklt.append(temp_sklt[x][0:28])
            #print("len:",len(cut_temp_sklt[x]))

        #print("sequence_length:",len(temp_sklt))

        sklt_input.append(cut_temp_sklt)
        #print('--------',file_list[fileNo][1:3])
        sklt_label.append(int(file_list[fileNo][1:3]))
        f.close()
    else:
        pass

    # if CAMERA_OR_SUBJECT == 1:
    #     # CAMERA
    #     for fileNo in range(len(file_list)):
    #         for camNo in cv_set:
    #             # print file_list[fileNo][1:4]
    #             if int(file_list[fileNo][5:8]) == camNo:
    #                 # print int(file_list[fileNo][5:7])
    #                 # print os.path.join(os.getcwd(), 'sklt_data', file_list[fileNo])
    #                 f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
    #                 csv_reader = csv.reader(f, delimiter=',')
    #                 temp_sklt = []
    #                 for row in csv_reader:
    #                     temp_sklt.append(row)
    #                 sklt_input.append(temp_sklt)
    #                 sklt_label.append(int(file_list[fileNo][17:20]))
    #                 f.close()
    #             else:
    #                 pass
    # else:
    #     # SUBJECT
    #     for fileNo in range(len(file_list)):
    #         for subjNo in cv_set:
    #             # print file_list[fileNo][1:4]
    #             if int(file_list[fileNo][9:12]) == subjNo:
    #                 # print int(file_list[fileNo][9:12])
    #                 # print os.path.join(os.getcwd(), 'sklt_data', file_list[fileNo])
    #                 f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
    #                 csv_reader = csv.reader(f, delimiter=',')
    #                 temp_sklt = []
    #                 for row in csv_reader:
    #                     temp_sklt.append(row)
    #                 sklt_input.append(temp_sklt)
    #                 sklt_label.append(int(file_list[fileNo][17:20]))
    #                 f.close()
    #             else:
    #                 pass

    return sklt_input, sklt_label

def read_test_by_sequence(DATA_PATH, config=None):                                      #for single folder
    # for 3 view_fusion.py
    # for root, dirs, files in os.walk(os.path.join(os.getcwd(), DATA_PATH)):
    #     file_list = files

    file_list = os.listdir(os.path.join(os.getcwd(), DATA_PATH))
    file_list.sort()                   #核心！！    ref:https://www.cnblogs.com/chester-cs/p/12252358.html

    sklt_input = []
    sklt_label = []

    s = []   #zeng

    for fileNo in range(len(file_list)):
        s_mean = 0   #zeng
    #for fileNo in range(len(file_list)):
        f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
        csv_reader = csv.reader(f, delimiter=',')
        temp_sklt = []
        cut_temp_sklt = []
        for row in csv_reader:
            temp_sklt.append(row)
            
            #temp_sklt = temp_sklt[0:28]     #zeng!!!
            #print("len:",len(temp_sklt[0]))
        for x in range(len(temp_sklt)):          #zeng!!!
            #print("len:",len(temp_sklt[x]))
            cut_temp_sklt.append(temp_sklt[x][0:28])

            list_init = temp_sklt[x][0:28]    #读进来是字符串，要转成整型
            list_init = list(map(int, list_init))
            list_np = np.array(list_init)
            #print("--------:",list_init)
            rect = cv2.minAreaRect(list_np.reshape(14,2))
            s_rect = rect[1][0] * rect[1][1]
            s_mean = s_rect + s_mean
            #print("len:",len(cut_temp_sklt[x]))   #28
        s_mean = s_mean/len(temp_sklt)
        s.append([s_mean])   # 二维列表，每行一个元素，每个元素代表一个序列平均面积,在最后转成ndarray并返回  
        #print(s)

        #print("sequence_length:",len(temp_sklt))

        sklt_input.append(cut_temp_sklt)
        #print('--------',file_list[fileNo][1:3])
        sklt_label.append(int(file_list[fileNo][1:3]))
        f.close()
    else:
        pass

    # if CAMERA_OR_SUBJECT == 1:
    #     # CAMERA
    #     for fileNo in range(len(file_list)):
    #         for camNo in cv_set:
    #             # print file_list[fileNo][1:4]
    #             if int(file_list[fileNo][5:8]) == camNo:
    #                 # print int(file_list[fileNo][5:7])
    #                 # print os.path.join(os.getcwd(), 'sklt_data', file_list[fileNo])
    #                 f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
    #                 csv_reader = csv.reader(f, delimiter=',')
    #                 temp_sklt = []
    #                 for row in csv_reader:
    #                     temp_sklt.append(row)
    #                 sklt_input.append(temp_sklt)
    #                 sklt_label.append(int(file_list[fileNo][17:20]))
    #                 f.close()
    #             else:
    #                 pass
    # else:
    #     # SUBJECT
    #     for fileNo in range(len(file_list)):
    #         for subjNo in cv_set:
    #             # print file_list[fileNo][1:4]
    #             if int(file_list[fileNo][9:12]) == subjNo:
    #                 # print int(file_list[fileNo][9:12])
    #                 # print os.path.join(os.getcwd(), 'sklt_data', file_list[fileNo])
    #                 f = open(os.path.join(os.getcwd(), DATA_PATH, file_list[fileNo]))
    #                 csv_reader = csv.reader(f, delimiter=',')
    #                 temp_sklt = []
    #                 for row in csv_reader:
    #                     temp_sklt.append(row)
    #                 sklt_input.append(temp_sklt)
    #                 sklt_label.append(int(file_list[fileNo][17:20]))
    #                 f.close()
    #             else:
    #                 pass

    return sklt_input, sklt_label, np.array(s)

# def read(data_path=None, config=None):
#
#     file_used_path = os.path.join(data_path, "file_used.txt")
#     f1 = open(file_used_path, 'r')
#     file_used = f1.readlines()
#     f1.close()
#
#     Num_batch = len(file_used)
#     Num_Steps = config.num_steps
#
#     MSR_data = np.zeros((Num_batch, Num_Steps*60))
#     MSR_label = np.zeros((Num_batch, 1))
#     MSR_subject = np.zeros((Num_batch, 1))
#     MSR_instance = np.zeros((Num_batch, 1))
#     MSR_time_stamp = np.zeros((Num_batch, 1))
#
#     for i in range(Num_batch):
#         raw_string = []
#         raw_float = []
#
#         raw_path = os.path.join(data_path, file_used[i][0:11] + "_skeleton3D.txt")
#         f2 = open(raw_path, 'r')
#         raw_line = f2.readlines()
#         for line_string in raw_line:
#             raw_string.append(line_string.split(' '))
#         f2.close()
#
#         for j in range(len(raw_string)):
#             del raw_string[j][3]
#
#         for n in range(len(raw_string)):
#             raw_float.append([float(x) for x in raw_string[n]])
#
#         k = 0
#         for n in range(len(raw_string)):
#             for m in range(3):
#                 MSR_data[i, Num_Steps*60-len(raw_string)*3+k] = raw_float[n][m]
#                 k += 1
#
#         MSR_label[i, 0] = float(file_used[i][1:3])
#         MSR_subject[i, 0] = float(file_used[i][5:7])
#         MSR_instance[i, 0] = float(file_used[i][9:11])
#         MSR_time_stamp[i, 0] = float(len(raw_float)/60)
#
#     return MSR_data, MSR_label, MSR_subject, MSR_instance, MSR_time_stamp


# def MSR_iterator(raw_data, label, batch_size, input_size, num_steps):
#     data_len = len(raw_data)
#     batch_len = data_len // batch_size
#
#     batch_index = np.arange(batch_size)

#     epoch_size = (batch_len - 1) // num_steps
#
#     for i in range(batch_len):
#         for randNo in range(batch_size):
#             batch_index[randNo] = random.randint(0, data_len-1)
#         x = raw_data[batch_index, :, :]
#         y = label[batch_index, :]
#         yield (x, y)

# def MSR_iterator(raw_data, label, cos_data, batch_size, input_size, num_steps, is_training):
#     # raw_data = np.array(raw_data, dtype=np.float32)
#     #
#     data_len = len(raw_data)
#     batch_len = data_len // batch_size

#     batch_index = np.arange(batch_size)

#     # data = np.zeros([batch_size, batch_len], dtype=np.float32)
#     # for i in range(batch_size):
#     #   data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
#     #
#     epoch_size = (batch_len - 1) // num_steps
#     #
#     # if epoch_size == 0:
#     #   raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

#     for i in range(batch_len):
#         if is_training == True:
#             for randNo in range(batch_size):
#                 batch_index[randNo] = random.randint(0, data_len-1)
#             x = raw_data[batch_index, :, :]
#             y = label[batch_index, :]
#             z = cos_data[batch_index, :, :]    #zeng
#         else:
#             x = raw_data[i*batch_size:(i+1)*batch_size, :, :]
#             y = label[i*batch_size:(i+1)*batch_size, :]
#             z = cos_data[i*batch_size:(i+1)*batch_size, :, :]
#         yield (x, y, z)


def MSR_iterator(raw_data,cos_data, label, batch_size, input_size, num_steps, is_training):
    # raw_data = np.array(raw_data, dtype=np.float32)
    #
    data_len = len(raw_data)
    batch_len = data_len // batch_size

    batch_index = np.arange(batch_size)

    # data = np.zeros([batch_size, batch_len], dtype=np.float32)
    # for i in range(batch_size):
    #   data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    #
    epoch_size = (batch_len - 1) // num_steps
    #
    # if epoch_size == 0:
    #   raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(batch_len):
        if is_training == True:
            for randNo in range(batch_size):
                batch_index[randNo] = random.randint(0, data_len-1)
            x = raw_data[batch_index, :, :]
            y = label[batch_index, :]
            z = cos_data[batch_index, :, :]
        else:     # 测试的时候还是按照 顺序 索引的
            x = raw_data[i*batch_size:(i+1)*batch_size, :, :]
            y = label[i*batch_size:(i+1)*batch_size, :]
            z = cos_data[i*batch_size:(i+1)*batch_size, :, :]
        yield (x, y, z)

def MSR_iterator_lstm(raw_data, label, batch_size, input_size, num_steps, is_training):
    # raw_data = np.array(raw_data, dtype=np.float32)
    #
    data_len = len(raw_data)
    batch_len = data_len // batch_size

    batch_index = np.arange(batch_size)

    # data = np.zeros([batch_size, batch_len], dtype=np.float32)
    # for i in range(batch_size):
    #   data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    #
    epoch_size = (batch_len - 1) // num_steps
    #
    # if epoch_size == 0:
    #   raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(batch_len):
        if is_training == True:
            for randNo in range(batch_size):
                batch_index[randNo] = random.randint(0, data_len-1)
            x = raw_data[batch_index, :, :]
            y = label[batch_index, :]
        else:
            x = raw_data[i*batch_size:(i+1)*batch_size, :, :]
            y = label[i*batch_size:(i+1)*batch_size, :]
        yield (x, y)