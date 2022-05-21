# coding=utf-8
import time

from datetime import datetime
from time import localtime, strftime

import tensorflow as tf
import numpy as np

import os
import csv

import Action_input

from PreProcesses import body_rotation_cos, body_rotation, feature_only_diff, one_hot_labeling

import matplotlib.pyplot as plt

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS

_now = datetime.now()
BatchDivider = 10    
myKeepProb = 0.4
myHiddenSize = 112    
win_size = [95, 85, 75, 55, 40, 25, 90]
stride = [15, 20, 25, 25, 20, 25, 20]
start_time = [1, 5, 10, 5, 3, 2, 1] 

def getDevice():
    return FLAGS.num_gpus

device_index = [ 0, 0, 0, 0, 0, 0 ]

gradient_device = ['/gpu:0','/gpu:0','/gpu:0','/gpu:0']

class Sim_runner(object):
    def __init__(self, is_training, config):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps  # Max_size
        self.class_size = class_size = config.class_size 
        size = config.hidden_size  

        counter = 0

        self.data_initializer = tf.placeholder(tf.float32, [batch_size, num_steps, feature_size], name="x-input")
        self.cos_data_initializer = tf.placeholder(tf.float32, [batch_size, num_steps, feature_size], name="z-input")

        self.input_data = tf.Variable(self.data_initializer, trainable=False, collections=[], name="x-top")
        self.input_cos_data = tf.Variable(self.cos_data_initializer, trainable=False, collections=[], name="z-top")

        if is_training:
            sw_0 = sw_1 = sw_2 = sw_3 = '/gpu:0'
        else:
            sw_0 = sw_1 = sw_2 = sw_3 = '/cpu:0'

        with tf.device(sw_0):
            with tf.name_scope('%s_%d' % ('mGPU', 0)) as scope:
                self.mL = mL = Long_Term_Top(is_training, config, self.input_data, size)
                logits_l, state_0 = mL.get_inference()
                self._logits_0, self._final_state = \
                    logits_0, final_state = logits_l, state_0

        with tf.device(sw_2):
            with tf.name_scope('%s_%d' % ('mGPU', 2)) as scope:
                self.mS = mS = Short_Term_Top(is_training, config, self.input_data, size)
                logits_s, state_4 = mS.get_inference()
                self._logits_2, self._state_4 = \
                    logits_2, state_4 = logits_s, state_4

        with tf.device(sw_3):
            with tf.name_scope('%s_%d' % ('mGPU', 3)) as scope:
                self.m_A = m_A = Angle_LSTM_Top(is_training, config, self.input_cos_data, size) 
                logits_angle, state_angle = m_A.get_inference()                             
                self._logits_angle, self._state_angle = \
                    logits_angle, state_angle = logits_angle, state_angle

        with tf.device(sw_3):
            print("Parallelized Model is on building!!")


            if not is_training:
                with tf.name_scope("Accuracy") as scope:
                    real_logits = (logits_0  + logits_2 + logits_angle) / 3                 
                    self.real_logits = real_logits     

                    self.pred_labels = tf.argmax(real_logits, 1)                          
            else:
                with tf.name_scope("Accuracy") as scope:
                    real_logits = (logits_l + logits_s + logits_angle) / 3
                    self.pred_labels = tf.argmax(real_logits, 1)


        if is_training:
            with tf.device(gradient_device[3]):
                with tf.name_scope("train") as scope:
                    with tf.name_scope("Merging_Gradient"):
                        print("L : ", len(self.grad_before_sum_L))            
                        print("S : ", len(self.grad_before_sum_S))
                        print("Angle : ", len(self.grad_before_sum_angle))
                        self.grad_after_sum = self.grad_before_sum_L + self.grad_before_sum_S  + self.grad_before_sum_angle   
                        print("L+S+Angle : ", len(self.grad_after_sum))
                        self._lr = tf.Variable(0.0, trainable=False)
                        self.grads, _ = tf.clip_by_global_norm(self.grad_after_sum, config.max_grad_norm)
                        self.optimizer = tf.train.GradientDescentOptimizer(self.lr)  #梯度下降法
                        self.tvars = tf.trainable_variables()
                        with tf.name_scope("Applying-Gradient"):
                            self._train_op = self.optimizer.apply_gradients(zip(self.grads,self.tvars))

        print("Calculating Graph is fully connected!!")

    def assign_lr(self, sess, lr_value):
        sess.run(tf.assign(self.lr, lr_value))

    def mp_init(self, sess):
        self.mL.init_all_var(sess)
        self.mS.init_all_var(sess)
        self.m_A.init_all_var(sess)   

    @property
    def initial_state_L(self):
        return self.mL.initial_state

    @property
    def initial_state_S(self):
        return self.mS.initial_state

    @property
    def initial_state_Angle(self):
        return self.m_A.initial_state     

    @property
    def logits_0(self):
        return self._logits_0

    @property
    def logits_2(self):
        return self._logits_2

    @property
    def logits_angle(self):
        return self._logits_angle

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def training(self):
        return self._training

    @property
    def ac_summ(self):
        return self._ac_summ

    @property
    def summary_op(self):
        return self._summary_op


class Long_Term_Top(object):
    def __init__(self, is_training, config, top_input_data, top_hidden_size):
        if is_training:
            sw_0 = sw_1 = sw_2 = sw_3 = sw_4 = '/gpu:0'
        else:
            sw_0 = sw_1 = sw_2 = sw_3 = sw_4 = '/cpu:0'

        with tf.device(sw_0):
            self._training = is_training
            self.batch_size = batch_size = config.batch_size
            self.feature_size = feature_size = config.feature_size
            self.num_steps = num_steps = config.num_steps 
            self.class_size = class_size = config.class_size
            size = top_hidden_size  

            self.input_data = top_input_data

            num_LSTMs_0 = len(range(start_time[0], num_steps - win_size[0] + start_time[0], stride[0]))
            num_LSTMs_1 = len(range(start_time[1], num_steps - win_size[1] + start_time[1], stride[1]))
            num_LSTMs_2 = len(range(start_time[2], num_steps - win_size[2] + start_time[2], stride[2]))

        with tf.variable_scope("Long_Term"):
            print("Long_Term_Top")

            with tf.device(sw_1):
                with tf.name_scope('%s_%d' % ('Long0_GPU', device_index[0])) as scope:
                    with tf.variable_scope('l0'):
                        self.mL0 = mL0 = Long_Term_0(is_training, config, self.input_data)
                        self._initial_state = mL0.initial_state
                        output_depthconcat_long_0 = mL0.get_depth_concat_output()
                        self.output_depthconcat_long_0 = output_depthconcat_long_0

            with tf.device(sw_2):
                with tf.name_scope('%s_%d' % ('Long1_GPU', device_index[1])) as scope:
                    with tf.variable_scope('l1'):
                        self.mL1 = mL1 = Long_Term_1(is_training, config, self.input_data)
                        self._initial_state = mL1.initial_state   
                        output_depthconcat_long_1 = mL1.get_depth_concat_output()
                        self.output_depthconcat_long_1 = output_depthconcat_long_1

            with tf.device(sw_3):
                with tf.name_scope('%s_%d' % ('Long2_GPU', device_index[2])) as scope:
                    with tf.variable_scope('l2'):
                        self.mL2 = mL2 = Long_Term_2(is_training, config, self.input_data)
                        self._initial_state = mL2.initial_state
                        output_depthconcat_long_2 = mL2.get_depth_concat_output()
                        self.output_depthconcat_long_2 = output_depthconcat_long_2

            with tf.device(sw_4):
                with tf.variable_scope("Concat_0"):
                    output_real_temp_0 = tf.concat([output_depthconcat_long_0, output_depthconcat_long_1], 1)
                    output_real_0 = tf.concat([output_real_temp_0, output_depthconcat_long_2], 1)

                with tf.variable_scope("Drop_0"):
                    if is_training and config.keep_prob < 1:
                        output_real_0 = tf.nn.dropout(output_real_0, config.keep_prob)

                    if not is_training and config.keep_prob < 1:
                        output_real_0 = output_real_0 * config.keep_prob

                with tf.variable_scope("Softmax_0"):
                    self.softmax_w_0 = softmax_w_0 = tf.get_variable("softmax_w_0",
                                                                     [(num_LSTMs_0 + num_LSTMs_1 + num_LSTMs_2) * size,
                                                                      class_size])
                    self.softmax_b_0 = softmax_b_0 = tf.get_variable("softmax_b_0", [class_size])
                    self.logits = logits = tf.nn.softmax(tf.matmul(output_real_0, softmax_w_0) + softmax_b_0)

                    ######

                    self._final_state = mL0.get_state()
                    self._lr = tf.Variable(0.0, trainable=False)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value, validate_shape=False))

    def get_inference(self):
        return self.logits, self.final_state

    def init_all_var(self, session):
        init = tf.global_variables_initializer()
        session.run(init)

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def training(self):
        return self._training


class Short_Term_Top(object):
    def __init__(self, is_training, config, top_input_data, top_hidden_size):
        if is_training:
            sw_0 = sw_1 = sw_2 = sw_3 = sw_4 = '/gpu:0'
        else:
            sw_0 = sw_1 = sw_2 = sw_3 = sw_4 = '/cpu:0'

        with tf.device(sw_0):
            self._training = is_training
            self.batch_size = batch_size = config.batch_size
            self.feature_size = feature_size = config.feature_size
            self.num_steps = num_steps = config.num_steps 
            self.class_size = class_size = config.class_size
            size = top_hidden_size

            self.input_data = top_input_data

            num_LSTMs_3 = len(range(start_time[3], num_steps - win_size[3] + start_time[3], stride[3]))
            num_LSTMs_4 = len(range(start_time[4], num_steps - win_size[4] + start_time[4], stride[4]))    
            num_LSTMs_5 = len(range(start_time[5], num_steps - win_size[5] + start_time[5], stride[5]))

        with tf.variable_scope("Short_Term"):
            print("Short_Term_Top")

            with tf.device(sw_1):
                with tf.name_scope('%s_%d' % ('Short3_GPU', device_index[3])) as scope:
                    with tf.variable_scope('s3'):
                        self.mS3 = mS3 = Short_Term_3(is_training, config, self.input_data)
                        self._initial_state = mS3.initial_state
                        output_depthconcat_medium_3 = mS3.get_depth_concat_output()
                        self.output_depthconcat_medium_3 = output_depthconcat_medium_3
            
            with tf.device(sw_2):
                with tf.name_scope('%s_%d' % ('Short4_GPU', device_index[4])) as scope:
                    with tf.variable_scope('s4'):
                        self.mS4 = mS4 = Short_Term_4(is_training, config, self.input_data)
                        self._initial_state = mS4.initial_state
                        output_depthconcat_medium_4 = mS4.get_depth_concat_output()
                        self.output_depthconcat_medium_4 = output_depthconcat_medium_4

            with tf.device(sw_3):
                with tf.name_scope('%s_%d' % ('Short5_GPU', device_index[5])) as scope:
                    with tf.variable_scope('s5'):
                        self.mS5 = mS5 = Short_Term_5(is_training, config, self.input_data)
                        self._initial_state = mS5.initial_state
                        output_depthconcat_short_5 = mS5.get_depth_concat_output()
                        self.output_depthconcat_short_5 = output_depthconcat_short_5

            with tf.device(sw_4):
                with tf.variable_scope("Concat_2"):
                    output_real_temp_2 = tf.concat([output_depthconcat_medium_3, output_depthconcat_medium_4], 1)
                    output_real_2 = tf.concat([output_real_temp_2, output_depthconcat_short_5], 1)

                with tf.variable_scope("Drop_2"):
                    if is_training and config.keep_prob < 1:
                        output_real_2 = tf.nn.dropout(output_real_2, config.keep_prob)

                    if not is_training and config.keep_prob < 1:
                        output_real_2 = output_real_2 * config.keep_prob

                with tf.variable_scope("Softmax_2"):
                    self.softmax_w_2 = softmax_w_2 = tf.get_variable("softmax_w", [(num_LSTMs_3 + num_LSTMs_4 + num_LSTMs_5) * size, class_size])
                    self.softmax_b_2 = softmax_b_2 = tf.get_variable("softmax_b", [class_size])
                    self.logits = logits = tf.nn.softmax(tf.matmul(output_real_2, softmax_w_2) + softmax_b_2)
                    ######
                    self._final_state = mS5.state_5
                    self._lr = tf.Variable(0.0, trainable=False)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value, validate_shape=False))

    def get_inference(self):
        return self.logits, self.final_state     

    def init_all_var(self, session):
        init = tf.global_variables_initializer()
        session.run(init)

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def training(self):
        return self._training

class Angle_LSTM_Top(object):
    def __init__(self, is_training, config, top_input_data, top_hidden_size):
        if is_training:
            sw_0 = sw_1 = sw_2 = '/gpu:0'
        else:
            sw_0 = sw_1 = sw_2 = '/cpu:0'

        with tf.device(sw_0):
            self._training = is_training
            self.batch_size = batch_size = config.batch_size
            self.feature_size = feature_size = config.feature_size
            self.num_steps = num_steps = config.num_steps
            self.class_size = class_size = config.class_size
            size = top_hidden_size 

            self.input_data = top_input_data

            num_LSTMs_6 = len(range(start_time[6], num_steps - win_size[6] + start_time[6], stride[6]))

        with tf.variable_scope("Angle_Sliding"):
            print("Angle_Sliding_Top")

            with tf.device(sw_1):
                with tf.name_scope('%s_%d' % ('Angle6_GPU', device_index[5])) as scope:
                    with tf.variable_scope('a6'):
                        self.mA6 = mA6 = Angle_6(is_training, config, self.input_data)
                        self._initial_state = mA6.initial_state
                        output_depthconcat_short_6 = mA6.get_depth_concat_output()
                        self.output_depthconcat_short_6 = output_depthconcat_short_6

            with tf.device(sw_2):
                with tf.variable_scope("Concat_3"):
                    output_real_2 = output_depthconcat_short_6

                with tf.variable_scope("Drop_3"):
                    if is_training and config.keep_prob < 1:
                        output_real_2 = tf.nn.dropout(output_real_2, config.keep_prob)

                    if not is_training and config.keep_prob < 1:
                        output_real_2 = output_real_2 * config.keep_prob

                with tf.variable_scope("Softmax_3"):
                    self.softmax_w_2 = softmax_w_2 = tf.get_variable("softmax_w", [num_LSTMs_6 * size, class_size])
                    self.softmax_b_2 = softmax_b_2 = tf.get_variable("softmax_b", [class_size])
                    self.logits = logits = tf.nn.softmax(tf.matmul(output_real_2, softmax_w_2) + softmax_b_2)
                    ######
                    self._final_state = mA6.state_6
                    self._lr = tf.Variable(0.0, trainable=False)


    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value, validate_shape=False))

    def get_inference(self):
        return self.logits, self.final_state

    def init_all_var(self, session):
        init = tf.initialize_all_variables()
        session.run(init)

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def training(self):
        return self._training

class Long_Term_0(object):
    def __init__(self, is_training, config, input_data):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps 
        self.class_size = class_size = config.class_size
        self.hidden_size = size = config.hidden_size 
        inputs = input_data

        lstm_cell = tf.nn.rnn_cell.LSTMCell(size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])     #构建多层RNN
        self._initial_state = cell.zero_state(batch_size, tf.float32)     #全0状态,和层数、hidden_size，batch_size有关
        self.outputs_0 = []
        self.state_0 = state_0 = []
        win_size_0 = win_size[0] 
        stride_0 = stride[0]  
        start_time_0 = start_time[0]  
        num_LSTMs_0 = len(range(start_time_0, num_steps - win_size_0 + start_time_0, stride_0))
        print(range(start_time_0, num_steps - win_size_0 + start_time_0, stride_0))
        print("num_LSTMs_0: ", num_LSTMs_0)
        # 正向
        for time_step in range(start_time_0, num_steps):
            for win_step in range(num_LSTMs_0):
                if time_step == start_time_0:
                    self.outputs_0.append([])  
                    state_0.append([])
                    state_0[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step < start_time_0 + win_step * stride_0:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_0[win_step].append(cell_output)
                    elif time_step >= start_time_0 + win_step * stride_0 and time_step < start_time_0 + win_step * stride_0 + win_size_0:
                        if time_step > start_time_0 + win_step * stride_0: tf.get_variable_scope().reuse_variables()

                        if win_step == 0:
                            distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_0, :]) / (
                            start_time_0 + 1)  
                            (cell_output, state_0[win_step]) = cell(distance * 100, state_0[win_step])
                            self.outputs_0[win_step].append(cell_output)
                        else:
                            if time_step < start_time_0 + (win_step-1) * stride_0 + win_size_0: 
                                distance = self.outputs_0[win_step-1][time_step-start_time_0] 
                                (cell_output, state_0[win_step]) = cell(distance, state_0[win_step])
                                self.outputs_0[win_step].append(cell_output)
                            else:
                                distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_0, :]) / (
                                start_time_0 + 1)
                                (cell_output, state_0[win_step]) = cell(distance * 100, state_0[win_step])
                                self.outputs_0[win_step].append(cell_output)
                        #-----------------------------------------------------------------------------------------
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_0[win_step].append(cell_output)

        # Bi-LSTM  反向 ---------------------------------------
        # 两种处理方式：第一，正常得到输出后列表反向(reverse()；第二，创建固定长度从后往前得到
        for time_step in range(num_steps-1, start_time_0-1, -1):
            for win_step in range(num_LSTMs_0, 2*num_LSTMs_0):
                if time_step == num_steps-1:
                    self.outputs_0.append([])
                    state_0.append([])
                    state_0[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step >= start_time_0 + (2*num_LSTMs_0 - win_step - 1) * stride_0 + win_step:   #等号
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_0[win_step].append(cell_output)
                    elif time_step >= start_time_0 + (2*num_LSTMs_0 - win_step - 1) * stride_0 and time_step < start_time_0 + (2*num_LSTMs_0 - win_step - 1) * stride_0 + win_size_0:
                        if time_step < start_time_0 + (2*num_LSTMs_0 - win_step - 1) * stride_0 + win_size_0 - 1: tf.get_variable_scope().reuse_variables()
                        

                        if win_step == num_LSTMs_0:         
                            distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_0, :]) / (
                            start_time_0 + 1)   
                            
                            (cell_output, state_0[win_step]) = cell(distance * 100, state_0[win_step])
                            self.outputs_0[win_step].append(cell_output)
                        else:
                            if time_step >= start_time_0 + (2*num_LSTMs_0 - win_step) * stride_0:  
                                (cell_output, state_0[win_step]) = cell(distance, state_0[win_step])
                                self.outputs_0[win_step].append(cell_output)
                            else:
                                distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_0, :]) / (
                                start_time_0 + 1)
                                
                                (cell_output, state_0[win_step]) = cell(distance * 100, state_0[win_step])
                                self.outputs_0[win_step].append(cell_output)
                        #-----------------------------------------------------------------------------------------
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_0[win_step].append(cell_output)
        # list.reverse() bi-lstm所得到的输出         
        for win_step in range(num_LSTMs_0, 2*num_LSTMs_0):
            self.outputs_0[win_step].reverse()
        #----------------------------bi end----------------------------------------------
        output_0 = []
        for win_step in range(num_LSTMs_0 * 2):
            output_0.append([])
            output_0[win_step] = tf.reshape(tf.concat(self.outputs_0[win_step], 1), [-1, num_steps - start_time_0, size])   #-1代表自动计算维度

        with tf.name_scope("Attention_0"):
            temp_temp_output_0 = []
            # 得到Attention的输出
            for win_step in range(num_LSTMs_0):
                temp_temp_output_0.append([])    
                H = output_0[win_step] + output_0[2*num_LSTMs_0-win_step-1]
                temp_temp_output_0[win_step] = self.attention(H)
        output_0 = temp_temp_output_0

        # cancat
        with tf.variable_scope("Dep_Con_0"):
            temp_output_0 = []
            for win_step in range(num_LSTMs_0):
                temp_output_0.append([])
                temp_output_0[win_step] = tf.reshape(output_0[win_step],
                                                     [batch_size, 1, size])
                if win_step == 0:
                    input_0 = temp_output_0[win_step]
                else:
                    input_0 = tf.concat([input_0, temp_output_0[win_step]], 1)
            input_0 = tf.reshape(input_0, [batch_size, num_LSTMs_0, 1, size])            
            self.out_concat_output_real_0 = tf.reshape(input_0 , [batch_size, num_LSTMs_0 * size])

    def get_depth_concat_output(self):
        return self.out_concat_output_real_0

    def get_state(self):
        return self.state_0

    def attention(self, H):

        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.hidden_size
        
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.nn.leaky_relu(H)
        
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.num_steps - start_time[0]])
        
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = alpha = tf.nn.softmax(restoreM)
        # alpha = tf.nn.softmax(restoreM)
        
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.num_steps - start_time[0], 1]))
        
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, hiddenSize])
        
        sentenceRepren = tf.tanh(sequeezeR)
        
        # 对Attention的输出可以做dropout处理,测试时不要
        # output = tf.nn.dropout(sentenceRepren, 0.4)
        output = sentenceRepren
        
        return output

    @property
    def training(self):
        return self._training

    @property
    def initial_state(self):
        return self._initial_state


class Long_Term_1(object):
    def __init__(self, is_training, config, input_data):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps  
        self.class_size = class_size = config.class_size
        self.hidden_size = size = config.hidden_size 
        inputs = input_data

        lstm_cell = tf.nn.rnn_cell.LSTMCell(size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self.outputs_1 = []
        self.state_1 = state_1 = []
        win_size_1 = win_size[1] 
        stride_1 = stride[1]  
        start_time_1 = start_time[1] 
        num_LSTMs_1 = len(range(start_time_1, num_steps - win_size_1 + start_time_1, stride_1))
        print(range(start_time_1, num_steps - win_size_1 + start_time_1, stride_1))
        print("num_LSTMs_1: ", num_LSTMs_1)
        for time_step in range(start_time_1, num_steps):
            for win_step in range(num_LSTMs_1):
                if time_step == start_time_1:
                    self.outputs_1.append([])
                    state_1.append([])
                    state_1[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step < start_time_1 + win_step * stride_1:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_1[win_step].append(cell_output)
                    elif time_step >= start_time_1 + win_step * stride_1 and time_step < start_time_1 + win_step * stride_1 + win_size_1:
                        if time_step > start_time_1 + win_step * stride_1: tf.get_variable_scope().reuse_variables()

                        if win_step == 0:
                            distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_1, :]) / (
                            start_time_1 + 1)     
                            
                            (cell_output, state_1[win_step]) = cell(distance * 100, state_1[win_step])
                            self.outputs_1[win_step].append(cell_output)
                        else:
                            if time_step < start_time_1 + (win_step-1) * stride_1 + win_size_1:  
                                distance = self.outputs_1[win_step-1][time_step-start_time_1]  

                                (cell_output, state_1[win_step]) = cell(distance, state_1[win_step])
                                self.outputs_1[win_step].append(cell_output)
                            else:
                                distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_1, :]) / (
                                start_time_1 + 1)
                                
                                (cell_output, state_1[win_step]) = cell(distance * 100, state_1[win_step])
                                self.outputs_1[win_step].append(cell_output)
                        #-----------------------------------------------------------------------------------------
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_1[win_step].append(cell_output)

        # Bi-LSTM  反向 ---------------------------------------
        # 两种处理方式：第一，正常得到输出后列表反向(reverse()；第二，创建固定长度从后往前得到
        for time_step in range(num_steps-1, start_time_1-1, -1):
            for win_step in range(num_LSTMs_1, 2*num_LSTMs_1):
                if time_step == num_steps-1:
                    self.outputs_1.append([])
                    state_1.append([])
                    state_1[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step >= start_time_1 + (2*num_LSTMs_1 - win_step - 1) * stride_1 + win_step:   #等号
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_1[win_step].append(cell_output)
                    elif time_step >= start_time_1 + (2*num_LSTMs_1 - win_step - 1) * stride_1 and time_step < start_time_1 + (2*num_LSTMs_1- win_step - 1) * stride_1 + win_size_1:
                        if time_step < start_time_1 + (2*num_LSTMs_1 - win_step - 1) * stride_1 + win_size_1 - 1: tf.get_variable_scope().reuse_variables()
                        

                        if win_step == num_LSTMs_1:         
                            distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_1, :]) / (
                            start_time_1 + 1)     
                            
                            (cell_output, state_1[win_step]) = cell(distance * 100, state_1[win_step])
                            self.outputs_1[win_step].append(cell_output)
                        else:
                            if time_step >= start_time_1 + (2*num_LSTMs_1 - win_step) * stride_1:  
                                distance = self.outputs_1[win_step-1][time_step-start_time_1]  
                                (cell_output, state_1[win_step]) = cell(distance, state_1[win_step])
                                self.outputs_1[win_step].append(cell_output)
                            else:
                                distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_1, :]) / (
                                start_time_1 + 1)
                                
                                (cell_output, state_1[win_step]) = cell(distance * 100, state_1[win_step])
                                self.outputs_1[win_step].append(cell_output)
                        #-----------------------------------------------------------------------------------------
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_1[win_step].append(cell_output)
        # reverse() bi-lstm所得到的输出         
        for win_step in range(num_LSTMs_1, 2*num_LSTMs_1):
            self.outputs_1[win_step].reverse()
        #----------------------------bi end----------------------------------------------
        output_1 = []
        for win_step in range(num_LSTMs_1 * 2):
            output_1.append([])
            output_1[win_step] = tf.reshape(tf.concat(self.outputs_1[win_step], 1), [-1, num_steps - start_time_1, size])

        
        with tf.name_scope("Attention_1"):
            temp_temp_output_1 = []
            # 得到Attention的输出
            for win_step in range(num_LSTMs_1):
                temp_temp_output_1.append([])    
                H = output_1[win_step] + output_1[2*num_LSTMs_1-win_step-1]
                temp_temp_output_1[win_step] = self.attention(H)
        output_1 = temp_temp_output_1

        with tf.variable_scope("Dep_Con_1"):
            temp_output_1 = []
            for win_step in range(num_LSTMs_1):
                temp_output_1.append([])
                temp_output_1[win_step] = tf.reshape(output_1[win_step],
                                                     [batch_size, 1, size])
                if win_step == 0:
                    input_1 = temp_output_1[win_step]
                else:
                    input_1 = tf.concat([input_1, temp_output_1[win_step]], 1)
            input_1 = tf.reshape(input_1, [batch_size, num_LSTMs_1, 1, size])
            self.out_concat_output_real_1 = tf.reshape(input_1, [batch_size, num_LSTMs_1 * size])

    def get_depth_concat_output(self):
        return self.out_concat_output_real_1

    def get_state(self):
        return self.state_1

    def attention(self, H):
        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.hidden_size
        
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.nn.leaky_relu(H)
        
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.num_steps - start_time[1]])
        
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)
        
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.num_steps - start_time[1], 1]))
        
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, hiddenSize])
        
        sentenceRepren = tf.tanh(sequeezeR)
        
        # 对Attention的输出可以做dropout处理
        # output = tf.nn.dropout(sentenceRepren, 0.4)
        output = sentenceRepren
        
        return output

    @property
    def training(self):
        return self._training

    @property
    def initial_state(self):
        return self._initial_state


class Long_Term_2(object):
    def __init__(self, is_training, config, input_data):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps  
        self.class_size = class_size = config.class_size
        self.hidden_size = size = config.hidden_size  
        inputs = input_data

        lstm_cell = tf.nn.rnn_cell.LSTMCell(size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self.outputs_2 = []
        self.state_2 = state_2 = []
        win_size_2 = win_size[2]  
        stride_2 = stride[2] 
        start_time_2 = start_time[2] 
        num_LSTMs_2 = len(range(start_time_2, num_steps - win_size_2 + start_time_2, stride_2))
        print(range(start_time_2, num_steps - win_size_2 + start_time_2, stride_2))
        print("num_LSTMs_2: ", num_LSTMs_2)
        for time_step in range(start_time_2, num_steps):
            for win_step in range(num_LSTMs_2):
                if time_step == start_time_2:
                    self.outputs_2.append([])
                    state_2.append([])
                    state_2[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step < start_time_2 + win_step * stride_2:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_2[win_step].append(cell_output)
                    elif time_step >= start_time_2 + win_step * stride_2 and time_step < start_time_2 + win_step * stride_2 + win_size_2:
                        if time_step > start_time_2 + win_step * stride_2: tf.get_variable_scope().reuse_variables()

                        if win_step == 0:
                            distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_2, :]) / (
                            start_time_2 + 1)     
                            
                            (cell_output, state_2[win_step]) = cell(distance * 100, state_2[win_step])
                            self.outputs_2[win_step].append(cell_output)
                        else:
                            if time_step < start_time_2 + (win_step-1) * stride_2 + win_size_2:  
                                distance = self.outputs_2[win_step-1][time_step-start_time_2]  
                                (cell_output, state_2[win_step]) = cell(distance, state_2[win_step])
                                self.outputs_2[win_step].append(cell_output)
                            else:
                                distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_2, :]) / (
                                start_time_2 + 1)
                                
                                (cell_output, state_2[win_step]) = cell(distance * 100, state_2[win_step])
                                self.outputs_2[win_step].append(cell_output)
                        #-----------------------------------------------------------------------------------------
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_2[win_step].append(cell_output)

        # Bi-LSTM  反向 ---------------------------------------
        # 两种处理方式：第一，正常得到输出后列表反向(reverse()；第二，创建固定长度从后往前得到
        for time_step in range(num_steps-1, start_time_2-1, -1):
            for win_step in range(num_LSTMs_2, 2*num_LSTMs_2):
                if time_step == num_steps-1:
                    self.outputs_2.append([])
                    state_2.append([])
                    state_2[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step >= start_time_2 + (2*num_LSTMs_2 - win_step - 1) * stride_2 + win_step:   
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_2[win_step].append(cell_output)
                    elif time_step >= start_time_2 + (2*num_LSTMs_2 - win_step - 1) * stride_2 and time_step < start_time_2 + (2*num_LSTMs_2 - win_step - 1) * stride_2 + win_size_2:
                        if time_step < start_time_2 + (2*num_LSTMs_2 - win_step - 1) * stride_2 + win_size_2 - 1: tf.get_variable_scope().reuse_variables()
                        

                        if win_step == num_LSTMs_2:         
                            distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_2, :]) / (
                            start_time_2 + 1)     
                            
                            (cell_output, state_2[win_step]) = cell(distance * 100, state_2[win_step])
                            self.outputs_2[win_step].append(cell_output)
                        else:
                            if time_step >= start_time_2 + (2*num_LSTMs_2 - win_step) * stride_2:  
                                distance = self.outputs_2[win_step-1][time_step-start_time_2]  
                                (cell_output, state_2[win_step]) = cell(distance, state_2[win_step])
                                self.outputs_2[win_step].append(cell_output)
                            else:
                                distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_2, :]) / (
                                start_time_2 + 1)
                                
                                (cell_output, state_2[win_step]) = cell(distance * 100, state_2[win_step])
                                self.outputs_2[win_step].append(cell_output)
                        #-----------------------------------------------------------------------------------------
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_2[win_step].append(cell_output)
        # reverse() bi-lstm所得到的输出         
        for win_step in range(num_LSTMs_2, 2*num_LSTMs_2):
            self.outputs_2[win_step].reverse()
        #----------------------------bi end----------------------------------------------
        output_2 = []
        for win_step in range(num_LSTMs_2 * 2):
            output_2.append([])
            output_2[win_step] = tf.reshape(tf.concat(self.outputs_2[win_step], 1), [-1, num_steps - start_time_2, size])

        
        with tf.name_scope("Attention_2"):
            temp_temp_output_2 = []
            # 得到Attention的输出
            for win_step in range(num_LSTMs_2):
                temp_temp_output_2.append([])    
                H = output_2[win_step] + output_2[2*num_LSTMs_2-win_step-1]
                temp_temp_output_2[win_step] = self.attention(H)
        output_2 = temp_temp_output_2
        
        with tf.variable_scope("Dep_Con_2"):
            temp_output_2 = []
            for win_step in range(num_LSTMs_2):
                temp_output_2.append([])
                temp_output_2[win_step] = tf.reshape(output_2[win_step],
                                                     [batch_size, 1, size])
                if win_step == 0:
                    input_2 = temp_output_2[win_step]
                else:
                    input_2 = tf.concat([input_2, temp_output_2[win_step]], 1)
            input_2 = tf.reshape(input_2, [batch_size, num_LSTMs_2, 1, size])
            self.out_concat_output_real_2 = tf.reshape(input_2, [batch_size, num_LSTMs_2 * size])

    def get_depth_concat_output(self):
        return self.out_concat_output_real_2

    def get_state(self):
        return self.state_2

    def attention(self, H):

        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.hidden_size
        
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.nn.leaky_relu(H)
        
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.num_steps - start_time[2]])
        
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)
        
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.num_steps - start_time[2], 1]))
        
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, hiddenSize])
        
        sentenceRepren = tf.tanh(sequeezeR)
        
        # 对Attention的输出可以做dropout处理
        # output = tf.nn.dropout(sentenceRepren, 0.4)
        output = sentenceRepren
        
        return output

    @property
    def training(self):
        return self._training

    @property
    def initial_state(self):
        return self._initial_state


class Short_Term_3(object):
    def __init__(self, is_training, config, input_data):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps  
        self.class_size = class_size = config.class_size
        self.hidden_size = size = config.hidden_size  
        inputs = input_data

        lstm_cell = tf.nn.rnn_cell.LSTMCell(size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self.outputs_3 = []
        self.state_3 = state_3 = []
        win_size_3 = win_size[3]  
        stride_3 = stride[3] 
        start_time_3 = start_time[3]  
        num_LSTMs_3 = len(range(start_time_3, num_steps - win_size_3 + start_time_3, stride_3))
        print(range(start_time_3, num_steps - win_size_3 + start_time_3, stride_3))
        print("num_LSTMs_3: ", num_LSTMs_3)
        for time_step in range(start_time_3, num_steps):
            for win_step in range(num_LSTMs_3):
                if time_step == start_time_3:
                    self.outputs_3.append([])
                    state_3.append([])
                    state_3[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step < start_time_3 + win_step * stride_3:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_3[win_step].append(cell_output)
                    elif time_step >= start_time_3 + win_step * stride_3 and time_step < start_time_3 + win_step * stride_3 + win_size_3:
                        if time_step > start_time_3 + win_step * stride_3: tf.get_variable_scope().reuse_variables()

                        if win_step == 0:
                            distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_3, :]) / (
                            start_time_3 + 1)     
                            
                            (cell_output, state_3[win_step]) = cell(distance * 100, state_3[win_step])
                            self.outputs_3[win_step].append(cell_output)
                        else:
                            if time_step < start_time_3 + (win_step-1) * stride_3 + win_size_3:  
                                distance = self.outputs_3[win_step-1][time_step-start_time_3]  
                                (cell_output, state_3[win_step]) = cell(distance, state_3[win_step])
                                self.outputs_3[win_step].append(cell_output)
                            else:
                                distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_3, :]) / (
                                start_time_3 + 1)
                                
                                (cell_output, state_3[win_step]) = cell(distance * 100, state_3[win_step])
                                self.outputs_3[win_step].append(cell_output)
                        #-----------------------------------------------------------------------------------------
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_3[win_step].append(cell_output)

        # Bi-LSTM  反向 ---------------------------------------
        # 两种处理方式：第一，正常得到输出后列表反向(reverse()；第二，创建固定长度从后往前得到
        for time_step in range(num_steps-1, start_time_3-1, -1):
            for win_step in range(num_LSTMs_3, 2*num_LSTMs_3):
                if time_step == num_steps-1:
                    self.outputs_3.append([])
                    state_3.append([])
                    state_3[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step >= start_time_3 + (2*num_LSTMs_3 - win_step - 1) * stride_3 + win_step:   #等号
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_3[win_step].append(cell_output)
                    elif time_step >= start_time_3 + (2*num_LSTMs_3 - win_step - 1) * stride_3 and time_step < start_time_3 + (2*num_LSTMs_3 - win_step - 1) * stride_3 + win_size_3:
                        if time_step < start_time_3 + (2*num_LSTMs_3 - win_step - 1) * stride_3 + win_size_3 - 1: tf.get_variable_scope().reuse_variables()
                        

                        if win_step == num_LSTMs_3:         
                            distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_3, :]) / (
                            start_time_3 + 1)     
                            
                            (cell_output, state_3[win_step]) = cell(distance * 100, state_3[win_step])
                            self.outputs_3[win_step].append(cell_output)
                        else:
                            if time_step >= start_time_3 + (2*num_LSTMs_3 - win_step) * stride_3:  
                                distance = self.outputs_3[win_step-1][time_step-start_time_3]  
                                (cell_output, state_3[win_step]) = cell(distance, state_3[win_step])
                                self.outputs_3[win_step].append(cell_output)
                            else:
                                distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_3, :]) / (
                                start_time_3 + 1)
                                
                                (cell_output, state_3[win_step]) = cell(distance * 100, state_3[win_step])
                                self.outputs_3[win_step].append(cell_output)
                        #-----------------------------------------------------------------------------------------
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_3[win_step].append(cell_output)
        # reverse() bi-lstm所得到的输出         
        for win_step in range(num_LSTMs_3, 2*num_LSTMs_3):
            self.outputs_3[win_step].reverse()
        #----------------------------bi end----------------------------------------------
        output_3 = []
        for win_step in range(num_LSTMs_3 * 2):
            output_3.append([])
            output_3[win_step] = tf.reshape(tf.concat(self.outputs_3[win_step], 1), [-1, num_steps - start_time_3, size])

        with tf.name_scope("Attention_3"):
            temp_temp_output_3 = []
            # 得到Attention的输出
            for win_step in range(num_LSTMs_3):
                temp_temp_output_3.append([])    
                H = output_3[win_step] + output_3[2*num_LSTMs_3-win_step-1]
                temp_temp_output_3[win_step] = self.attention(H)
        output_3 = temp_temp_output_3
        
        with tf.variable_scope("Dep_Con_3"):
            temp_output_3 = []
            for win_step in range(num_LSTMs_3):
                temp_output_3.append([])
                temp_output_3[win_step] = tf.reshape(output_3[win_step],
                                                     [batch_size, 1, size])
                if win_step == 0:
                    input_3 = temp_output_3[win_step]
                else:
                    input_3 = tf.concat([input_3, temp_output_3[win_step]], 1)
            input_3 = tf.reshape(input_3, [batch_size, num_LSTMs_3, 1, size])
            self.out_concat_output_real_3 = tf.reshape(input_3, [batch_size, num_LSTMs_3 * size])

    def get_depth_concat_output(self):
        return self.out_concat_output_real_3

    def get_state(self):
        return self.state_3

    def attention(self, H):

        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.hidden_size
        
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.nn.leaky_relu(H)
        
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.num_steps - start_time[3]])
        
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)
        
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.num_steps - start_time[3], 1]))
        
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, hiddenSize])
        
        sentenceRepren = tf.tanh(sequeezeR)
        
        # 对Attention的输出可以做dropout处理
        # output = tf.nn.dropout(sentenceRepren, 0.4)
        output = sentenceRepren
        
        return output

    @property
    def training(self):
        return self._training

    @property
    def initial_state(self):
        return self._initial_state


class Short_Term_4(object):
    def __init__(self, is_training, config, input_data):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps  
        self.class_size = class_size = config.class_size
        self.hidden_size = size = config.hidden_size  
        inputs = input_data

        lstm_cell = tf.nn.rnn_cell.LSTMCell(size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self.outputs_4 = []
        self.state_4 = state_4 = []
        win_size_4 = win_size[4]
        stride_4 = stride[4] 
        start_time_4 = start_time[4]  
        num_LSTMs_4 = len(range(start_time_4, num_steps - win_size_4 + start_time_4, stride_4))
        print(range(start_time_4, num_steps - win_size_4 + start_time_4, stride_4))
        print("num_LSTMs_4: ", num_LSTMs_4)
        for time_step in range(start_time_4, num_steps):
            for win_step in range(num_LSTMs_4):
                if time_step == start_time_4:
                    self.outputs_4.append([])
                    state_4.append([])
                    state_4[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step < start_time_4 + win_step * stride_4:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_4[win_step].append(cell_output)
                    elif time_step >= start_time_4 + win_step * stride_4 and time_step < start_time_4 + win_step * stride_4 + win_size_4:
                        if time_step > start_time_4 + win_step * stride_4: tf.get_variable_scope().reuse_variables()

                        if win_step == 0:
                            distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_4, :]) / (
                            start_time_4 + 1)     
                            
                            (cell_output, state_4[win_step]) = cell(distance * 100, state_4[win_step])
                            self.outputs_4[win_step].append(cell_output)
                        else:
                            if time_step < start_time_4 + (win_step-1) * stride_4 + win_size_4:  
                                distance = self.outputs_4[win_step-1][time_step-start_time_4]  
                                (cell_output, state_4[win_step]) = cell(distance, state_4[win_step])
                                self.outputs_4[win_step].append(cell_output)
                            else:
                                distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_4, :]) / (
                                start_time_4 + 1)
                                
                                (cell_output, state_4[win_step]) = cell(distance * 100, state_4[win_step])
                                self.outputs_4[win_step].append(cell_output)
                        #-----------------------------------------------------------------------------------------
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_4[win_step].append(cell_output)

        # Bi-LSTM  反向 ---------------------------------------
        # 两种处理方式：第一，正常得到输出后列表反向(reverse()；第二，创建固定长度从后往前得到
        for time_step in range(num_steps-1, start_time_4-1, -1):
            for win_step in range(num_LSTMs_4, 2*num_LSTMs_4):
                if time_step == num_steps-1:
                    self.outputs_4.append([])
                    state_4.append([])
                    state_4[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step >= start_time_4 + (2*num_LSTMs_4 - win_step - 1) * stride_4 + win_step:  
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_4[win_step].append(cell_output)
                    elif time_step >= start_time_4 + (2*num_LSTMs_4 - win_step - 1) * stride_4 and time_step < start_time_4 + (2*num_LSTMs_4 - win_step - 1) * stride_4 + win_size_4:
                        if time_step < start_time_4 + (2*num_LSTMs_4 - win_step - 1) * stride_4 + win_size_4 - 1: tf.get_variable_scope().reuse_variables()
                        

                        if win_step == num_LSTMs_4:         
                            distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_4, :]) / (
                            start_time_4 + 1)     
                            
                            (cell_output, state_4[win_step]) = cell(distance * 100, state_4[win_step])
                            self.outputs_4[win_step].append(cell_output)
                        else:
                            if time_step >= start_time_4 + (2*num_LSTMs_4 - win_step) * stride_4:  
                                distance = self.outputs_4[win_step-1][time_step-start_time_4]  
                                (cell_output, state_4[win_step]) = cell(distance, state_4[win_step])
                                self.outputs_4[win_step].append(cell_output)
                            else:
                                distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_4, :]) / (
                                start_time_4 + 1)
                                
                                (cell_output, state_4[win_step]) = cell(distance * 100, state_4[win_step])
                                self.outputs_4[win_step].append(cell_output)
                        #-----------------------------------------------------------------------------------------
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_4[win_step].append(cell_output)
        # reverse() bi-lstm所得到的输出         
        for win_step in range(num_LSTMs_4, 2*num_LSTMs_4):
            self.outputs_4[win_step].reverse()
        #----------------------------bi end----------------------------------------------
        output_4 = []
        for win_step in range(num_LSTMs_4 * 2):
            output_4.append([])
            output_4[win_step] = tf.reshape(tf.concat(self.outputs_4[win_step], 1), [-1, num_steps - start_time_4, size])

        with tf.name_scope("Attention_4"):
            temp_temp_output_4 = []
            # 得到Attention的输出
            for win_step in range(num_LSTMs_4):
                temp_temp_output_4.append([])    
                H = output_4[win_step] + output_4[2*num_LSTMs_4-win_step-1]
                temp_temp_output_4[win_step] = self.attention(H)
        output_4 = temp_temp_output_4
        
        with tf.variable_scope("Dep_Con_4"):
            temp_output_4 = []
            for win_step in range(num_LSTMs_4):
                temp_output_4.append([])
                temp_output_4[win_step] = tf.reshape(output_4[win_step],
                                                     [batch_size, 1, size])
                if win_step == 0:
                    input_4 = temp_output_4[win_step]
                else:
                    input_4 = tf.concat([input_4, temp_output_4[win_step]], 1)
            input_4 = tf.reshape(input_4, [batch_size, num_LSTMs_4, 1, size])
            self.out_concat_output_real_4 = tf.reshape(input_4, [batch_size, num_LSTMs_4 * size])

    def get_depth_concat_output(self):
        return self.out_concat_output_real_4

    def get_state(self):
        return self.state_4

    def attention(self, H):

        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.hidden_size
        
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.nn.leaky_relu(H)
        
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.num_steps - start_time[4]])
        
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)
        
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.num_steps - start_time[4], 1]))
        
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, hiddenSize])
        
        sentenceRepren = tf.tanh(sequeezeR)
        
        # 对Attention的输出可以做dropout处理
        # output = tf.nn.dropout(sentenceRepren, 0.4)
        output = sentenceRepren
        
        return output

    @property
    def training(self):
        return self._training

    @property
    def initial_state(self):
        return self._initial_state


class Short_Term_5(object):
    def __init__(self, is_training, config, input_data):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps  
        self.class_size = class_size = config.class_size
        self.hidden_size = size = config.hidden_size  
        inputs = input_data

        lstm_cell = tf.nn.rnn_cell.LSTMCell(size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self.outputs_5 = []
        self.state_5 = state_5 = []
        win_size_5 = win_size[5]  
        stride_5 = stride[5]  
        start_time_5 = start_time[5]  
        num_LSTMs_5 = len(range(start_time_5, num_steps - win_size_5 + start_time_5,
                                stride_5))  
        print(range(start_time_5, num_steps - win_size_5 + start_time_5, stride_5))
        print("num_LSTMs_5: ", num_LSTMs_5)
        for time_step in range(start_time_5, num_steps): 
            for win_step in range(num_LSTMs_5): 
                if time_step == start_time_5:
                    self.outputs_5.append([])
                    state_5.append([])
                    state_5[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step < start_time_5 + win_step * stride_5:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_5[win_step].append(cell_output)
                    elif time_step >= start_time_5 + win_step * stride_5 and time_step < start_time_5 + win_step * stride_5 + win_size_5:
                        if time_step > start_time_5 + win_step * stride_5: tf.get_variable_scope().reuse_variables()

                        if win_step == 0:
                            distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_5, :]) / (
                            start_time_5 + 1)     
                            
                            (cell_output, state_5[win_step]) = cell(distance * 100, state_5[win_step])
                            self.outputs_5[win_step].append(cell_output)
                        else:
                            if time_step < start_time_5 + (win_step-1) * stride_5 + win_size_5:  
                                distance = self.outputs_5[win_step-1][time_step-start_time_5]  
                                (cell_output, state_5[win_step]) = cell(distance, state_5[win_step])
                                self.outputs_5[win_step].append(cell_output)
                            else:
                                distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_5, :]) / (
                                start_time_5 + 1)
                                
                                (cell_output, state_5[win_step]) = cell(distance * 100, state_5[win_step])
                                self.outputs_5[win_step].append(cell_output)
                        #-----------------------------------------------------------------------------------------
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_5[win_step].append(cell_output)

        # Bi-LSTM  反向 ---------------------------------------
        # 两种处理方式：第一，正常得到输出后列表反向(reverse()；第二，创建固定长度从后往前得到
        for time_step in range(num_steps-1, start_time_5-1, -1):
            # end_time_0 = start_time_5 + (num_LSTMs_5 - 1) * stride_5 + win_size_5   1
            for win_step in range(num_LSTMs_5, 2*num_LSTMs_5):
                if time_step == num_steps-1:
                    self.outputs_5.append([])
                    state_5.append([])
                    state_5[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step >= start_time_5 + (2*num_LSTMs_5 - win_step - 1) * stride_5 + win_step:   
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_5[win_step].append(cell_output)
                    elif time_step >= start_time_5 + (2*num_LSTMs_5 - win_step - 1) * stride_5 and time_step < start_time_5 + (2*num_LSTMs_5 - win_step - 1) * stride_5 + win_size_5:
                        if time_step < start_time_5 + (2*num_LSTMs_5 - win_step - 1) * stride_5 + win_size_5 - 1: tf.get_variable_scope().reuse_variables()
                        

                        if win_step == num_LSTMs_5:         
                            distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_5, :]) / (
                            start_time_5 + 1)     
                            
                            (cell_output, state_5[win_step]) = cell(distance * 100, state_5[win_step])
                            self.outputs_5[win_step].append(cell_output)
                        else:
                            if time_step >= start_time_5 + (2*num_LSTMs_5 - win_step) * stride_5:  
                                distance = self.outputs_5[win_step-1][time_step-start_time_5]  
                                (cell_output, state_5[win_step]) = cell(distance, state_5[win_step])
                                self.outputs_5[win_step].append(cell_output)
                            else:
                                distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_5, :]) / (
                                start_time_5 + 1)
                                
                                (cell_output, state_5[win_step]) = cell(distance * 100, state_5[win_step])
                                self.outputs_5[win_step].append(cell_output)
                        #-----------------------------------------------------------------------------------------
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_5[win_step].append(cell_output)
        # reverse() bi-lstm所得到的输出         
        for win_step in range(num_LSTMs_5, 2*num_LSTMs_5):
            self.outputs_5[win_step].reverse()
        #----------------------------bi end----------------------------------------------
        output_5 = []
        for win_step in range(num_LSTMs_5 * 2):
            output_5.append([])
            output_5[win_step] = tf.reshape(tf.concat(self.outputs_5[win_step], 1), [-1, num_steps - start_time_5, size])

        with tf.name_scope("Attention_5"):
            temp_temp_output_5 = []
            # 得到Attention的输出
            for win_step in range(num_LSTMs_5):
                temp_temp_output_5.append([])    
                H = output_5[win_step] + output_5[2*num_LSTMs_5-win_step-1]
                temp_temp_output_5[win_step] = self.attention(H)
        output_5 = temp_temp_output_5
        
        with tf.variable_scope("Dep_Con_5"):
            temp_output_5 = []
            for win_step in range(num_LSTMs_5):
                temp_output_5.append([])
                temp_output_5[win_step] = tf.reshape(output_5[win_step],
                                                     [batch_size, 1, size])
                if win_step == 0:
                    input_5 = temp_output_5[win_step]
                else:
                    input_5 = tf.concat([input_5, temp_output_5[win_step]], 1)
            input_5 = tf.reshape(input_5, [batch_size, num_LSTMs_5, 1, size])
            self.out_concat_output_real_5 = tf.reshape(input_5, [batch_size, num_LSTMs_5 * size])

    def get_depth_concat_output(self):
        return self.out_concat_output_real_5

    def get_state(self):
        return self.state_5

    def attention(self, H):

        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.hidden_size
        
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.nn.leaky_relu(H)
        
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.num_steps - start_time[5]])
        
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)
        
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.num_steps - start_time[5], 1]))
        
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, hiddenSize])
        
        sentenceRepren = tf.tanh(sequeezeR)
        
        # 对Attention的输出可以做dropout处理
        # output = tf.nn.dropout(sentenceRepren, 0.4)
        output = sentenceRepren
        
        return output

    @property
    def training(self):
        return self._training

    @property
    def initial_state(self):
        return self._initial_state


class Angle_6(object):
    def __init__(self, is_training, config, input_data):
        self._training = is_training
        self.batch_size = batch_size = config.batch_size
        self.feature_size = feature_size = config.feature_size
        self.num_steps = num_steps = config.num_steps  
        self.class_size = class_size = config.class_size
        self.hidden_size = size = config.hidden_size  
        inputs = input_data

        lstm_cell = tf.nn.rnn_cell.LSTMCell(size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell])
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self.outputs_6 = []
        self.state_6 = state_6 = []
        win_size_6 = win_size[6]  
        stride_6 = stride[6]  
        start_time_6 = start_time[6]  
        num_LSTMs_6 = len(range(start_time_6, num_steps - win_size_6 + start_time_6,
                                stride_6)) 
        print(range(start_time_6, num_steps - win_size_6 + start_time_6, stride_6))
        print("num_LSTMs_6: ", num_LSTMs_6)
        for time_step in range(start_time_6, num_steps):  
            for win_step in range(num_LSTMs_6): 
                if time_step == start_time_6:
                    self.outputs_6.append([])
                    state_6.append([])
                    state_6[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step < start_time_6 + win_step * stride_6:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_6[win_step].append(cell_output)
                    elif time_step >= start_time_6 + win_step * stride_6 and time_step < start_time_6 + win_step * stride_6 + win_size_6:
                        if time_step > start_time_6 + win_step * stride_6: tf.get_variable_scope().reuse_variables()

                        if win_step == 0:  
                            distance = (inputs[:, time_step, :]) / (start_time_6 + 1)
                            
                            (cell_output, state_6[win_step]) = cell(distance * 100, state_6[win_step])
                            self.outputs_6[win_step].append(cell_output)
                        else:
                            if time_step < start_time_6 + (win_step-1) * stride_6 + win_size_6:  
                                distance = self.outputs_6[win_step-1][time_step-start_time_6]  
                                (cell_output, state_6[win_step]) = cell(distance, state_6[win_step])
                                self.outputs_6[win_step].append(cell_output)
                            else:
                                distance = (inputs[:, time_step, :]) / (start_time_6 + 1)
                                
                                (cell_output, state_6[win_step]) = cell(distance * 100, state_6[win_step])
                                self.outputs_6[win_step].append(cell_output)
                        #-----------------------------------------------------------------------------------------
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_6[win_step].append(cell_output)

        # Bi-LSTM  反向 ---------------------------------------
        # 两种处理方式：第一，正常得到输出后列表反向(reverse()；第二，创建固定长度从后往前得到
        for time_step in range(num_steps-1, start_time_6-1, -1):
            for win_step in range(num_LSTMs_6, 2*num_LSTMs_6):
                if time_step == num_steps-1:
                    self.outputs_6.append([])
                    state_6.append([])
                    state_6[win_step] = self._initial_state
                LSTM_path = os.path.join('LSTM', str(win_step))
                with tf.variable_scope(LSTM_path):
                    if time_step >= start_time_6 + (2*num_LSTMs_6 - win_step - 1) * stride_6 + win_step:   #等号
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_6[win_step].append(cell_output)
                    elif time_step >= start_time_6 + (2*num_LSTMs_6 - win_step - 1) * stride_6 and time_step < start_time_6 + (2*num_LSTMs_6 - win_step - 1) * stride_6 + win_size_6:
                        if time_step < start_time_6 + (2*num_LSTMs_6 - win_step - 1) * stride_6 + win_size_6 - 1: tf.get_variable_scope().reuse_variables()
                        

                        if win_step == num_LSTMs_6:      
                            distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_6, :]) / (
                            start_time_6 + 1)     
                            
                            (cell_output, state_6[win_step]) = cell(distance * 100, state_6[win_step])
                            self.outputs_6[win_step].append(cell_output)
                        else:
                            if time_step >= start_time_6 + (2*num_LSTMs_6 - win_step) * stride_6:  
                                distance = self.outputs_6[win_step-1][time_step-start_time_6]  
                                (cell_output, state_6[win_step]) = cell(distance, state_6[win_step])
                                self.outputs_6[win_step].append(cell_output)
                            else:
                                distance = (inputs[:, time_step, :] - inputs[:, time_step - start_time_6, :]) / (
                                start_time_6 + 1)
                                
                                (cell_output, state_6[win_step]) = cell(distance * 100, state_6[win_step])
                                self.outputs_6[win_step].append(cell_output)
                        #-----------------------------------------------------------------------------------------
                    else:
                        cell_output = tf.zeros([batch_size, size])
                        self.outputs_6[win_step].append(cell_output)
        # reverse() bi-lstm所得到的输出         
        for win_step in range(num_LSTMs_6, 2*num_LSTMs_6):
            self.outputs_6[win_step].reverse()
        #----------------------------bi end----------------------------------------------
        output_6 = []
        for win_step in range(num_LSTMs_6 * 2):
            output_6.append([])
            output_6[win_step] = tf.reshape(tf.concat(self.outputs_6[win_step], 1), [-1, num_steps - start_time_6, size])

        with tf.name_scope("Attention_6"):
            temp_temp_output_6 = []
            # 得到Attention的输出
            for win_step in range(num_LSTMs_6):
                temp_temp_output_6.append([])    
                H = output_6[win_step] + output_6[2*num_LSTMs_6-win_step-1]
                temp_temp_output_6[win_step] = self.attention(H)
        output_6 = temp_temp_output_6
        
        with tf.variable_scope("Dep_Con_6"):
            temp_output_6 = []
            for win_step in range(num_LSTMs_6):
                temp_output_6.append([])
                temp_output_6[win_step] = tf.reshape(output_6[win_step],
                                                     [batch_size, 1, size])
                if win_step == 0:
                    input_6 = temp_output_6[win_step]
                else:
                    input_6 = tf.concat([input_6, temp_output_6[win_step]], 1)
            input_6 = tf.reshape(input_6, [batch_size, num_LSTMs_6, 1, size])
            # concat_output_real_6 = tf.reduce_mean(input_6, 2)
            self.out_concat_output_real_6 = tf.reshape(input_6, [batch_size, num_LSTMs_6 * size])

    def get_depth_concat_output(self):
        return self.out_concat_output_real_6

    def get_state(self):
        return self.state_6

    def attention(self, H):

        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.hidden_size
        
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.nn.leaky_relu(H)
        
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.num_steps - start_time[6]])
        
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)
        
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.num_steps - start_time[6], 1]))
        
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, hiddenSize])
        
        sentenceRepren = tf.tanh(sequeezeR)
        
        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, 0.4)
        output = sentenceRepren
        
        return output

    @property
    def training(self):
        return self._training

    @property
    def initial_state(self):
        return self._initial_state
        
class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 0.1
    learning_rate2 = 0.05
    learning_rate3 = 0.01
    learning_rate4 = 0.005
    learning_rate5 = 0.001
    learning_rate6 = 0.0005
    learning_rate7 = 0.0001

    clearning_rate = 0.05
    clearning_rate2 = 0.01
    clearning_rate3 = 0.005
    clearning_rate4 = 0.001
    clearning_rate5 = 0.0005
    max_grad_norm = 5
    num_layers = 1
    num_steps = 114
    hidden_size1 = 10
    hidden_size2 = 30
    hidden_size = 60

    max_epoch1 = 500
    max_epoch2 = 1200
    max_epoch3 = 2000
    max_epoch4 = 3000
    max_epoch5 = 4000
    max_epoch6 = 6000
    max_max_epoch = 30000

    keep_prob = 1.0
    lr_decay = 0.99
    batch_size = 442
    input_size = 105
    feature_size = 105
    ori_class_size = 20
    class_size = 11
    fusion1_size = 20
    fusion2_size = 20
    AS1 = [2, 3, 5, 6, 10, 13, 18, 20]
    AS2 = [1, 4, 7, 8, 9, 11, 12, 14]
    AS3 = [6, 14, 15, 16, 17, 18, 19, 20]
    use_batch_norm_rnn = False
    use_seq_wise = False
    use_batch_norm = False
    num_frames = num_steps * batch_size
    num_zeros = 0
    mode = 0


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.7
    batch_size = 20
    class_size = 21


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    class_size = 21


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 114
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.8
    batch_size = 1
    class_size = 11


def run_epoch(session, m, data, cos_data, eval_op, verbose=False, is_training=True):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps

    iters = 0
    accuracys = 0.0

    ###
    state = session.run(m.initial_state_L)
    session.run(m.initial_state_S)
    session.run(m.initial_state_Angle)

    sumsums = 0.0
    p_ls = []
    g_ls = []

    for step, (x, z) in enumerate(Action_input.MSR_iterator(data,cos_data, m.batch_size, m.feature_size,
                                                            m.num_steps, is_training)):

        state, real_logits, p_l, _ = session.run(
            [m.final_state, m.real_logits, m.pred_labels,
            eval_op],
            {m.input_data: x, m.input_cos_data: z,
             })            

        for element in p_l:
            p_ls.append(element)

    return real_logits, p_ls    

def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

def get_confusion_matrix(class_size, te_g_l, te_p_l):
    confusion_matrix = np.zeros([class_size, class_size + 1])
    class_prob = np.zeros([class_size])
    for j in range(len(te_g_l)):
        confusion_matrix[te_g_l[j]][te_p_l[j]] += 1
    for j in range(class_size):
        class_prob[j] = confusion_matrix[j][j] / np.sum(confusion_matrix[j][0:class_size])
    for j in range(class_size):
        confusion_matrix[j][class_size] = class_prob[j]
    return confusion_matrix

def get_confusion_matrix_for_plot(class_size, te_g_l, te_p_l):
    confusion_matrix = np.zeros([class_size, class_size])
    class_prob = np.zeros([class_size])
    for j in range(len(te_g_l)):
        confusion_matrix[te_g_l[j]][te_p_l[j]] += 1
    for j in range(class_size):
        class_prob[j] = confusion_matrix[j][j] / np.sum(confusion_matrix[j][0:class_size])

    for row in range(class_size):
        count = np.sum(confusion_matrix[row])
        for index in range(class_size):
            confusion_matrix[row,index] = confusion_matrix[row, index]/count
    
    return confusion_matrix  

def plot_confusion_matrix(cm, classes, src, num, title='Confusion Matrix'):

    plt.figure(figsize=(4, 4), dpi=300)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        plt.text(x_val, y_val, "%0.2f" % (c,), color="white"  if c > cm.max()/2 else "black", fontsize=10, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes)
    plt.yticks(xlocations, classes)
    plt.ylabel('Ground trurh')
    plt.xlabel('Predict')
    
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', color="gray", linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.05)

    plt.show()

def return_result(number):
    if number == 0:
        return '右转'
    elif number == 1:
        return '上行'
    elif number == 2:
        return '下降'
    elif number == 3:
        return '跟随'
    elif number == 4:
        return '盘旋'
    else:
        return '左转'

def play_action(DATA_PATH1, DATA_PATH2, DATA_PATH3):
    config = get_config()
    config.class_size = 6  
    config.feature_size = 112  
    config.input_size = 28    
    config.hidden_size = myHiddenSize
    config.keep_prob = myKeepProb

    eval_config = get_config()
    eval_config.class_size = 6
    eval_config.feature_size = 112      
    eval_config.input_size = 28
    eval_config.hidden_size = myHiddenSize
    eval_config.keep_prob = myKeepProb

    print("------------read data-------------")
    test_sklt1_1, s1, l1 = Action_input.read_test_video(DATA_PATH1, eval_config)
    test_sklt2_1, s2, l2 = Action_input.read_test_video(DATA_PATH2, eval_config)
    test_sklt3_1, s3, l3 = Action_input.read_test_video(DATA_PATH3, eval_config)

    MAX_LENGTH = 111
    print("MAX_LENGTH:",MAX_LENGTH) 
    config.num_steps = MAX_LENGTH
    eval_config.num_steps = MAX_LENGTH

    test_sklt1 = feature_only_diff(test_sklt1_1, MAX_LENGTH, eval_config)
    test_sklt2 = feature_only_diff(test_sklt2_1, MAX_LENGTH, eval_config)
    test_sklt3 = feature_only_diff(test_sklt3_1, MAX_LENGTH, eval_config)


    del test_sklt1_1, test_sklt2_1, test_sklt3_1

    #坐标变换。变原点
    feature_test1 = body_rotation(test_sklt1)
    feature_test_cos1 = body_rotation_cos(test_sklt1)
    feature_test2 = body_rotation(test_sklt2)
    feature_test_cos2 = body_rotation_cos(test_sklt2)
    feature_test3 = body_rotation(test_sklt3)
    feature_test_cos3 = body_rotation_cos(test_sklt3)

    #-------------------------------------扩展数据-----------------------------------------
    feature_test1 = np.concatenate((feature_test1,feature_test1,feature_test1,feature_test1), axis=2)
    feature_test_cos1 = np.concatenate((feature_test_cos1,feature_test_cos1,feature_test_cos1,feature_test_cos1),axis=2)
    feature_test2 = np.concatenate((feature_test2,feature_test2,feature_test2,feature_test2),axis=2)
    feature_test_cos2 = np.concatenate((feature_test_cos2,feature_test_cos2,feature_test_cos2,feature_test_cos2),axis=2)
    feature_test3 = np.concatenate((feature_test3,feature_test3,feature_test3,feature_test3),axis=2)
    feature_test_cos3 = np.concatenate((feature_test_cos3,feature_test_cos3,feature_test_cos3,feature_test_cos3),axis=2)
    
    del test_sklt1, test_sklt2, test_sklt3
    print("feature_test1.shape:",feature_test1.shape)
    print("feature_test2.shape:",feature_test2.shape)
    print("feature_test3.shape:",feature_test3.shape)

    eval_config.batch_size = np.int32(len(feature_test1))
    eval_config.batch_size = np.int32(len(feature_test1))

    # TODO===========================================================================================


    # TODO=========================================================================================== 

    sessConfig = tf.ConfigProto(log_device_placement=False)   #配置tf.Session的运算方式，比如gpu运算或者cpu运算
    sessConfig.gpu_options.allow_growth = True

    writeConfig_tocsv = False

    if writeConfig_tocsv:
        csvWriter.writerow(['DateTime:', strftime("%Y%m%d_%H:%M:%S", localtime())])
        csvWriter.writerow([])
        csvWriter.writerow(['Total Dataset Length', 'Train Batch Divider', 'Train Batch Size', 'Eval Batch Size', ])
        csvWriter.writerow(
            [len(feature_train), len(feature_train) / config.batch_size, config.batch_size, eval_config.batch_size])

        csvWriter.writerow(['Control', 'Long 0', 'Long 1', 'Long 2', 'Medium 3', 'Medium 4', 'Short 5'])
        csvWriter.writerow(['win_size', win_size[0], win_size[1], win_size[2], win_size[3], win_size[4], win_size[5]])
        csvWriter.writerow(['stride', stride[0], stride[1], stride[2], stride[3], stride[4], stride[5]])
        csvWriter.writerow(
            ['start_time', start_time[0], start_time[1], start_time[2], start_time[3], start_time[4], start_time[5]])
        csvWriter.writerow([])

    # TODO=========================================================================================== 
    with tf.Graph().as_default(), tf.Session(config=sessConfig) as session:
        with tf.device('/cpu:0'):
            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)      

            with tf.variable_scope("model", reuse=tf.AUTO_REUSE, initializer=initializer):
                mtest = Sim_runner(is_training=False, config=eval_config)

            print("\nTesting Model Established!!\n")

            saver = tf.train.Saver(tf.global_variables())   #保存和加载模型

            saver.restore(session, "./TrialSaver/Bi Long-Short Term Att/TJ_Bi Long-Short Term Att.ckpt-400")    
            print("Model restored.")

            stt_loop = time.time()
            print(strftime("%Y%m%d_%H:%M:%S", localtime()))


            stt_lr = time.time()
      
            logits1, te_p_l1 = run_epoch(session,
                                        mtest,
                                        feature_test1,
                                        feature_test_cos1,
                                        tf.no_op(),
                                        is_training=False)
            logits2, te_p_l2 = run_epoch(session,
                                        mtest,
                                        feature_test2,
                                        feature_test_cos2,                                                                                                           
                                        tf.no_op(),
                                        is_training=False)
            logits3, te_p_l3 = run_epoch(session,
                                        mtest,
                                        feature_test3,
                                        feature_test_cos3,
                                        tf.no_op(),
                                        is_training=False)                                                                                                                                                                                                                                                                       
            

            w1 = (s1/(s1+s2+s3) + l1/(l1+l2+l3))/2
            w2 = (s2/(s1+s2+s3) + l2/(l1+l2+l3))/2
            w3 = (s3/(s1+s2+s3) + l3/(l1+l2+l3))/2
            print(te_p_l1, te_p_l2, te_p_l3)
            view1 = return_result(te_p_l1[0])
            view2 = return_result(te_p_l2[0])
            view3 = return_result(te_p_l3[0])


            logits_fusion = (w1 * logits1 + w2 * logits2 + w3 * logits3)
            print(logits_fusion)
            print("weight")
            te_p_l_fusion = session.run(tf.argmax(logits_fusion, 1))
            print(te_p_l_fusion)
            view_fusion = return_result(te_p_l_fusion[0])

            # w1 = s1/(s1+s2+s3)
            # w2 = s2/(s1+s2+s3)
            # w3 = s3/(s1+s2+s3)
            # logits_fusion = (w1 * logits1 + w2 * logits2 + w3 * logits3)
            # print("only W")
            
            # logits_fusion = (logits1 + logits2 + logits3)/3   #不同融合方式
            # print("ave")
            #------------------------------
            # print("1",confusion_matrix1)
            # print("2",confusion_matrix2)
            # print("3",confusion_matrix3)
            # print("Test Accuracy: %.5f\n" % (test_accuracy))
            # csvWriter.writerow(["Test Accuracy :", test_accuracy])

            # argmax返回下标
            # correct_prediction = tf.equal(tf.argmax(logits_fusion, 1), tf.argmax(targets1, 1))     # only need one targets        #argmax返回下标
            # te_pl = tf.argmax(logits_fusion, 1)   #confusion matrix
            # te_pl = session.run(te_pl)
            # fusion_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) / eval_config.batch_size
            # fusion_accuracy = session.run(fusion_accuracy)
            # print("fusion_accuracy is:",fusion_accuracy)

            # for j in range(len(te_g_l)):

            # for j in range(len(te_g_l1)):
            #     confusion_matrix[te_g_l1[j]][te_pl[j]] += 1
            # for j in range(config.class_size):
            #     class_prob[j] = confusion_matrix[j][j] / np.sum(confusion_matrix[j][0:config.class_size])
            # for j in range(config.class_size):
            #     confusion_matrix[j][config.class_size] = class_prob[j]
            # print(confusion_matrix)
            # with open(folder_path + "/view-test-" + str(i) + ".csv", "w") as csvfile:
            #     csvwriter2 = csv.writer(csvfile)
            #     for j in range(config.class_size):
            #         csvwriter2.writerow(confusion_matrix[j])
    return view1, view2, view3, view_fusion

# if __name__ == "__main__":
#     tf.app.run()