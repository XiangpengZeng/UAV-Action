import numpy as np
from sklearn import preprocessing

def one_hot_labeling(original_label, config):
    new_label = np.zeros([len(original_label), config.class_size])

    for batch_step in range(len(original_label)):
        new_label[batch_step][original_label[batch_step]-1] = 1     

    return new_label


def body_rotation(feature_train):
    ret1, _ = body_rotation_2(feature_train, feature_train)
    return ret1



def body_rotation_2(feature_train, feature_test):                  
    feature_hc_train = np.zeros(feature_train.shape)
    feature_hc_test = np.zeros(feature_test.shape)

    for batchNo in range(feature_train.shape[0]):
        first_index = False
        for frmNo in range(feature_train.shape[1]):
            if (np.linalg.norm(feature_train[batchNo][frmNo]) != 0) and (first_index == False):
                new_origin = feature_train[batchNo][frmNo][2:4]        #坐标转换，针对每行数据设计新原点，自己选14个点，此处原点：脖子

                for nodeNo in range(np.int32(feature_train.shape[2] / 2)):                  #多少个点的二维数据
                    feature_hc_train[batchNo][frmNo][2 * nodeNo:2 * (nodeNo + 1)] \
                        = feature_train[batchNo][frmNo][2 * nodeNo:2 * (nodeNo + 1)] - new_origin
                first_index = True          
            elif (np.linalg.norm(feature_train[batchNo][frmNo]) != 0) and (first_index == True):
                for nodeNo in range(np.int32(feature_train.shape[2] / 2)):
                    feature_hc_train[batchNo][frmNo][2 * nodeNo:2 * (nodeNo + 1)] \
                        = feature_train[batchNo][frmNo][2 * nodeNo:2 * (nodeNo + 1)] - new_origin

            else:
                pass

    for batchNo in range(feature_test.shape[0]):
        first_index = False
        for frmNo in range(feature_test.shape[1]):
            if (np.linalg.norm(feature_test[batchNo][frmNo]) != 0) and (first_index == False):
                new_origin = feature_test[batchNo][frmNo][2:4]
                for nodeNo in range(np.int32(feature_test.shape[2] / 2)):
                    feature_hc_test[batchNo][frmNo][2 * nodeNo:2 * (nodeNo + 1)] \
                        = feature_test[batchNo][frmNo][2 * nodeNo:2 * (nodeNo + 1)] - new_origin
                        
                first_index = True
            elif (np.linalg.norm(feature_test[batchNo][frmNo]) != 0) and (first_index == True):
                for nodeNo in range(np.int32(feature_test.shape[2] / 2)):
                    feature_hc_test[batchNo][frmNo][2 * nodeNo:2 * (nodeNo + 1)] \
                        = feature_test[batchNo][frmNo][2 * nodeNo:2 * (nodeNo + 1)] - new_origin
            else:
                pass


    return feature_hc_train, feature_hc_test

#cosine similarity
def cos_similarity(x1,y1,x2,y2):
    a = np.array([x1,y1])
    b = np.array([x2,y2])
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return float(1) if a.all() == b.all() else float(0)  
    cos_sim = np.dot(a,b)/(a_norm * b_norm)
    return cos_sim

def body_rotation_cos(feature_train):                     
    feature_hc_train = np.zeros(feature_train.shape)
    feature_temp = np.zeros(feature_train.shape)       

    for batchNo in range(feature_train.shape[0]):
        for frmNo in range(feature_train.shape[1]):
            if (np.linalg.norm(feature_train[batchNo][frmNo]) != 0):
                new_origin = feature_train[batchNo][frmNo][2:4]        #坐标转换，针对每行数据设计新原点，自己选14个点，此处原点：脖子

                for nodeNo in range(np.int32(feature_train.shape[2] / 2)):                 

                    feature_temp[batchNo][frmNo][2 * nodeNo:2 * (nodeNo + 1)] \
                        = feature_train[batchNo][frmNo][2 * nodeNo:2 * (nodeNo + 1)] - new_origin
                for nodeNo_1 in range(np.int32(2),np.int32(feature_train.shape[2] / 2)): 
                    feature_hc_train[batchNo][frmNo][nodeNo_1-2] \
                        =cos_similarity(feature_temp[batchNo][frmNo][0],feature_temp[batchNo][frmNo][1],feature_temp[batchNo][frmNo][2*nodeNo_1],feature_temp[batchNo][frmNo][2*nodeNo_1+1])                              
                
                #向量数据扩充
                for nodeNo_2 in range(np.int32(feature_train.shape[2] / 2), np.int32(feature_train.shape[2])):
                    feature_hc_train[batchNo][frmNo][nodeNo_2] \
                        = feature_hc_train[batchNo][frmNo][nodeNo_2 - np.int32(feature_train.shape[2] / 2)]
            else:
                pass

    return feature_hc_train

def feature_only_diff_0(data, maxValue, config):
    new_data = np.zeros([len(data), maxValue, config.feature_size])
    for batch_step in range(len(data)):
        new_data[batch_step][maxValue-len(data[batch_step]):maxValue] = data[batch_step]

    return new_data

def Pose_Motion(feature):
    Feature_PM = np.zeros(feature.shape)
    for time_step in range(feature.shape[1]):
        if time_step > 0:
            Feature_PM[:, time_step, :] = feature[:, time_step, :] - feature[:, time_step - 1, :]
        else:
            Feature_PM[:, time_step, :] = feature[:, time_step, :]

    return Feature_PM

def feature_only_diff(data, maxValue, config):
    new_data = np.zeros([len(data), maxValue, config.input_size]) 

    for batch_step in range(len(data)):       

        # x,y分段的规范化
        raw_data = np.array(data[batch_step])    
        norl_data = np.zeros([raw_data.shape[0],raw_data.shape[1]])
        x_data = raw_data[:,0:raw_data.shape[1]:2]
        y_data = raw_data[:,1:raw_data.shape[1]+1:2]
        x_data = preprocessing.minmax_scale(x_data,feature_range=(0,1),axis=0)
        y_data = preprocessing.minmax_scale(y_data,feature_range=(0,1),axis=0)
        norl_data[:,0:norl_data.shape[1]:2] = x_data
        norl_data[:,1:norl_data.shape[1]+1:2] = y_data

        new_data[batch_step][0:len(data[batch_step])] = norl_data 
        del norl_data
        for time_step in range(maxValue):
            if np.sum(new_data[batch_step][time_step]) != 0:
                for ttime_step in range(time_step):
                    new_data[batch_step][ttime_step] = new_data[batch_step][time_step]   #给0值地方赋值
                break
            else:
                pass

    return new_data                
