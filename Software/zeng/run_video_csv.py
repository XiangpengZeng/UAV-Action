import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import csv
import os                  #zeng

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
#修改estimator和此文件


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    #parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

#--------------------------------------------------------------zeng------------------------------
    for root,dirs,files in os.walk("D:/#Master research/Action Detection/TS-LSTM-master/TS-LSTM-master/UCLA_Code/NWUCLA_csv_copy"):   #遍历文件夹
        for file_name in files:
            save_list = []                                    #初始置空
            csv_path = root+'/'+file_name                     #遍历原文件,注意filename的存在,很重要

            cap = cv2.VideoCapture(args.video)
            csv_save_path = 'D:/#Master research/Action Detection/TS-LSTM-master/TS-LSTM-master/UCLA_Code/'+'/modify_csv/'+file_name     #存储

            if cap.isOpened() is False:
                print("Error opening video stream or file")
            while cap.isOpened():
                ret_val, image = cap.read()
                if ret is False:
                    video.release()                          #这句是为了在一个视频读完以后跳出while进入下一个视频 zeng
                    print('please to check the video')
                    break

                humans = e.inference(image)
                if not args.showBG:                                                                        #background
                    image = np.zeros(image.shape)
                image, save_list_single = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)        #zeng
                save_list.append(save_list_single)                                                         #zeng

                cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow('tf-pose-estimation result', image)
                fps_time = time.time()

                with open(csv_save_path,"w",newline='') as w:      #有后面那个参数是为了避免写入时出现空格
                    csv_writter = csv.writer(w)                    #创建writter对象
                    for x in range(len(save_list)):
                        csv_writter.writerow(save_list[x])                    #支持按行写入，zeng
                    w.close()

                if cv2.waitKey(1) == 27:
                    break

    cv2.destroyAllWindows()
logger.debug('finished+')
