import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import tensorflow as tf

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

parser = argparse.ArgumentParser(description='tf-pose-estimation run')
#parser.add_argument('--image', type=str, default='./images/p1.jpg')
parser.add_argument('--model', type=str, default='cmu',
                    help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
parser.add_argument('--resize', type=str, default='0x0',
                    help='if provided, resize images before they are processed. '
                            'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                    help='if provided, resize heatmaps before they are post-processed. default=1.0')

args = parser.parse_args()

w, h = model_wh(args.resize)
if w == 0 or h == 0:
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
else:
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

# save_list = []   #总的
# save_list2 = []   #总的
# save_list3 = []   #总的

#fps_time = 0   #zeng
# if __name__ == '__main__':
def play(image, image2, image3):
    
    #image = common.read_imgfile(args.image, None, None)
    # cap = cv2.VideoCapture("/home/dreamer/Desktop/a05_s05_p01_e01_v01.mp4")

    t = time.time()
    # while True:
    # ret, image = cap.read()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    fps_time = time.time()
    image = cv2.resize(image, (640, 360))
    image2 = cv2.resize(image2, (640, 360))
    image3 = cv2.resize(image3, (640, 360))
    with tf.device('/gpu:0'):
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        humans2 = e.inference(image2, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        humans3 = e.inference(image3, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    elapsed = time.time() - t

    #logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    image, savelist = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    savelist = list(map(lambda x:3*x, savelist))
    # save_list.append(savelist)         #汇单成总
    image2, savelist2 = TfPoseEstimator.draw_humans(image2, humans2, imgcopy=False)
    savelist2 = list(map(lambda x:3*x, savelist2))   # 乘以3倍，还原到初始大小
    # save_list2.append(savelist2)         #汇单成总
    image3, savelist3 = TfPoseEstimator.draw_humans(image3, humans3, imgcopy=False)
    savelist3 = list(map(lambda x:3*x, savelist3))
    # save_list3.append(savelist3)         #汇单成总

    # try:
    #     import matplotlib.pyplot as plt

    #     fig = plt.figure()
    #     a = fig.add_subplot(2, 2, 1)
    #     a.set_title('Result')
    #     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #     bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    #     bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

    #     # show network output
    #     a = fig.add_subplot(2, 2, 2)
    #     plt.imshow(bgimg, alpha=0.5)
    #     tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
    #     plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    #     plt.colorbar()

    #     tmp2 = e.pafMat.transpose((2, 0, 1))
    #     tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)elapsed
    #     tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    #     a = fig.add_subplot(2, 2, 3)
    #     a.set_title('Vectormap-x')
    #     # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    #     plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    #     plt.colorbar()

    #     a = fig.add_subplot(2, 2, 4)
    #     a.set_title('Vectormap-y')
    #     # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    #     plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    #     plt.colorbar()
    #     plt.show()
    # except Exception as e:
    logger.warning('matplitlib error, %s' % e)
    cv2.putText(image, "FPS: %f" % (1.0 / (time.time() -fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # cv2.imshow('result', image)
    # if cv2.waitKey(1) & 0XFF ==ord('q'):
    #     break
    cv2.waitKey(1)
    return image, image2, image3, savelist, savelist2, savelist3
    # cv2.destroyAllWindows()

# play()