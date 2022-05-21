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

def play(image):
    
    t = time.time()

    fps_time = time.time()
    image = cv2.resize(image, (960, 540))
    with tf.device('/gpu:0'):
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    elapsed = time.time() - t

    #logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    image, savelist = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    
    logger.warning('matplitlib error, %s' % e)
    # fps
    # cv2.putText(image, "FPS: %f" % (1.0 / (time.time() -fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.waitKey(1)
    return image

# play()