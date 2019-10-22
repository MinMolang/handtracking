"""Main interface for hand-tracking Pong."""

from handtracking.utils import detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import matplotlib.pyplot as plt


detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    for i in range(26):
        img_name = "image"+str(i)+".jpg"
        im_width, im_height = (1280, 720)
        img_raw = cv2.imread(img_name)
        num_hands_detect = 2
        img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        boxes, scores, classes = detector_utils.detect_objects(img,
                                                   detection_graph, sess)
        detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                             scores, boxes, classes, im_width, im_height,
                             img,i)
        #plt.imshow(img)
