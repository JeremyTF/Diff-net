#! /usr/bin/env python
# coding=utf-8

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOV3

class YoloTest(object):
    def __init__(self):
        self.input_size       = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes      = len(self.classes)
        self.anchors          = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold  = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold    = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path  = cfg.TEST.ANNOT_PATH
        self.weight_file      = cfg.TEST.WEIGHT_FILE
        self.write_image      = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label       = cfg.TEST.SHOW_LABEL

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.input_mask = tf.placeholder(dtype=tf.float32, name='input_mask')
            self.trainable  = tf.placeholder(dtype=tf.bool,    name='trainable')

        model = YOLOV3(self.input_data, self.input_mask, self.trainable)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox

        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3)  
        # config=tf.ConfigProto(gpu_options=gpu_options) 

        config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        # config.gpu_options.allow_growth = True   

        self.sess  = tf.Session(config=config)
        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.weight_file)

    def predict(self, image, mask):

        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data, mask_data = utils.image_preporcess(image, mask, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]
        mask_data = mask_data[np.newaxis, ..., np.newaxis]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input_data: image_data,
                self.input_mask: mask_data,
                self.trainable: False
            }
        )

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes

    def encoder(self, image, bboxes):

        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data, mask_data = utils.image_preporcess(image, mask, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        features = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input_data: image_data,
                self.trainable: False
            }
        )


        return features

    def get_image_from_dataset(self):
        # predicted_dir_path = './mAP/predicted'
        # ground_truth_dir_path = './mAP/ground-truth'
        # if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        # if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        # os.mkdir(predicted_dir_path)
        # os.mkdir(ground_truth_dir_path)

        file_name_list = []
        with open(self.annotation_path, 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                annotation = line.strip().split()
                image_path = annotation[0]
                mask_path = annotation[1]
                file_name_list.append([image_path,mask_path])
        return file_name_list

    def get_video_from_dataset(self):

        file_name_list = []
        with open('/home/wangning/Desktop/data/haidian/label.txt', 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                annotation = line.strip().split()
                image_name = annotation[0]
                bboxes = annotation[1:]
                image_folder = image_name.split('_')[0]+'_'+image_name.split('_')[1]
                image_path = os.path.join('/home/wangning/Desktop/data/haidian/images',image_folder,image_name)
                if os.path.exists(image_path):
                    file_name_list.append([image_path,bboxes])
        return file_name_list


    def evaluate(self,image,mask):
        bboxes_pr = self.predict(image, mask)
        return  bboxes_pr







if __name__ == '__main__':

    diffnet = YoloTest()
    file_list = diffnet.get_image_from_dataset()
    for image_info in file_list:
        diffnet.evaluate(image_info[0],image_info[1])



