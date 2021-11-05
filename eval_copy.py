#! /usr/bin/env python
# coding=utf-8
import json

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOV3

def load_info_from_json(json_file_path):
    with open(json_file_path,'r') as load_f:
        dict_info = json.load(load_f)

    return dict_info

def parser_bbox_str(bbox):
    bbox_str = bbox.split('[')[1].split(']')[0].split(' ')
    bbox_str = [eval(x) for x in bbox_str if x != '']
    return bbox_str

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

    def evaluate(self):
        predicted_dir_path = './mAP/predicted'
        ground_truth_dir_path = './mAP/ground-truth'
        if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(self.write_image_path): shutil.rmtree(self.write_image_path)
        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)
        os.mkdir(self.write_image_path)

        val_info_list = []
        file_info_list = load_info_from_json('/home/wangning/Desktop/data/yizhuang/generation/total_video.json')
        for num, image_info in enumerate(file_info_list):
            if (image_info['img_path'].split('images')[1]).split('/')[1] in ['HQEV503_20201215091205','MKZ156_20201215112828','MKZ073_20201216225330']:
                val_info_list.append(image_info)
        count = 0
        tmp_file_name = 0
        save_video_folder_num = 0
        for num, image_info in enumerate(val_info_list):
            ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')
            predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')

            img_path = image_info['img_path']
            msk_path = image_info['msk_path']
            image = cv2.imread(img_path)
            mask = cv2.imread(msk_path,0)*180
            bboxes_pr = self.predict(image, mask)

            image_file_name = int((img_path.split('images')[1]).split('/')[2].split('_')[2])




            with open(predict_result_path, 'w') as f:
                for bbox in bboxes_pr:
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                    class_ind = int(bbox[5])
                    class_name = self.classes[class_ind]
                    score = '%.4f' % score
                    xmin, ymin, xmax, ymax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    # print('\t' + str(bbox_mess).strip())

            with open(ground_truth_path, 'w') as f:
                be_corrected_box_list = image_info['be_correct']
                to_add_box = image_info['to_add']
                to_del_box = image_info['to_del']
                for bbox in be_corrected_box_list:
                    xmin,ymin,xmax,ymax = bbox[1:-1].split(',')
                    bbox_mess = ' '.join(['be_corrected', xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)

                xmin, ymin, xmax, ymax = to_add_box[1:-1].split(',')
                bbox_mess = ' '.join(['be_add', xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)

                xmin, ymin, xmax, ymax = to_add_box[1:-1].split(',')
                bbox_mess = ' '.join(['be_del', xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)

            if True:
                image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
                if num == 0:
                    tmp_file_name=image_file_name

                same_video_flag = (image_file_name-tmp_file_name)<5
                if not same_video_flag:
                    save_video_folder_num+=1
                    tmp_file_name = image_file_name
                    count = 0


                else:
                    tmp_file_name = image_file_name
                    count += 1

                if not os.path.exists(self.write_image_path+'/'+str(save_video_folder_num)):
                    os.mkdir(self.write_image_path+'/'+str(save_video_folder_num))
                cv2.imwrite(self.write_image_path+'/'+str(save_video_folder_num)+'/'+ str(count)+'.jpg', image)



                cv2.imshow('image',image)
                cv2.waitKey(1)
            print(num)








    # def voc_2012_test(self, voc2012_test_path):
    #
    #     img_inds_file = os.path.join(voc2012_test_path, 'ImageSets', 'Main', 'test.txt')
    #     with open(img_inds_file, 'r') as f:
    #         txt = f.readlines()
    #         image_inds = [line.strip() for line in txt]
    #
    #     results_path = 'results/VOC2012/Main'
    #     if os.path.exists(results_path):
    #         shutil.rmtree(results_path)
    #     os.makedirs(results_path)
    #
    #     for image_ind in image_inds:
    #         image_path = os.path.join(voc2012_test_path, 'JPEGImages', image_ind + '.jpg')
    #         image = cv2.imread(image_path)
    #
    #         print('predict result of %s:' % image_ind)
    #         bboxes_pr = self.predict(image)
    #         for bbox in bboxes_pr:
    #             coor = np.array(bbox[:4], dtype=np.int32)
    #             score = bbox[4]
    #             class_ind = int(bbox[5])
    #             class_name = self.classes[class_ind]
    #             score = '%.4f' % score
    #             xmin, ymin, xmax, ymax = list(map(str, coor))
    #             bbox_mess = ' '.join([image_ind, score, xmin, ymin, xmax, ymax]) + '\n'
    #             with open(os.path.join(results_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
    #                 f.write(bbox_mess)
    #             print('\t' + str(bbox_mess).strip())


if __name__ == '__main__':
    YoloTest().evaluate()



