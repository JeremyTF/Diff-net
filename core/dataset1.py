#! /usr/bin/env python
# coding=utf-8
import json
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import yaml
import core.utils as utils
from core.config import cfg
import time
from concurrent.futures import ThreadPoolExecutor
import numexpr as ne

np.seterr(divide='ignore', invalid='ignore')

class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset_type):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
    def _reclear_anno(self,anno):
        total_image_info = []
        for task_id in anno.keys():
            task_dict = anno[task_id]
            for video_id in task_dict.keys():
                video_dict = task_dict[video_id]
                for image_info in video_dict:
                    total_image_info.append(image_info)
        return total_image_info

    def _clear_anno(self,anno_list):
        new_anno_list = []
        for anno in anno_list:
            line = anno.split()
            image_path = line[0]
            mask_path = line[1]
            if not os.path.exists(image_path):
                continue
            if not os.path.exists(mask_path):
                continue
            new_anno_list.append(anno)
        return new_anno_list

    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            txt = [line.replace('New_yizhuang/', '/home/wangning/Desktop/data/New_yizhuang/') for line in txt]
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:-1]) != 0]

            # print annotations
        np.random.shuffle(annotations)
        return annotations

    def load_info_from_json(self,json_file_path):
        with open(json_file_path, 'r') as load_f:
            dict_info = json.load(load_f)
        return dict_info

    def __iter__(self):
        return self
    #
    # def __next__(self):
    #
    #     with tf.device('/cpu:0'):
    #         self.train_input_size = random.choice(self.train_input_sizes)
    #         self.train_output_sizes = self.train_input_size // self.strides
    #
    #         batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))
    #
    #         batch_mask = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 1))
    #
    #         batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
    #                                       self.anchor_per_scale, 5 + self.num_classes))
    #         batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
    #                                       self.anchor_per_scale, 5 + self.num_classes))
    #         batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
    #                                       self.anchor_per_scale, 5 + self.num_classes))
    #
    #         batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
    #         batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
    #         batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
    #
    #         num = 0
    #         if self.batch_count < self.num_batchs:
    #             while num < self.batch_size:
    #                 index = self.batch_count * self.batch_size + num
    #                 if index >= self.num_samples: index -= self.num_samples
    #                 annotation = self.annotations[index]
    #                 image, mask, bboxes = self.parse_annotation(annotation)
    #                 if image is None or mask is None:
    #                     continue
    #                 label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
    #
    #                 batch_image[num, :, :, :] = image
    #                 batch_mask[num, :, :, :] = mask[:, :, np.newaxis]
    #                 batch_label_sbbox[num, :, :, :, :] = label_sbbox
    #                 batch_label_mbbox[num, :, :, :, :] = label_mbbox
    #                 batch_label_lbbox[num, :, :, :, :] = label_lbbox
    #                 batch_sbboxes[num, :, :] = sbboxes
    #                 batch_mbboxes[num, :, :] = mbboxes
    #                 batch_lbboxes[num, :, :] = lbboxes
    #                 num += 1
    #             self.batch_count += 1
    #             return batch_image, batch_mask, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
    #                    batch_sbboxes, batch_mbboxes, batch_lbboxes
    #         else:
    #             self.batch_count = 0
    #             np.random.shuffle(self.annotations)
    #             raise StopIteration

    def __next__(self):

        pool = ThreadPoolExecutor(max_workers=10)

        thread0 = pool.submit(self.get_data, 0, 1)
        thread1 = pool.submit(self.get_data, 1, 0)



        batch_image0, batch_mask0,batch_label_sbbox0, batch_label_mbbox0, batch_label_lbbox0, \
        batch_sbboxes0, batch_mbboxes0, batch_lbboxes0 = thread0.result()

        batch_image1, batch_mask1,batch_label_sbbox1, batch_label_mbbox1, batch_label_lbbox1, \
        batch_sbboxes1, batch_mbboxes1, batch_lbboxes1 = thread1.result()



        # start threads to add numpy array

        thread2 = pool.submit(self.add_numpy, batch_image0, batch_image1)

        thread3 = pool.submit(self.add_numpy,batch_mask0,batch_mask1)
        thread4 = pool.submit(self.add_numpy, batch_label_sbbox0, batch_label_sbbox1
                               )

        thread5 = pool.submit(self.add_numpy, batch_label_mbbox0, batch_label_mbbox1,
                               )

        thread6 = pool.submit(self.add_numpy, batch_label_lbbox0, batch_label_lbbox1,
                               )

        thread7 = pool.submit(self.add_numpy, batch_sbboxes0, batch_sbboxes1,  )

        thread8 = pool.submit(self.add_numpy, batch_mbboxes0, batch_mbboxes1,  )

        thread9 = pool.submit(self.add_numpy, batch_lbboxes0, batch_lbboxes1,  )


        batch_image = thread2.result()

        batch_mask = thread3.result()

        batch_label_sbbox = thread4.result()

        batch_label_mbbox = thread5.result()

        batch_label_lbbox = thread6.result()

        batch_sbboxes = thread7.result()

        batch_mbboxes = thread8.result()

        batch_lbboxes = thread9.result()

        pool.shutdown()
        return batch_image, batch_mask, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
               batch_sbboxes, batch_mbboxes, batch_lbboxes

    def add_numpy(self,a,b):
        return ne.evaluate('a+b')


    def get_data(self,num,batch_count):
        self.train_input_size = random.choice(self.train_input_sizes)
        # self.train_input_size = self.train_input_sizes
        self.train_output_sizes = self.train_input_size // self.strides

        batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))

        batch_mask = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 1))

        batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                      self.anchor_per_scale, 5 + self.num_classes))
        batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                      self.anchor_per_scale, 5 + self.num_classes))
        batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                      self.anchor_per_scale, 5 + self.num_classes))

        batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
        batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
        batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

        num = num
        if self.batch_count < self.num_batchs:
            while num < self.batch_size:
                index = int(self.batch_count * self.batch_size + num)
                if index >= self.num_samples: index -= self.num_samples
                annotation = self.annotations[index]
                # while not os.path.exists(annotation.strip().split(',')[0]):
                #     index += 1
                #     annotation = self.annotations[index]
                image, mask, bboxes = self.parse_annotation(annotation)
                if image is None or mask is None:
                    continue
                label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
                batch_image[num, :, :, :] = image
                batch_mask[num, :, :, :] = mask[:, :, np.newaxis]
                batch_label_sbbox[num, :, :, :, :] = label_sbbox
                batch_label_mbbox[num, :, :, :, :] = label_mbbox
                batch_label_lbbox[num, :, :, :, :] = label_lbbox
                batch_sbboxes[num, :, :] = sbboxes
                batch_mbboxes[num, :, :] = mbboxes
                batch_lbboxes[num, :, :] = lbboxes
                num += 2
            self.batch_count += batch_count

            return batch_image, batch_mask, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                   batch_sbboxes, batch_mbboxes, batch_lbboxes
        else:
            self.batch_count = 0
            np.random.shuffle(self.annotations)
            raise StopIteration

    def random_horizontal_flip(self, image, mask, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            mask = mask[:, ::-1]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, mask, bboxes

    def random_crop(self, image, mask, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
            mask = mask[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin



        return image, mask, bboxes

    def random_translate(self, image, mask, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, mask, bboxes
    def parse_anno_dict(self,annotation):
        image_path = annotation['img_path']
        mask_path = annotation['msk_path']

        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        if not os.path.exists(mask_path):
            raise KeyError("%s does not exist ... " %mask_path)

        image = np.array(cv2.imread(image_path))
        mask = np.array(cv2.imread(mask_path, 0)) * 128.0

        be_correct_list = annotation['be_correct']  # 0
        to_add_list = [list(map(int, box)).append(2) for box in annotation['to_add']]# 1
        to_del = [list(map(int, box)).append(2) for box in annotation['to_del']] # 2
        if len(be_correct_list)>0:
            be_correct_list = [list(map(int, box)).append(2) for box in be_correct_list]
        total_list = []
        pass



        total_list = []








    def parse_annotation(self, annotation):

        line = annotation.split(',')
        # line = annotation.split('  ')
        image_path = line[0]
        mask_path = line[1]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        if not os.path.exists(mask_path):
            raise KeyError("%s does not exist ... " %mask_path)

        image = np.array(cv2.imread(image_path))
        mask = np.array(cv2.imread(mask_path, 0)) * 128.0

        # cv2.imshow('image_source', image)
        # cv2.imshow('mask_source', mask)
        # # cv2.waitKey(0)

        bboxes = np.array([list(map(int, box.split())) for box in line[2:-1]])
        # bboxes = np.array([list(map(int, box.split(' '))) for box in line[2:]])

        # print 'source mask value: {}'.format(np.unique(mask))
        if self.data_aug:

            image, mask, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(mask), np.copy(bboxes))
            image, mask, bboxes = self.random_crop(np.copy(image), np.copy(mask), np.copy(bboxes))
            image, mask, bboxes = self.random_translate(np.copy(image), np.copy(mask), np.copy(bboxes))

        # cv2.imshow('image', image)
        # cv2.imshow('mask', mask)
        # print 'mask value: {}'.format(np.unique(mask))

        image, mask, bboxes = utils.image_preporcess(np.copy(image), np.copy(mask), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        
        # cv2.imshow('image_preporcess', image)
        # cv2.imshow('mask_preprocess', mask / 100.)
        # print 'mask_preprocess value: {}'.format(np.unique(mask))
        # cv2.waitKey(0)

        return image, mask, bboxes

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes):

        # print [[self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale, 5 + self.num_classes] for i in range(3)]

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
        
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            
            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                # print 'label shape is: {}'.format(np.array(label).shape)
                # print best_detect, yind, xind, best_anchor
                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs




if __name__ == '__main__':
    TEST_DATASET = Dataset('train')
    tmp = tqdm(TEST_DATASET)
    for data in tmp :
        print(data)
    print('stop')