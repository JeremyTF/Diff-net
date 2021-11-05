#! /usr/bin/env python
# coding=utf-8
import json

import cv2
import os
import shutil

import numpy
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



def evaluate():
    line_txt_list = []
    file_info_list = load_info_from_json('/home/wangning/Desktop/data/yizhuang/generation/total_video.json')
    for task_id in file_info_list:
        # 生成训练集数据
        if task_id not in ['MKZ156_20201215112828','MKZ073_20201216225330','HQEV503_20201215091205']:#
        # 生成验证集数据
        # if task_id not in ['MKZ156_20201215112828','MKZ073_20201216225330','HQEV503_20201215091205']:#
            task_dict = file_info_list[task_id]
            for video_id in task_dict:
                video_info = task_dict[video_id]
                for num, image_info in  enumerate(video_info):
                    img_path = image_info['img_path']
                    msk_path = image_info['msk_path']
                    be_corrected_box_list = image_info['be_correct']
                    to_add_box = image_info['to_add']
                    to_del_box = image_info['to_del']
                    line_str = img_path + ','+ msk_path
                    if len(be_corrected_box_list)> 0:
                        for box_str in be_corrected_box_list:
                            if len(box_str) > len('None'):
                                line_str  = line_str+ ','+ box_str[1:-1]+ ' ' + '0'
                    if len(to_add_box) > len('None'):
                            line_str = line_str + ',' + to_add_box[1:-1] + ' ' + '1'
                    if len(to_del_box) > len('None'):
                            line_str = line_str + ',' + to_del_box[1:-1] + ' ' + '2'
                    line_txt_list.append(line_str)

    print('stop')
    with open('/home/wangning/Desktop/data/yizhuang/generation/train_label.txt', 'w') as f:
        for line in line_txt_list:
            line = line + '\n'
            f.write(line)


# simulation_path = "/home/wangning/Desktop/data/yizhuang/generation/val_new_total.txt"
#
#
# def get_simu_dict(path):
#     simulation_path = path
#
#     simu_dict = {}
#
#     simu_class_dict = {}
#
#     with open(simulation_path, 'r') as simulation_file:
#         simulation_file = simulation_file.readlines()
#
#         simulation_file = [line.replace('wangning/Desktop/data', 'jiangshengjie/tensorflow-yolov3/tensorflow-yolov3')
#                            for
#
#                            line in simulation_file]
#
#         simulation_file = [line.replace('on', '') for line in simulation_file]
#
#         simulation_file = [line.replace('  ', ' ') for line in simulation_file]
#
#         for num, line in enumerate(simulation_file):
#             # for num, line in enumerate(annotation_file[250:400]):
#
#             # line.replace('wangning/Desktop/data', 'jiangshengjie/tensorflow-yolov3/tensorflow-yolov3')
#
#             annotation = line.strip().split(',')
#
#             image_path = annotation[0]
#
#             image_name = image_path.split('/')[-1]
#
#             # image = cv2.imread(image_path)
#
#             bbox_data_gt = np.array([list(map(int, box.split())) for box in annotation[2:] if len(box) > 5])
#
#             bboxes_gt, classes_gt = bbox_data_gt[:, :5], bbox_data_gt[:, 4]
#
#             # print('a')
#
#             num_bbox_gt = len(bboxes_gt)
#
#             simu_dict[image_path] = bboxes_gt
#
#             simu_class_dict[image_path] = classes_gt
#
#             # simu_dict.
#
#             # simu_dict.
#
#     return simu_dict, simu_class_dict


if __name__ == '__main__':
    evaluate()
    # tmp = 0
    # simu_dict,simu_class_dict = get_simu_dict(simulation_path)
    # for key in simu_dict.keys():
    #     array_boxex = simu_dict[key]
    #     shape_box = array_boxex.shape[0]
    #     tmp+=shape_box
    # print(tmp)
    #
    # print('stop')


