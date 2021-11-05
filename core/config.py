#! /usr/bin/env python
# coding=utf-8

from easydict import EasyDict as edict


__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.CLASSES                = "/home/wangning/Desktop/code/Hdmap/Diff-Net/diff_net/tlight_diff.names"
__C.YOLO.ANCHORS                = "/home/wangning/Desktop/code/Hdmap/Diff-Net/diff_net/basline_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.5
__C.YOLO.UPSAMPLE_METHOD        = "resize"
__C.YOLO.ORIGINAL_WEIGHT        = "/home/wangning/Desktop/code/Hdmap/Diff-Net/diff_net/checkpointloss=6.4202.ckpt-29"
# __C.YOLO.ORIGINAL_WEIGHT        = ""
__C.YOLO.DEMO_WEIGHT            = "/home/wangning/Desktop/code/Hdmap/Diff-Net/diff_net/checkpointloss=6.4202.ckpt-29"
# __C.YOLO.DEMO_WEIGHT            = ""

# Train options
__C.TRAIN                       = edict()

# __C.TRAIN.ANNOT_PATH            = "/home/wangning/Desktop/data/yizhuang/generation/train_label.txt"
__C.TRAIN.ANNOT_PATH            = "/home/wangning/Desktop/code/Hdmap/Diff-Net/diff_net/dataset/total_2d_labels_train.txt"
# __C.TRAIN.ANNOT_PATH            = "../data/codes/train.txt"
__C.TRAIN.BATCH_SIZE            = 4
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE            = [608]
__C.TRAIN.DATA_AUG              = False
__C.TRAIN.LEARN_RATE_INIT       = 1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-6
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 20
__C.TRAIN.SECOND_STAGE_EPOCHS   = 40
# __C.TRAIN.INITIAL_WEIGHT        ="/home/wangning/Desktop/code/Hdmap/Diff-Net/diff_net/ckpts/checkpointloss=6.4202.ckpt-29"
__C.TRAIN.INITIAL_WEIGHT        = "./checkpoint/yolov3_coco_demo.ckpt"



# TEST options
__C.TEST                        = edict()

__C.TEST.ANNOT_PATH             = "/home/wangning/Desktop/data/yizhuang/train_label_version2.txt"
__C.TEST.BATCH_SIZE             = 1
__C.TEST.INPUT_SIZE             = 608
__C.TEST.DATA_AUG               = True
__C.TEST.WRITE_IMAGE            = False
__C.TEST.WRITE_IMAGE_PATH       = "/home/wangning/Desktop/code/Hdmap/Diff-Net/diff_net/data/detection"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = False
__C.TEST.WEIGHT_FILE            = "/home/wangning/Desktop/code/Hdmap/Diff-Net/diff_net/ckpts/checkpointloss=6.4202.ckpt-29"
# __C.TEST.WEIGHT_FILE            = "/home/wangning/Desktop/code/Hdmap/Diff-Net/diff_net/ckpts/checkpointloss=6.4202.ckpt-29":total_recall    :96.93%(37020/38194), recall_be_correct    :98.25%(21330/21710), recall_to_add    :97.48%(8371/8587),   recall_to_del   :92.68%(7319/7897)
# total_precision :90.06%(37020/41107), precision_be_correct :97.83%(21330/21804), precision_to_add :78.60%(8371/10650),  precision_to_del :84.58%(7319/8653)
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.3
__C.TEST.IOU_THRESHOLD          = 0.3






