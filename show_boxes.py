# 集成攻击 general detection coco
import sys
sys.path.append('/home/huanghao/hhattack/yolov4/eval_code') 
sys.path.append('/home/huanghao/hhattack/mmdetection') 

import os
from argparse import ArgumentParser
import torch
from torchvision import transforms
from PIL import Image, ImageDraw,ImageFont
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import cv2
import copy
import numpy as np
from tqdm import tqdm
from skimage import measure
import math


from yolov4.eval_code.attack_yolov4 import inference_detector_yolov4 
from mmdetection.mmdet.apis.inference import init_detector as mmdetection_init_detector
from mmdetection.mmdet.apis.inference import faster_rcnn_inference
# from mmdetection.mmdet.apis.inference import inference_single_attack_focs as focs_attack
from yolov4.eval_code.tool.darknet2pytorch import *


def attack_imgs(root_path, imgs):
    

    ################## yolov4 init #################
    cfgfile = "yolov4/eval_code/models/yolov4.cfg"
    weightfile = "yolov4/eval_code/models/yolov4.weights"
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()
    ################################################

    ################# faster rcnn init ###############
    faster_rcnn_model = mmdetection_init_detector(config='mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', checkpoint='mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', device='cuda:0')
    ##################################################

    ################# focs init ###############
    # fcos_model = mmdetection_init_detector(config='mmdetection/configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco.py', checkpoint='mmdetection/checkpoints/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco_20200229-11f8c079.pth', device='cuda:0')
    ##################################################

    for im in imgs:
        img_path = os.path.join(root_path, im)
        original_img = None
        adversarial_degree = 255.
        noise = None
        momentum = 1.0
        min_bbox_num = 999 # 最少的检测框数量
        ori_bbox_num = None
        attack_map = None # 攻击的范围
        faster_rcnn_ori_bbox_num = None
        yolov4_ori_bbox_num = None
        fcos_ori_bbox_num = None
        max_weight = None

        faster_rcnn_weight = 0 # faster rcnn 得分权重
        yolov4_weight = 0 # yolov4 得分权重
        fcos_weight = 0 # fcos 得分权重
        img = Image.open(img_path).convert('RGB')
        size = img.size
        img_copy = copy.deepcopy(img) # 若本次结果最好，则保存本次结果
        # resize_small = transforms.Compose([
        #     transforms.Resize((608, 608)),
        # ])
        # img = resize_small(img)

        ############### 检测 ###############
        yolov4_boxes = inference_detector_yolov4(darknet_model, img) 
        faster_rcnn_boxes = faster_rcnn_inference(faster_rcnn_model, img_path) # faster rcnn攻击
        num = 0
        for i in range(len(faster_rcnn_boxes)):
            if faster_rcnn_boxes[i][-1] > 0.3:
                num += 1
        print(len(yolov4_boxes), num)
        ############### 检测 ###############
        print(yolov4_boxes)
        # print(faster_rcnn_boxes)

        draw =ImageDraw.Draw(img_copy)

        for box in yolov4_boxes:
            x1 = min(max(int((box[0] - box[2] / 2.0) * size[0]), 0), size[0]) # 原图大小不同 需要改
            y1 = min(max(int((box[1] - box[3] / 2.0) * size[1]), 0), size[1]) # 原图大小不同 需要改
            x2 = min(max(int((box[0] + box[2] / 2.0) * size[0]), 0), size[0]) # 原图大小不同 需要改
            y2 = min(max(int((box[1] + box[3] / 2.0) * size[1]), 0), size[1]) # 原图大小不同 需要改  
            draw.rectangle((x1, y1, x2, y2), outline ='red')

        for box in faster_rcnn_boxes:
            if box[-1] > 0.3:
                draw.rectangle((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        
        img_copy.save('./1.png')

def inference_by_yolo_fasterrcnn(darknet_model, faster_rcnn_model, img_path):
    original_img = None
    adversarial_degree = 255.
    noise = None
    momentum = 1.0
    min_bbox_num = 999 # 最少的检测框数量
    ori_bbox_num = None
    attack_map = None # 攻击的范围
    faster_rcnn_ori_bbox_num = None
    yolov4_ori_bbox_num = None
    fcos_ori_bbox_num = None
    max_weight = None

    faster_rcnn_weight = 0 # faster rcnn 得分权重
    yolov4_weight = 0 # yolov4 得分权重
    img = Image.open(img_path).convert('RGB')
    size = img.size
    img_copy = copy.deepcopy(img) # 若本次结果最好，则保存本次结果

    ############### 检测 ###############
    yolov4_boxes = inference_detector_yolov4(darknet_model, img) 
    faster_rcnn_boxes = faster_rcnn_inference(faster_rcnn_model, img_path) # faster rcnn攻击
    num = 0
    # print(yolov4_boxes)
    for i in range(len(faster_rcnn_boxes)):
        if faster_rcnn_boxes[i][-1] > 0.3:
            num += 1
    ############### 检测 ###############
    if len(yolov4_boxes) + num == 0:
        return True, len(yolov4_boxes), num 
    else:
        return False, len(yolov4_boxes), num 


def inference_by_yolo(darknet_model, img_path):
    original_img = None
    adversarial_degree = 255.
    noise = None
    momentum = 1.0
    min_bbox_num = 999 # 最少的检测框数量
    ori_bbox_num = None
    attack_map = None # 攻击的范围
    faster_rcnn_ori_bbox_num = None
    yolov4_ori_bbox_num = None
    fcos_ori_bbox_num = None
    max_weight = None

    faster_rcnn_weight = 0 # faster rcnn 得分权重
    yolov4_weight = 0 # yolov4 得分权重
    img = Image.open(img_path).convert('RGB')
    size = img.size
    img_copy = copy.deepcopy(img) # 若本次结果最好，则保存本次结果

    ############### 检测 ###############
    yolov4_boxes = inference_detector_yolov4(darknet_model, img) 
    print(yolov4_boxes)
    ############### 检测 ###############
    if len(yolov4_boxes) == 0:
        return True
    else:
        print('y', len(yolov4_boxes))

          


if __name__ == '__main__':
   
    root_path = './results'
    imgs = os.listdir(root_path)
    print(imgs)
    imgs = ['677.png']
    attack_imgs(root_path, imgs)