import time

import math
import matplotlib.pyplot as plt
from skimage import measure

import torch
from tqdm import tqdm
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import copy
from tqdm import tqdm
from argparse import ArgumentParser
import sys
sys.path.append('/home/huanghao/hhattack/yolov4/eval_code') 
sys.path.append('/home/huanghao/hhattack/mmdetection') 

from yolov4.eval_code.attack_yolov4 import inference_detector_yolov4 as yolov4_inference
from mmdetection.mmdet.apis.inference import init_detector as mmdetection_init_detector
from mmdetection.mmdet.apis.inference import inference_detector_faster_rcnn as faster_rcnn_inference
from mmdetection.mmdet.apis.inference import inference_single_attack_focs as focs_attack
from yolov4.eval_code.tool.darknet2pytorch import *


def Prim(p):
    n = len(p)
    G = np.zeros((n, n))
    node = []  # 记录边集

    # 邻接矩阵
    for i in range(n):
        for j in range(i, n):
            G[i][j] = G[j][i] = math.sqrt((p[i][0] - p[j][0]) ** 2 + (p[i][1] - p[j][1]) ** 2)

    inf = 0x3f3f3f3f
    lowcost = np.ones(n) * inf
    vis = np.zeros(n)
    sum = 0
    for j in range(n):
        lowcost[j] = G[0][j]
    k = 0
    vis[0] = 1
    node.append(0)
    for i in range(1, n):
        mmin = inf
        for j in range(0, n):
            if mmin > lowcost[j]:
                if vis[j] == 0:
                    k = j
                    mmin = lowcost[j]
        vis[k] = 1
        sum += lowcost[k]
        node.append(k)
        for j in range(0, n):
            if G[k][j] < lowcost[j] and vis[j] == 0:
                lowcost[j] = G[k][j]

    return node, sum


def count_connected_domin_score(max_total_area_rate, img0, img1, max_patch_number):

    resize2 = transforms.Compose([
        transforms.ToTensor()])
    connected_domin_score_dict = {}
    # img0 = Image.open(img_path0).convert('RGB')
    # img1 = Image.open(img_path1).convert('RGB')
    img0_t = resize2(img0).cuda()
    img1_t = resize2(img1).cuda()
    img_minus_t = img0_t - img1_t

    connected_domin_score, total_area_rate, patch_number, input_map_new = \
        connected_domin_detect_and_score(img_minus_t, max_total_area_rate, max_patch_number)
    return connected_domin_score, total_area_rate, patch_number, input_map_new


def connected_domin_detect_and_score(input_img, max_total_area_rate, max_patch_number):
    # detection
    input_img_new = (input_img[0]+input_img[1]+input_img[2])
    ones = torch.cuda.FloatTensor(input_img_new.size()).fill_(1)
    zeros = torch.cuda.FloatTensor(input_img_new.size()).fill_(0)

    whole_size = input_img_new.shape[0]*input_img_new.shape[1]
    input_map_new = torch.where((input_img_new != 0), ones, zeros)

    labels = measure.label(input_map_new.cpu().numpy()[:, :], background=0, connectivity=2)
    label_max_number = np.max(labels)
    if max_patch_number > 0:
        if label_max_number > max_patch_number:
            return 0, 0, float(label_max_number), input_map_new
    if label_max_number == 0:
        return 0, 0, 0, input_map_new

    total_area = torch.sum(input_map_new).item()
    total_area_rate = total_area / whole_size
    
    area_score = 2 - float(total_area_rate/max_total_area_rate)
    return float(area_score), float(total_area_rate), float(label_max_number), input_map_new


if __name__ == '__main__':
    img_path0 = './images/15.png'
    img_path1 = '../tianchi/images/15.png'
    img = Image.open(img_path0).convert('RGB')
    ori_img = Image.open(img_path1).convert('RGB')
    connected_domin_score, total_area_rate, patch_number, input_map_new = count_connected_domin_score(0.02, img, ori_img, 10)
    labels = measure.label(input_map_new.cpu().numpy(), background=0, connectivity=2)
    print(total_area_rate)
    print(input_map_new.cpu().numpy().sum())
    con_max_num = labels.max()
    p = []
    for i in range(1, con_max_num + 1):
        tp = np.argwhere(labels == i)
        cnt = np.random.randint(len(tp))
        p.append(tp[cnt])

    node, sum = Prim(p)
    print(node, sum)
    

   