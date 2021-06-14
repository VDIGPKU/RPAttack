import torch
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.utils import *
import json
import cv2
import copy

import numpy as np
import os
from tool.darknet2pytorch import *
from tqdm import tqdm
from skimage import measure

def adjust_bbox_size(bbox, rate, ori_rate):
    # bbox [[left, top)] [right, down]], rate 缩放的比例 rate为2则是缩小两倍
    # return bbox [(left, top), (right, down)] 缩放之后的
    rate += 0.5 # 冗余，使面积之比在0.02以内
    bbox[0][0] *= ori_rate
    bbox[0][1] *= ori_rate
    bbox[1][0] *= ori_rate
    bbox[1][1] *= ori_rate
    middle = (((bbox[1][0] - bbox[0][0]) / 2.0) + bbox[0][0], 
              ((bbox[1][1] - bbox[0][1]) / 2.0) + bbox[0][1])
    k = (bbox[1][1] - bbox[0][1]) / (bbox[1][0] - bbox[0][0])
    # print(middle)
    distance = middle[0] - bbox[0][0]
    # print("原bbox:", bbox)
    if distance > rate:
        distance /= rate
        x_left = (middle[0] - distance) 
        x_right = (middle[0] + distance)
        y_left = (k * (x_left - middle[0]) + middle[1]) 
        y_right = (k * (x_right - middle[0]) + middle[1])
        # print("调整之后的bbox:", (int(x_left), int(y_left)), (int(x_right), int(y_right)))
        # print("面积改变的比例:", pow((x_right - x_left) / (bbox[1][0] - bbox[0][0]), 2))
        return [(int(x_left), int(y_left)), (int(x_right), int(y_right))]
    else:
        return -1 # bbox太小了 放弃该bbox的优化

def attack_single_yolov4(img_path):
    cfgfile = "models/yolov4.cfg"
    weightfile = "models/yolov4.weights"
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()

    bb_score_dict = {}
    
    original_img = None
    adversarial_degree = 255.
    noise = None
    momentum = 1.0
    min_bbox_num = 999 # 最少的检测框数量
    ori_bbox_num = None
    attack_map = None # 攻击的范围

    for attack_iter in range(500):
        if attack_iter != 0:
            img = im
        else:
            img = Image.open(img_path).convert('RGB')
        img_copy = copy.deepcopy(img) # 若本次结果最好，则保存本次结果
        resize_small = transforms.Compose([
            transforms.Resize((608, 608)),
        ])
        img = resize_small(img)

        if original_img is None:
            original_img = cv2.imread(img_path)
            original_img = np.array(original_img, dtype = np.int16)
            clip_min = np.clip(original_img - adversarial_degree, 0, 255)
            clip_max = np.clip(original_img + adversarial_degree, 0, 255)

        boxes, grad = do_attack(darknet_model, img_path, img, original_img, 0.5, 0.4, True)
        if attack_map is None:
            width = original_img.shape[0] # 原图大小不同 需要改
            height = original_img.shape[1] # 原图大小不同 需要改
            detection_map = np.zeros(original_img.shape[:2]) 
            for box in boxes:
                x1 = min(max(int((box[0] - box[2] / 2.0) * width), 0), 500) # 原图大小不同 需要改
                y1 = min(max(int((box[1] - box[3] / 2.0) * height), 0), 500) # 原图大小不同 需要改
                x2 = min(max(int((box[0] + box[2] / 2.0) * width), 0), 500) # 原图大小不同 需要改
                y2 = min(max(int((box[1] + box[3] / 2.0) * height), 0), 500) # 原图大小不同 需要改          
                detection_map[x1:x2, y1:y2] += 1
               
            rate = detection_map[detection_map!=0].sum() / detection_map.size # 计算检测框面积（可叠加）占据原图面积之比，比例用作下面缩小检测框
            print("检测框面积与原图面积之比：{}，需要缩小{}倍。".format(rate, math.sqrt(rate/0.02)))

            attack_map = np.zeros(original_img.shape[:2])
            attack_area_num = 0
            for box in boxes:
                x1 = min(max(int((box[0] - box[2] / 2.0) * width), 0), 500) # 原图大小不同 需要改
                y1 = min(max(int((box[1] - box[3] / 2.0) * height), 0), 500) # 原图大小不同 需要改
                x2 = min(max(int((box[0] + box[2] / 2.0) * width), 0), 500) # 原图大小不同 需要改
                y2 = min(max(int((box[1] + box[3] / 2.0) * height), 0), 500) # 原图大小不同 需要改                

                if attack_area_num >= 10:
                    break
                adjust_bbox = adjust_bbox_size([[y1, x1], [y2, x2]], math.sqrt(rate/0.02), ori_rate=1)
                # if adjust_bbox != -1:
                if 1:
                    attack_area_num += 1
                    attack_map[adjust_bbox[0][0]:adjust_bbox[1][0], adjust_bbox[0][1]:adjust_bbox[1][1]] =1
                    # attack_map[y1:y2, x1:x2] =1

            attack_rate = attack_map[attack_map==1].size / attack_map.size 
            attack_map = np.stack((attack_map, attack_map, attack_map),axis=-1)
            print("攻击区域面积与原图面积之比：{}".format(attack_rate))

        if ori_bbox_num is None:
            ori_bbox_num = len(boxes)
        if len(boxes) <= min_bbox_num:
            min_bbox_num = len(boxes) # 寻找最少的检测框
            attack_image = img_copy
        print('攻击次数', attack_iter, '最初检测框的数量：', ori_bbox_num, '当前最少的检测框数量：', min_bbox_num, '当前的检测框数量：', len(boxes))

        if noise is None:
            noise = torch.sign(grad).squeeze(0).numpy().transpose(1, 2, 0)
            noise = cv2.resize(noise, original_img.shape[:2],interpolation=cv2.INTER_CUBIC)   
        else:
            temp_noise = torch.sign(grad).squeeze(0).numpy().transpose(1, 2, 0)
            temp_noise = cv2.resize(temp_noise, original_img.shape[:2],interpolation=cv2.INTER_CUBIC)   
            noise = momentum * noise + temp_noise

        img = cv2.cvtColor(np.asarray(img_copy),cv2.COLOR_RGB2BGR)  
        img = np.clip(img + noise * attack_map, clip_min, clip_max).astype(np.uint8)
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  
        # im.save(img_path)
    attack_image.save(img_path)
        




if __name__ == '__main__':
    img_path = './images/4883.png'
    attack_single_yolov4(img_path)
