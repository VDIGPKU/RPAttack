# 集成攻击 general detection coco
import sys
sys.path.append('/home/huanghao/RPAttack/yolov4/eval_code') 
sys.path.append('/home/huanghao/RPAttack/mmdetection') 

import os
from argparse import ArgumentParser
import torch
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import cv2
import copy
import numpy as np
from tqdm import tqdm
from skimage import measure
import math


from mmdetection.mmdet.apis.inference import inference_single_attack_focs as focs_attack
from yolov4.eval_code.attack_yolov4 import inference_detector_yolov4 as yolov4_inference
from mmdetection.mmdet.apis.inference import init_detector as mmdetection_init_detector
from mmdetection.mmdet.apis.inference import inference_detector_faster_rcnn as faster_rcnn_inference
from mmdetection.mmdet.apis.inference import inference_single_attack as faster_rcnn_attack
from yolov4.eval_code.attack_yolov4 import inference_single_attack as yolov4_attack
from show_boxes import inference_by_yolo_fasterrcnn


from yolov4.eval_code.tool.darknet2pytorch import *



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


def adjust_bbox_size(bbox, rate, ori_rate):
    rate += 0.5 # 冗余，使面积之比在0.02以内
    bbox[0][0] *= ori_rate
    bbox[0][1] *= ori_rate
    bbox[1][0] *= ori_rate
    bbox[1][1] *= ori_rate
    middle = (((bbox[1][0] - bbox[0][0]) / 2.0) + bbox[0][0], 
              ((bbox[1][1] - bbox[0][1]) / 2.0) + bbox[0][1])
    k = (bbox[1][1] - bbox[0][1]) / (bbox[1][0] - bbox[0][0])
    distance = middle[0] - bbox[0][0]
    if distance > rate:
        distance /= rate
        x_left = (middle[0] - distance) 
        x_right = (middle[0] + distance)
        y_left = (k * (x_left - middle[0]) + middle[1]) 
        y_right = (k * (x_right - middle[0]) + middle[1])
      
        return [(int(x_left), int(y_left)), (int(x_right), int(y_right))]
    else:
        return -1 # bbox太小了 放弃该bbox的优化


def makeGrid(rate, width):
    """rate为比例，例如rate=2,则网格面积占图片面积的1/2,width为表格线条宽度"""
    grid = np.zeros((500,500))
    begin = 0
    while begin < 500:
        for iter in range(width):
            grid[:,begin] = 1
            grid[begin,:] = 1
            begin += 1
        begin += (rate-1)
    return grid



def py_cpu_nms(dets, thresh):  
    """Pure Python NMS baseline."""  
    y1 = dets[:, 0]  
    x1 = dets[:, 1]  
    y2 = dets[:, 2]  
    x2 = dets[:, 3]  
    scores = dets[:, 4]  #bbox打分
  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    #打分从大到小排列，取index  
    order = scores.argsort()[::-1]  
    #keep为最后保留的边框  
    keep = []  
    while order.size > 0:  
    #order[0]是当前分数最大的窗口，肯定保留  
        i = order[0]  
        keep.append(i)  
        #计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        #交/并得到iou值  
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  
        #inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收  
        inds = np.where(ovr <= thresh)[0]  
        #order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]  
  
    return keep


def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().cuda().unsqueeze(0)  # 255也可以改为256


def adjust_attack_map(attack_map, img, original_img):
    perturbation = attack_map * (img - original_img)
    perturbation_avg = np.abs(perturbation).sum()*3 / perturbation[perturbation!=np.array([0,0,0])].size
    decrease_index = np.where((np.abs(perturbation[:,:,0])+np.abs(perturbation[:,:,1])+np.abs(perturbation[:,:,2]))<perturbation_avg)
    attack_map[decrease_index] = np.array([0,0,0])
    cv2.imwrite("./perturbation.png", perturbation*100)   
    return attack_map, True



def attack_imgs(root_path, imgs, clean_path):    

    ################## yolov4 init #################
    cfgfile = "yolov4/eval_code/models/yolov4.cfg"
    weightfile = "yolov4/eval_code/models/yolov4.weights"
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()
    ################################################

    ################# faster rcnn init ###############
    faster_rcnn_model = mmdetection_init_detector(config='mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', checkpoint='/home/huanghao/hhattack/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', device='cuda:0')
    ##################################################

    for im in imgs:
        img_path = os.path.join(root_path, im)
        original_img = None
        adversarial_degree = 255.
        noise = None
        min_bbox_num = 999 # 最少的检测框数量
        ori_bbox_num = None
        attack_map = None # 攻击的范围
        faster_rcnn_ori_bbox_num = None
        yolov4_ori_bbox_num = None
        max_weight = None
        attack_image = None
        img_copy = None
        img = None
        is_decrease = False
        is_break = False
        img_cv2 = toTensor(cv2.imread(img_path))
        img_cv2.requires_grad=True
        img_cv2_800_800 = F.interpolate(img_cv2, (800, 800), mode='bilinear')
        img_cv2_608_608 = F.interpolate(img_cv2, (608, 608), mode='bilinear')
        last_state_attack_map = None
        zero_iter = 0
        
        # 寻找攻击区域 全部攻击结束 或者100次迭代之后
        pbar = tqdm(range(1000))
        for attack_iter in pbar:
            
            if attack_iter != 0:
                cv2.imwrite(os.path.join('/home/huanghao/hhattack_icme/results/', im), img_cv2)
                img_cv2 = toTensor(img_cv2)
                img_cv2.requires_grad=True
                img_cv2_800_800 = F.interpolate(img_cv2, (800, 800), mode='bilinear')
                img_cv2_608_608 = F.interpolate(img_cv2, (608, 608), mode='bilinear')
                
            faster_rcnn_weight = 0 # faster rcnn 得分权重
            yolov4_weight = 0 # yolov4 得分权重
            if img is None:
                img = Image.open(img_path).convert('RGB')
            img_copy = copy.deepcopy(img) # 若本次结果最好
           
            if original_img is None:
                original_img = cv2.imread(img_path)
                original_img = np.array(original_img, dtype = np.int16)
                clip_min = np.clip(original_img - adversarial_degree, 0, 255)
                clip_max = np.clip(original_img + adversarial_degree, 0, 255)

            
            ############### 通过yolov4确定attack map ###############
            if attack_map is None:
                connected_domin_score, total_area_rate, patch_number, input_map_new = count_connected_domin_score(0.02, Image.open(img_path).convert('RGB'), Image.open(os.path.join(clean_path, im)).convert('RGB'), 10)

                yolov4_noise, yolov4_boxes = yolov4_attack(img_path, darknet_model, img_cv2_608_608, img_cv2) 
                faster_rcnn_noise, faster_rcnn_boxes = faster_rcnn_attack(img_path, faster_rcnn_model, img_cv2_800_800, img_cv2) # faster rcnn攻击 
                width = original_img.shape[0] # 原图大小不同 需要改
                height = original_img.shape[1] # 原图大小不同 需要改
                detection_map = np.zeros(original_img.shape[:2]) 
                attack_map = np.zeros(original_img.shape[:2])
                boxes = []
                for box in faster_rcnn_boxes:
                    if box[-1] > 0.3:
                        detection_map[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1
                        boxes.append([[int(box[1]),int(box[0])], [int(box[3]),int(box[2])]])

                for box in yolov4_boxes:
                    x1 = min(max(int((box[0] - box[2] / 2.0) * height), 0), height) # 原图大小不同 需要改
                    y1 = min(max(int((box[1] - box[3] / 2.0) * width), 0), width) # 原图大小不同 需要改
                    x2 = min(max(int((box[0] + box[2] / 2.0) * height), 0), height) # 原图大小不同 需要改
                    y2 = min(max(int((box[1] + box[3] / 2.0) * width), 0), width) # 原图大小不同 需要改          
                    detection_map[x1:x2, y1:y2] = 1
                    boxes.append([[y1, x1], [y2, x2]])
                
                for box in boxes:
                    y1 = box[0][0]
                    x1 = box[0][1]
                    y2 = box[1][0]
                    x2 = box[1][1]
                    attack_map[y1:y2,x1:x2] = 1
                
                temp = copy.deepcopy(attack_map)
                temp[temp==1] = 255  
                # cv2.imwrite("./2.png", temp)      
                attack_map = np.stack((attack_map, attack_map, attack_map),axis=-1)        
                
            ############### 获取attack map结束 ###############


            ############### 检测器攻击 ###############            
            yolov4_noise, yolov4_boxes = yolov4_attack(img_path, darknet_model, img_cv2_608_608, img_cv2) 
            faster_rcnn_noise, faster_rcnn_boxes = faster_rcnn_attack(img_path, faster_rcnn_model, img_cv2_800_800, img_cv2) # faster rcnn攻击 
            num = 0
            for i in range(len(faster_rcnn_boxes)):
                if faster_rcnn_boxes[i][-1] > 0.3:
                    num += 1
            if len(yolov4_boxes)==0 and num==0:
                is_break  = inference_by_yolo_fasterrcnn(darknet_model, faster_rcnn_model, os.path.join('/home/huanghao/hhattack_icme/results/', im))
                if is_break:
                    # 缩小攻击的比例
                    last_state_attack_map = copy.deepcopy(attack_map)
                    attack_map, is_decrease = adjust_attack_map(attack_map, img, original_img)
                    zero_iter = 0
               
            zero_iter += 1
            if zero_iter >= 30 and last_state_attack_map is not None:
                # 放大攻击的比例
                attack_map = last_state_attack_map
                zero_iter = 0
            elif zero_iter >= 30 and last_state_attack_map is None:
                # 保持攻击的比例
                zero_iter = 0
            ############### 攻击结束 ###############


            ############### 交替梯度 ###############
            yolo_w = 1
            faster_w = 1
            if ori_bbox_num is not None:
                yolo_w = max((len(yolov4_boxes)- yolov4_ori_bbox_num + 1), 1)
                faster_w = max(num - faster_rcnn_ori_bbox_num + 1, 1)
            if noise is None:
                if len(yolov4_boxes)!= 0 and num != 0:
                    noise = yolo_w*yolov4_noise + faster_w*faster_rcnn_noise 
                elif num != 0:
                    noise = faster_w*faster_rcnn_noise
                elif len(yolov4_boxes) != 0:
                    noise = yolo_w*yolov4_noise
            else:
                if len(yolov4_boxes)!= 0 and num != 0:
                    noise = yolo_w*yolov4_noise + faster_w*faster_rcnn_noise 
                elif num != 0:
                    noise = faster_w*faster_rcnn_noise
                elif len(yolov4_boxes) != 0:
                    noise = yolo_w*yolov4_noise
            noise_img = np.sign(noise)
            ############### 汇总结束 ###############

            ############### 输出状态 ###############
            bbox_num = num + len(yolov4_boxes)
            if ori_bbox_num is None:
                ori_bbox_num = bbox_num
                faster_rcnn_ori_bbox_num = num # 记录最初fastercnn检测框数量
                yolov4_ori_bbox_num = len(yolov4_boxes) # 记录最初yolov4检测框数量
                
            if faster_rcnn_ori_bbox_num != 0:
                faster_rcnn_weight = max((faster_rcnn_ori_bbox_num - num) / faster_rcnn_ori_bbox_num, 0)
            else:
                faster_rcnn_weight = 1
            if yolov4_ori_bbox_num != 0:
                yolov4_weight = max((yolov4_ori_bbox_num - len(yolov4_boxes)) / yolov4_ori_bbox_num, 0)
            else:
                yolov4_weight = 1
          
            if max_weight is None:
                max_weight = faster_rcnn_weight + yolov4_weight 
            
        
            if (faster_rcnn_weight + yolov4_weight) > max_weight:
                min_bbox_num = min(num, faster_rcnn_ori_bbox_num) + min(len(yolov4_boxes), yolov4_ori_bbox_num)
                max_weight = faster_rcnn_weight + yolov4_weight
                attack_image = img_copy
            attack_rate =  attack_map[attack_map==1].size / attack_map.size
            if is_break:
                output_str = im + '当前{}/{}'.format(imgs.index(im), len(imgs)) + '次数{}'.format(attack_iter)+'最初：{}'.format(ori_bbox_num)+'当前最少：{}'.format(min_bbox_num)+'当前yolo:{}'.format(len(yolov4_boxes))+'当前faster rcnn:{}'.format(num)+"当前攻击比例:{}".format(attack_rate)+"当前inference为0"
            else:
                output_str = im + '当前{}/{}'.format(imgs.index(im), len(imgs)) + '次数{}'.format(attack_iter)+'最初：{}'.format(ori_bbox_num)+'当前最少：{}'.format(min_bbox_num)+'当前yolo:{}'.format(len(yolov4_boxes))+'当前faster rcnn:{}'.format(num)+"当前攻击比例:{}".format(attack_rate)+"当前inference不为0"
            is_break = False #  恢复
            pbar.set_description(output_str)
            ############### 输出结束 ###############
            
            ############### 保存结果 ###############
            img_last = img_cv2.cpu().detach().clone().squeeze(0).numpy().transpose(1, 2, 0)
            img_last = cv2.cvtColor(img_last, cv2.COLOR_RGB2BGR)
            if is_decrease ==True:
                # 恢复图片
                # print((img - original_img).min())
                img_last = original_img + attack_map * (img_last - original_img)
                
                is_decrease = False
            a = noise_img.astype(np.float)*attack_map
            a = a[...,::-1].copy()   
            img = np.clip(img_last + a, clip_min, clip_max).astype(np.uint8)
            img_cv2 = copy.deepcopy(img)
            img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ############### 保存结束 ###############
            
   
            


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    
    root_path = './images/'
    clean_path = './clean_data'
    imgs = os.listdir(root_path)
    imgs.sort()
    imgs = ['4996.png']
    attack_imgs(root_path, imgs, clean_path)
