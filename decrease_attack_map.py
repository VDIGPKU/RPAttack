import torch
from skimage import measure
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
import cv2
import numpy as np
import copy
sys.path.append('/home/huanghao/hhattack_icme/yolov4/eval_code') 
sys.path.append('/home/huanghao/hhattack_icme/mmdetection') 

from yolov4.eval_code.attack_yolov4 import inference_detector_yolov4 as yolov4_inference
from mmdetection.mmdet.apis.inference import init_detector as mmdetection_init_detector
from mmdetection.mmdet.apis.inference import inference_detector_faster_rcnn as faster_rcnn_inference
from mmdetection.mmdet.apis.inference import inference_single_attack_focs as focs_attack
from yolov4.eval_code.tool.darknet2pytorch import *
sys.setrecursionlimit(2000000000)



def get_map_edge(img, p):
    G = []
    mm = {}
    for (i, pp) in enumerate(p):
        mm[tuple(pp)] = i

    label = img[p[0][0]][p[0][1]]
    dis = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]]
    for pp in p:
        x = pp[0]
        y = pp[1]
        for d in dis:
            tx = x + d[0]
            ty = y + d[1]
            if mm.get(tuple([tx, ty])) is not None and img[tx][ty] == label \
                    and tx >= 0 and ty >= 0 and tx < 500 and ty < 500:
                G.append([mm[tuple([x, y])], mm[tuple([tx, ty])]])
    return G, p


class Tarjan:
    # 求无向连通图的桥
    @staticmethod
    def getCuttingPointAndCuttingEdge(edges):
        link, dfn, low = {}, {}, {}
        global_time = [0]
        for a, b in edges:
            if a not in link:
                link[a] = []
            if b not in link:
                link[b] = []
            link[a].append(b)
            link[b].append(a)
            dfn[a], dfn[b] = 0x7fffffff, 0x7fffffff
            low[a], low[b] = 0x7fffffff, 0x7fffffff

        cutting_points, cutting_edges = [], []

        def dfs(cur, prev, root):
            global_time[0] += 1
            dfn[cur], low[cur] = global_time[0], global_time[0]
            children_cnt = 0
            flag = False
            for next in link[cur]:
                if next != prev:
                    if dfn[next] == 0x7fffffff:
                        children_cnt += 1
                        dfs(next, cur, root)

                        if cur != root and low[next] >= dfn[cur]:
                            flag = True
                        low[cur] = min(low[cur], low[next])

                        if low[next] > dfn[cur]:
                            cutting_edges.append([cur, next] if cur < next else [next, cur])
                    else:
                        low[cur] = min(low[cur], dfn[next])

            if flag or (cur == root and children_cnt >= 2):
                cutting_points.append(cur)

        dfs(edges[0][0], None, edges[0][0])
        return cutting_points, cutting_edges


def criticalConnections(n, connections):
    edges = [(a, b) for a, b in connections]
    cutting_points, _ = Tarjan.getCuttingPointAndCuttingEdge(edges)
    return cutting_points


def get_available_point(img):
    labels = measure.label(img, background=0, connectivity=2)
    # print("共计%d个像素" % len(np.argwhere(labels != 0)))
    regions = measure.regionprops(labels)
    available_point = []
    print(len(regions),'连通分量')
    for region in regions:
        # 若不是单点的情况
        if len(region.coords) > 2:
            G, node = get_map_edge(labels, region.coords)
            print("共计%d个像素" % len(region.coords))
            cutting_points = criticalConnections(len(node), G)
            print("node=", len(node))
            print("cutting_points=", len(cutting_points))
            temp = [node[i] for i in range(len(node)) if i not in cutting_points]
            print("temp=", len(temp))
            available_point.extend(temp)
            # print("available_point = ", len(available_point))

        # 单点是可以随机消除的
        else:
            # print(region.coords)
            available_point.extend(region.coords)
            # print("available_point = ", len(available_point))
    available_point = np.array(available_point)
    available_point_x = available_point[:,0].reshape(1,-1)
    available_point_y = available_point[:,1].reshape(1,-1)
    available_point = np.vstack((available_point_x, available_point_y))
    available_point = tuple(available_point)
    return available_point


def get_available_point_one_patch(img):
    labels = measure.label(img, background=0, connectivity=2)
    regions = measure.regionprops(labels)
    print(len(regions))
    if len(regions) == 1:
        img_index = list(np.argwhere(img==1))
        decrease_points = random.sample(img_index, int(0.5*len(img_index)))
        img_copy = copy.deepcopy(img)
        # print(decrease_points, type(decrease_points[0]), len(decrease_points), img_copy.shape)
        
        decrease_points = np.array(decrease_points)
        decrease_points_x = decrease_points[:,0].reshape(1,-1)
        decrease_points_y = decrease_points[:,1].reshape(1,-1)
        decrease_points = np.vstack((decrease_points_x, decrease_points_y))
        decrease_points = tuple(decrease_points)

        cv2.imwrite("./img.png", img*255)   
        img_copy[decrease_points] = 0
        cv2.imwrite("./img_copy.png", img_copy*255)   
        labels = measure.label(img_copy, background=0, connectivity=2)
        regions = measure.regionprops(labels)
        if len(regions) == 1:
            print("出去的时候是一个啊")
            return decrease_points
        else:
            iter = 0.05
            while len(regions) != 1:
                decrease_points = random.sample(img_index, int((0.5-iter)*len(img_index)))
                decrease_points = np.array(decrease_points)
                print(decrease_points.shape, iter)
                decrease_points_x = decrease_points[:,0].reshape(1,-1)
                decrease_points_y = decrease_points[:,1].reshape(1,-1)
                decrease_points = np.vstack((decrease_points_x, decrease_points_y))
                decrease_points = tuple(decrease_points)

                img_copy = copy.deepcopy(img)
                img_copy[decrease_points] = 0
                labels = measure.label(img_copy, background=0, connectivity=2)
                regions = measure.regionprops(labels)
                if len(regions) == 1:
                    print("出去的时候是一个啊")
                    return decrease_points
                else:
                    iter += 0.005
    return available_point


def get_available_point_ten_patch(img, max_patch_num):
    labels = measure.label(img, background=0, connectivity=2)
    regions = measure.regionprops(labels)
    # print(len(regions))
    if len(regions) <= max_patch_num:
        img_index = list(np.argwhere(img==1))
        decrease_points = random.sample(img_index, int(0.1*len(img_index)))
        img_copy = copy.deepcopy(img)
        
        decrease_points = np.array(decrease_points)
        decrease_points_x = decrease_points[:,0].reshape(1,-1)
        decrease_points_y = decrease_points[:,1].reshape(1,-1)
        decrease_points = np.vstack((decrease_points_x, decrease_points_y))
        decrease_points = tuple(decrease_points)

        img_copy[decrease_points] = 0
        labels = measure.label(img_copy, background=0, connectivity=2)
        regions = measure.regionprops(labels)
        if len(regions) <= max_patch_num:
            return decrease_points
        else:
            iter = 1
            while len(regions) > max_patch_num:
                decrease_points = random.sample(img_index, int(100-iter))
                decrease_points = np.array(decrease_points)
                decrease_points_x = decrease_points[:,0].reshape(1,-1)
                decrease_points_y = decrease_points[:,1].reshape(1,-1)
                decrease_points = np.vstack((decrease_points_x, decrease_points_y))
                decrease_points = tuple(decrease_points)
                img_copy = copy.deepcopy(img)
                img_copy[decrease_points] = 0
                labels = measure.label(img_copy, background=0, connectivity=2)
                regions = measure.regionprops(labels)
                if len(regions) <= max_patch_num:
                    return decrease_points
                else:
                    if iter<99:
                        iter += 1
        return available_point
    
    else:
        return None


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
    # print(connected_domin_score)
    return connected_domin_score, total_area_rate, patch_number, input_map_new

def connected_domin_detect_and_score(input_img, max_total_area_rate, max_patch_number):
    # detection
    input_img_new = (torch.abs(input_img[0])+torch.abs(input_img[1])+torch.abs(input_img[2]))
    ones = torch.cuda.FloatTensor(input_img_new.size()).fill_(1)
    zeros = torch.cuda.FloatTensor(input_img_new.size()).fill_(0)

    whole_size = input_img_new.shape[0]*input_img_new.shape[1]
    input_map_new = torch.where((input_img_new != 0), ones, zeros)

    labels = measure.label(input_map_new.cpu().numpy()[:, :], background=0, connectivity=2)
    label_max_number = np.max(labels)
    # if max_patch_number > 0:
    #     if label_max_number > max_patch_number:
    #         return 0, 0, float(label_max_number), input_map_new
    if label_max_number == 0:
        return 0, 0, 0, input_map_new

    total_area = torch.sum(input_map_new).item()
    # print(total_area)
    total_area_rate = total_area / whole_size
    
    area_score = 2 - float(total_area_rate/max_total_area_rate)
    return float(area_score), float(total_area_rate), float(label_max_number), input_map_new


def random_decrease(input_map_new, img_old, ori_img, decrease_iter):
    """input_map_new 扰动分布图，img_path 当前最好结果的路径"""
    
    # img_last = Image.open(img_path).convert('RGB') # 保留上一次的结果
    img_last = img_old
    img = np.array(img_last)
    input_map_new = input_map_new.cpu().numpy()
    # attack_index = get_available_point(input_map_new)
    attack_index = list(np.argwhere(input_map_new==1))
    if decrease_iter > 500:
        recover_index = random.sample(attack_index, 1)
    elif decrease_iter > 100:
        recover_index = random.sample(attack_index, 1)
    elif decrease_iter >= 0:
        recover_index = random.sample(attack_index, 1)
    for index in recover_index:
        input_map_new[index[0], index[1]] = 0 # 0表示扰动区域之外 1表示扰动区域之内
    input_map_new_reverse = copy.deepcopy(input_map_new) 
    input_map_new_reverse[input_map_new_reverse==0] = 2
    input_map_new_reverse[input_map_new_reverse==1] = 0
    input_map_new_reverse[input_map_new_reverse==2] = 1 # 0表示扰动区域之内 1表示扰动区域之外
    input_map_new_reverse = np.stack((input_map_new_reverse, input_map_new_reverse, input_map_new_reverse),axis=-1)
    input_map_new = np.stack((input_map_new, input_map_new, input_map_new),axis=-1)
    new_img = np.array(ori_img) * input_map_new_reverse + np.array(img) * input_map_new
    return new_img, img_last


def attack_imgs(result_path, clean_path, imgs):
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

    for im in imgs:
        # try:
        img_path = os.path.join(result_path, im)
        original_img_path = os.path.join(clean_path, im)
        attack_map = None # 攻击的范围
        faster_rcnn_ori_bbox_num = None
        yolov4_ori_bbox_num = None
        fcos_ori_bbox_num = None


        img_PIL = Image.open(img_path).convert('RGB')
        img_cv2 = cv2.imread(img_path)
        ori_img = Image.open(original_img_path).convert('RGB')
        # 检测目前对检测框个数
        yolov4_boxes = yolov4_inference(darknet_model, img_PIL) 
        faster_rcnn_boxes = faster_rcnn_inference(img_path, faster_rcnn_model, img_cv2)
        # 保存初始检测框记录
        fast_rcnn_num = 0
        for i in range(len(faster_rcnn_boxes)):
            if faster_rcnn_boxes[i][-1] > 0.3:
                fast_rcnn_num += 1
        faster_rcnn_ori_bbox_num = fast_rcnn_num
        yolov4_ori_bbox_num = len(yolov4_boxes)
        # 记录初始连通阈信息
        connected_domin_score, total_area_rate, patch_number, input_map_new = count_connected_domin_score(0.02, img_PIL, ori_img, 10)
        ori_connected_domin_score = connected_domin_score
        pbar = tqdm(range(4000))
        for decrease_iter in pbar:
            new_img, img_last = random_decrease(input_map_new, img_PIL, ori_img, decrease_iter)
            new_img = Image.fromarray(np.uint8(new_img))
            img_cv2 = cv2.cvtColor(np.asarray(new_img),cv2.COLOR_RGB2BGR)
            # new_img.save(img_path)
            connected_domin_score, total_area_rate, patch_number, input_map_new = count_connected_domin_score(0.02, new_img, ori_img, 10)
            # while patch_number > 10 :
            #     # img_last.save(img_path)
            #     connected_domin_score, total_area_rate, patch_number, input_map_new = count_connected_domin_score(0.02, img_last, ori_img, 10) # 计算上一次的
            #     new_img, img_last = random_decrease(input_map_new, img_PIL, ori_img, decrease_iter)
            #     new_img = Image.fromarray(np.uint8(new_img))
            #     img_cv2 = cv2.cvtColor(np.asarray(new_img),cv2.COLOR_RGB2BGR)
            #     connected_domin_score, total_area_rate, patch_number, input_map_new = count_connected_domin_score(0.02, new_img, ori_img, 10)
            # new_img.save(img_path)

            
            yolov4_boxes = yolov4_inference(darknet_model, new_img) 
            faster_rcnn_boxes = faster_rcnn_inference(img_path, faster_rcnn_model, img_cv2)

            fast_rcnn_num = 0
            for i in range(len(faster_rcnn_boxes)):
                if faster_rcnn_boxes[i][-1] > 0.3:
                    fast_rcnn_num += 1
            
            # print('yolo', len(yolov4_boxes), yolov4_ori_bbox_num, 'faster rcnn', fast_rcnn_num, faster_rcnn_ori_bbox_num)
            if len(yolov4_boxes) <= yolov4_ori_bbox_num and fast_rcnn_num <= faster_rcnn_ori_bbox_num:
                img_PIL = new_img
                img_copy = new_img
                pbar.set_description("当前分数增长倍数 %f" % (connected_domin_score / ori_connected_domin_score))
            else:
                img_PIL = img_last
                connected_domin_score, total_area_rate, patch_number, input_map_new = count_connected_domin_score(0.02, img_PIL, ori_img, 10)
        try:
            img_copy.save(os.path.join('/home/huanghao/hhattack_icme/results/', im))
        except:
            print(im,'出现异常，从下一个继续运行')



if __name__ == '__main__':
    MAX_TOTAL_AREA_RATE = 0.02
    max_patch_number = 10
    img_path0 = './images/'
    img_path1 = '../tianchi/images/'
    imgs = os.listdir(img_path0)
    imgs.sort()
    parser = ArgumentParser()
    parser.add_argument('n', help='different task for different gpu')
    args = parser.parse_args()
    n = int(args.n)
    attack_imgs(img_path0, img_path1, imgs[125*n:125*(n+1)])
    # attack_imgs(img_path0, img_path1, ['4102.png'])


