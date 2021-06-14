import warnings

import copy
import matplotlib.pyplot as plt
import mmcv
import torch
import cv2
import numpy as np
import math
import torchvision
import torch
from PIL import Image  
from torch.autograd import Variable
from mmcv.ops import RoIAlign, RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes, encode_mask_results, tensor2imgs
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector



def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def imnormalize_(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    # print(img)
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img
    

class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        # if isinstance(results['img'], str):
        #     results['filename'] = results['img']
        #     results['ori_filename'] = results['img']
        # else:
        #     results['filename'] = None
        #     results['ori_filename'] = None
        # try:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])]) 
            # torchvision.transforms.Normalize(mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395])]) 

        img = transform(results['img'].squeeze(0)).unsqueeze(0)
        # except:
        #     img = mmcv.imread(results['img'])
        results['img'] = [img]
        # results['img_fields'] = ['img']
        # try:
            # results['img_shape'] = img.shape
        # except:
        #     print(results['filename'] )
        # results['ori_shape'] = img.shape
        return results

class LoadImage_original(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        try:
            img = results['img_cv2']
        except:
            img = mmcv.imread(results['img'])
        results['img'] = img
        
        results['img_fields'] = ['img']
        try:
            results['img_shape'] = img.shape
        except:
            print(results['filename'] )
        results['ori_shape'] = img.shape
        
        return results

def focs_disappear_loss(scores):
    """score 是一个tensor (bbox num, class+1(背景)), 我们的目的是将背景概率最大化，类别概率最小化"""
    # scores_target = copy.deepcopy(scores)
    # print(scores.shape)
    # scores = scores[scores[:,-1]>0.03]
    # print(scores.shape)
    scores_target = torch.zeros(scores.shape).cuda()
    scores_target[:,:-1] = 0
    scores_target[:,-1] = 1
    bceloss =torch.nn.BCELoss()
    loss = 0
    for i in range(scores.shape[0]):
        if scores[i].max() >= 0.03:
            loss += (1. - bceloss(scores[i].unsqueeze(0), scores_target[i].unsqueeze(0)))
    return loss

    
def inference_single_attack_focs(img, model):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    data['img'][0] = Variable(data['img'][0], requires_grad=True)

    results, mlvl_scores = model(return_loss=False, rescale=True, **data) # 推理
    
    bboxes, labels = get_bbox_and_label(model, img, results) # 获取bbox
    
    # disappear loss
    loss = focs_disappear_loss(mlvl_scores)
    loss.backward()
    img_metas = data['img_metas'][0]
    # noise = torch.sign(data['img'][0].grad.data.cpu().detach().clone()).squeeze(0).numpy().transpose(1, 2, 0)
    noise = data['img'][0].grad.data.cpu().detach().clone().squeeze(0)
    noise = noise / torch.norm(noise,p=1)
    noise = noise.numpy().transpose(1, 2, 0)
    noise = cv2.resize(noise,img_metas[0]['ori_shape'][:2],interpolation=cv2.INTER_CUBIC)   
    return noise, bboxes


def inference_single_attack(img, model):
    """集成攻击接口，输入模型和路径，返回原图大小的grad"""
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    model.eval()
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    data['img'][0] = Variable(data['img'][0], requires_grad=True)
    results, cls_score, bbox_pred, det_bboxes, det_labels,= model(return_loss=False, rescale=True, **data) # 推理
    bbox_result, segm_result = results, None
    bboxes, labels = get_bbox_and_label(model, img, results) # 获取bbox
    
    # disappear loss
    loss = disappear_loss(det_bboxes[:,-1])
    loss.backward()
    img_metas = data['img_metas'][0]
    # noise = torch.sign(data['img'][0].grad.data.cpu().detach().clone()).squeeze(0).numpy().transpose(1, 2, 0)
    noise = data['img'][0].grad.data.cpu().detach().clone().squeeze(0)
    noise = noise / torch.norm(noise,p=1)
    noise = noise.numpy().transpose(1, 2, 0)
    noise = cv2.resize(noise,img_metas[0]['ori_shape'][:2],interpolation=cv2.INTER_CUBIC)   


    # img_metas[0]['img_norm_cfg']['to_rgb'] = False
    # noise = mmcv.imnormalize(noise, **img_metas[0]['img_norm_cfg'])
    return noise, bboxes


def inference_detector_faster_rcnn(img, model, img_cv2):
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage_original()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img, img_cv2=img_cv2)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        results, cls_score, bbox_pred, det_bboxes, det_labels,= model(return_loss=False, rescale=True, **data) # 推理
    bbox_result, segm_result = results, None
    bboxes, labels = get_bbox_and_label(model, img, results) # 获取bbox
    return bboxes


def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage_original()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result

def faster_rcnn_inference(model, img):
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage_original()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        # result = model(return_loss=False, rescale=True, **data)
        results, cls_score, bbox_pred, det_bboxes, det_labels,= model(return_loss=False, rescale=True, **data) # 推理
    bboxes, labels = get_bbox_and_label(model, img, results) # 获取bbox
    return bboxes


def disappear_loss(cls_score):
    # disappera loss for general detection cls score
    mseloss = torch.nn.MSELoss()
    loss = 1. - mseloss(cls_score[cls_score>=0.3], torch.zeros(cls_score[cls_score>=0.3].shape).cuda())
    # loss = 1. - torch.sum(cls_score[cls_score>=0.3]) / torch.numel(cls_score)
    # loss = 1. - torch.nn.BCELoss()(cls_score, torch.zeros(cls_score.shape).cuda())
    return loss

def adjust_bbox_size(bbox, rate, ori_rate):
    # bbox [[left, top], [right, down]], rate 缩放的比例 rate为2则是缩小两倍
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

    
def inference_single_attack(img, model, img_cv2_800_800, img_cv2):
    """faster rcnn集成攻击接口，输入模型和路径，返回原图大小的grad"""
    model.zero_grad()
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    # test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = [LoadImage()]
    test_pipeline = Compose(test_pipeline)
    # prepare data

    results = {}
    results['img_shape'] = np.array([1,3,800,800])
    # in case that there is no padding
    results['pad_shape'] = np.array([1,3,800,800])
    scale_factor = np.array([1.6, 1.6, 1.6, 1.6],
                                    dtype=np.float32)
    results['scale_factor'] = scale_factor
    results['keep_ratio'] = True

    data = dict(img=img_cv2_800_800, img_metas=[[{'filename': './images/106.png', 'ori_filename': './images/106.png', 'ori_shape': (500, 500, 3), 'img_shape': (800, 800, 3), 'pad_shape': (800, 800, 3), 'scale_factor': np.array([1.6, 1.6, 1.6, 1.6], dtype=np.float32), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53 ], dtype=np.float32), 'std': np.array([58.395, 57.12 , 57.375], dtype=np.float32), 'to_rgb': True}}]])
    data = test_pipeline(data)
    model.eval().cuda()
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data
    # print(data['img'][0])
    # data['img'][0] = Variable(data['img'][0], requires_grad=True)
    # data['img'][0] = data['img'][0]
    results, cls_score, bbox_pred, det_bboxes, det_labels,= model(return_loss=False, rescale=True, **data) # 推理
    bbox_result, segm_result = results, None
    bboxes, labels = get_bbox_and_label(model, img, results) # 获取bbox
    # disappear loss
    if len(det_bboxes) == 0:
        loss = disappear_loss(cls_score[:,-1])
    else:
        loss = disappear_loss(det_bboxes[:,-1])
    model.zero_grad()
    loss.backward()
    
    # for name, parms in model.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad,' -->grad_value:', parms.grad)
    img_metas = data['img_metas'][0]
    noise = img_cv2.grad.data.cpu().detach().clone().squeeze(0)
    # print('f',loss,img_cv2.grad)
    # print(loss, noise.sum())
    noise = (noise / torch.norm(noise,p=1)).numpy().transpose(1, 2, 0)
    return noise, bboxes,


def inference_attack(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    original_img = None
    attack_map = None
    original_img_copy = img # 保存图片的路径，用来获取bboxs和labels，
    noise = None
    adversarial_degree = 255
    momentum = 1.0
    attack_image = None
    min_bbox_num = 999
    filename = None
    ori_bbox_num = -1 # 最初的检测框数量
    for attack_iter in range(500):
        
        # 数据和模型准备
        device = next(model.parameters()).device  # model device
        # build the data pipeline
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        # prepare data
        data = dict(img=img)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            # Use torchvision ops for CPU mode instead
            for m in model.modules():
                if isinstance(m, (RoIPool, RoIAlign)):
                    if not m.aligned:
                        # aligned=False is not implemented on CPU
                        # set use_torchvision on-the-fly
                        m.use_torchvision = True
            warnings.warn('We set use_torchvision=True in CPU mode.')
            # just get the actual data from DataContainer
            data['img_metas'] = data['img_metas'][0].data

        # forward the model
        # with torch.no_grad():
        if filename is None:
            filename = data['img_metas'][0][0]['filename']
            
        model.eval()
        data['img'][0] = Variable(data['img'][0], requires_grad=True)
        result, cls_score, bbox_pred = model(return_loss=False, rescale=True, **data) # 推理
        print(results)
        bboxes, labels = get_bbox_and_label(model, original_img_copy, result) # 获取bbox
        
        bbox_num = 0
        for i in range(len(labels)):
            if bboxes[i][-1] > 0.3:
                bbox_num += 1
        if ori_bbox_num == -1:
            ori_bbox_num = bbox_num
        print(filename, "原图检测框的数目:", ori_bbox_num, ",当前最少的检测框是", min_bbox_num)
        loss = disappear_loss(cls_score)
        loss.backward()
        img_metas = data['img_metas'][0]

        # 保存结果
        if bbox_num <= min_bbox_num:
            min_bbox_num = bbox_num # 寻找最少的检测框
            attack_image = img
        

        if attack_map is None:
            detection_map = np.zeros(img_metas[0]['ori_shape'][:2]) 
            label_bbox_map = {}
            for i in range(len(labels)):
            # for i in range(1,2):
                if bboxes[i][-1] > 0.3:
                    detection_map[int(bboxes[i][1]):int(bboxes[i][3]), int(bboxes[i][0]):int(bboxes[i][2])] += 1
                    if labels[i] not in label_bbox_map:
                        label_bbox_map[labels[i]] = [[[int(bboxes[i][1]),int(bboxes[i][0])], [int(bboxes[i][3]),int(bboxes[i][2])]]]
                    else:
                        label_bbox_map[labels[i]].append([[int(bboxes[i][1]),int(bboxes[i][0])], [int(bboxes[i][3]),int(bboxes[i][2])]])
            rate = detection_map[detection_map!=0].sum() / detection_map.size # 计算检测框面积（可叠加）占据原图面积之比，比例用作下面缩小检测框
            print("检测框面积与原图面积之比：{}，需要缩小{}倍。".format(rate, math.sqrt(rate/0.02)))

            attack_map = np.zeros(img_metas[0]['ori_shape'][:2])
            attack_area_num = 0
            for label in label_bbox_map:
                for bbox in label_bbox_map[label]:
                    if attack_area_num >= 10:
                        break
                    adjust_bbox = adjust_bbox_size(bbox, math.sqrt(rate/0.02), ori_rate=1)
                    if adjust_bbox != -1:
                        attack_area_num += 1
                        attack_map[adjust_bbox[0][0]:adjust_bbox[1][0], adjust_bbox[0][1]:adjust_bbox[1][1]] =1
            attack_rate = attack_map[attack_map==1].size / attack_map.size 
            attack_map = np.stack((attack_map, attack_map, attack_map),axis=-1)
            print("攻击区域面积与原图面积之比：{}".format(attack_rate))
            cv2.imwrite('/home/huanghao/hhattack/mmdetection/attack_results/5.png', detection_map)
        
        if original_img is None:
            print('111')
            img = cv2.imread(data['img_metas'][0][0]['filename'])
            original_img = np.array(img, dtype = np.int16)
            clip_min = np.clip(original_img - adversarial_degree, 0, 255)
            clip_max = np.clip(original_img + adversarial_degree, 0, 255)
       
        # 扰动
        if noise is None:
            noise = torch.sign(data['img'][0].grad.data.cpu().detach().clone()).squeeze(0).numpy().transpose(1, 2, 0)
            noise = cv2.resize(noise,img_metas[0]['ori_shape'][:2],interpolation=cv2.INTER_CUBIC)   
        else:
            temp_noise = torch.sign(data['img'][0].grad.data.cpu().detach().clone()).squeeze(0).numpy().transpose(1, 2, 0)
            temp_noise = cv2.resize(temp_noise, img_metas[0]['ori_shape'][:2],interpolation=cv2.INTER_CUBIC)   
            noise = momentum * noise + temp_noise
           
        noise = cv2.resize(noise,img_metas[0]['ori_shape'][:2],interpolation=cv2.INTER_CUBIC) 
        
        

        img = np.clip(img + noise * attack_map, clip_min, clip_max).astype(np.uint8)
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  
        im.save(filename)
   

        
        # img_metas[0]['img_norm_cfg']['to_rgb'] = False
        # loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  
        # img = loader(mmcv.imnormalize(img, **img_metas[0]['img_norm_cfg'])).unsqueeze(0)
        # data['img'][0] = img.cuda()
    attack_image = Image.fromarray(cv2.cvtColor(attack_image, cv2.COLOR_BGR2RGB))  
    attack_image.save(filename)
    return result


async def async_inference_detector(model, img):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        Awaitable detection results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    result = await model.aforward_test(rescale=True, **data)
    return result


def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img, bboxes, labels = model.show_result(img, result, score_thr=score_thr, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.imsave('/home/huanghao/hhattack/mmdetection/attack_results/2.png', mmcv.bgr2rgb(img))
    plt.close()
    # plt.show()

def get_bbox_and_label(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img, bboxes, labels = model.show_result(img, result, score_thr=score_thr, show=False)
    return bboxes, labels