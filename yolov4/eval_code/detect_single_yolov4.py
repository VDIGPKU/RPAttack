import torch
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.utils import *
import json

import numpy as np
import os
from tool.darknet2pytorch import *
from tqdm import tqdm
from skimage import measure

def detection_single_yolov4(img_path):
    cfgfile = "models/yolov4.cfg"
    weightfile = "models/yolov4.weights"
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()

    bb_score_dict = {}
   
   
    img = Image.open(img_path).convert('RGB')
    resize_small = transforms.Compose([
        transforms.Resize((608, 608)),
    ])
    img = resize_small(img)

    boxes = do_detect(darknet_model, img, 0.5, 0.4, True)
    return boxes


if __name__ == '__main__':
    img_path = './images/4883.png'
    detection_single_yolov4(img_path)
