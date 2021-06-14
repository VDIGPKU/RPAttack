from argparse import ArgumentParser
import os
import sys
import sys
sys.path.append('/home/huanghao/hhattack/yolov4/eval_code') 
sys.path.append('/home/huanghao/hhattack/mmdetection') 
from mmdet.apis import inference_detector, init_detector, show_result_pyplot, inference_attack, inference_single_attack


def attack_imgs(args, model, img_list):
    print(img_list)
    for img in img_list:
        print(img_list.index(img), os.path.join(args.img_path, img))
        result = inference_attack(model, os.path.join(args.img_path, img))[0] # 这里本来没有[0],因为改了model的返回值，加上了cls_score和bbox_pred所以才有了[0]
        # result = inference_single_attack(model, os.path.join(args.img_path, img))[0] # 这里本来没有[0],因为改了model的返回值，加上了cls_score和bbox_pred所以才有了[0]

    # build the model from a config file and a checkpoint file
    # test a single image
    
    # show the results
    # show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    
    # hh add to test if the script could run successful
    # print('inference success fully!')

def main():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument('gpu_task', help='different task for different gpus')
    parser.add_argument('img_path', help='Image file path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    model = init_detector(args.config, args.checkpoint, device=args.device)

    gpu = int(args.gpu_task)
    img_list = os.listdir(args.img_path)[125*gpu:125*(gpu+1)]
    # img_list = os.listdir(args.img_path)[200:400]
    # img_list = os.listdir(args.img_path)[400:600]
    # img_list = os.listdir(args.img_path)[600:800]
    # img_list = os.listdir(args.img_path)[800:1000]
    attack_imgs(args, model, img_list)



    
    

if __name__ == '__main__':
    main()
