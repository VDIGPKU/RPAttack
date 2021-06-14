from argparse import ArgumentParser
import os
import sys

from mmdet.apis import inference_detector, init_detector, show_result_pyplot, inference_attack


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    # parser.add_argument('gpu_task', help='different task for different gpus')
    # parser.add_argument('img_path', help='Image file path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    model = init_detector(args.config, args.checkpoint, device=args.device)
    result = inference_single_attack(model, args.img)[0] # 这里本来没有[0],因为改了model的返回值，加上了cls_score和bbox_pred所以才有了[0]
    

if __name__ == '__main__':
    main()
