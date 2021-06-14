from argparse import ArgumentParser
import sys
sys.path.append('/home/huanghao/hhattack/yolov4/eval_code') 
sys.path.append('/home/huanghao/hhattack/mmdetection') 
from mmdet.apis import inference_detector_focs, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector_focs(model, args.img) # 这里本来没有[0],因为改了model的返回值，加上了cls_score和bbox_pred所以才有了[0]
    # result = inference_detector(model, args.img)[0] # 这里本来没有[0],因为改了model的返回值，加上了cls_score和bbox_pred所以才有了[0]

    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    
    # hh add to test if the script could run successful
    # print('inference success fully!')
    

if __name__ == '__main__':
    main()
