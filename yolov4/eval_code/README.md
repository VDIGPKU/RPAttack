## Data preparation

- Download 1000 pictures needed for the competition on the official [website](https://tianchi.aliyun.com/competition/entrance/531806/information)
- You can get data (`images.zip`)，and the definition, weight and evaluation code of two white box models (`eval_code.zip`). We use yolov4 and faster_rcnn as whitebox models.

  

## Requirements

This code is based on pytorch. Some basic dependencies are recorded in `requirements.txt`

- torch
- torchvision
- pillow
- numpy
- tqdm
- scipy
- scikit-image
 
You can run yolov4 now if all above requirements are satisfied.

Another faster rcnn model is implemented based on mmdetection. So, ensure that the mmdetection library has been installed and can be run on your machine. You can refer install guide of mmdetection to [github](https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md)

After installation, put the mmdetection directory into `eval_code/` below. Alternatively, it is optional that using [docker](https://github.com/open-mmlab/mmdetection/blob/master/docker/Dockerfile) provided by mmdetection.

## Usage

Unzip `eval_code.zip`，move and unzip `images.zip` to `select1000_new`, ensure the following structure:

```
|--select1000_new
    |-- XXX.png
    |-- XXX.png
    |-- XXX.png
    …
    |-- XXX.png
```

Meanwhile, keep the filename unchanged and place the patched images into `select1000_new_p` 

Finally, run `python eval.py`

## Frequent issues

```
ERROR move http://tianchi-race-upload.oss-cn- hangzhou.aliyuncs.com/result/race/231760/516/282507/1095279440490/1575518156673_images.zip? Expires=1575604621&OSSAccessKeyId=LTAIzTotdnzdypip&Signature=tqz9QjtYoPG85lIhlGz43A3VrRw%3D&response- content-disposition=attachment%3B%20 failed
```
This is caused by the illegal submission of the file. For example, all images are directly compressed and packaged without placing into `images` folder, or the folder is named incorrectly. In order to submit successfully, please make sure that all images are placed in the `images` folder, then zip the `images` directory and submit.