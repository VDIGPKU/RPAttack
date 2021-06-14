import os.path as osp
import pickle
import shutil
import tempfile
import time
import cv2
import numpy as np

import mmcv
import torch
import torch.distributed as dist
from torch.autograd import Variable
from mmcv.runner import get_dist_info
import torchvision

from mmdet.core import encode_mask_results, tensor2imgs


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result, cls_score, bbox_pred = model(return_loss=False, rescale=True, **data)
            
        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result, tuple):
            bbox_results, mask_results = result
            encoded_mask_results = encode_mask_results(mask_results)
            result = bbox_results, encoded_mask_results
        results.append(result)

        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def disappear_loss(cls_score):
    # disappera loss for general detection cls score
    loss = 1. - torch.sum(cls_score[cls_score>0.3]) / cls_score[cls_score>0.3].numel()
    return loss


def single_gpu_attack(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        # with torch.no_grad():
        filename = data['img_metas'][0].data[0][0]['ori_filename']
        original_img = None
        noise = None
        adversarial_degree = 8
        momentum = 1.0
        for attack_iter in range(1000):
            data['img'][0] = Variable(data['img'][0], requires_grad=True)
            result, cls_score, bbox_pred = model(return_loss=False, rescale=True, **data)
            loss = disappear_loss(cls_score)
            loss.backward()
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            print(cls_score.sum(), cls_score[cls_score>0.3].sum())
            img = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])[0]

            cv2.imwrite('/home/huanghao/mmdetection/attack_results/{}.png'.format(filename.split('.')[0]), img)
            if original_img is None:
                original_img = np.array(img, dtype = np.int16)
                clip_min = np.clip(original_img - adversarial_degree, 0, 255)
                clip_max = np.clip(original_img + adversarial_degree, 0, 255)
                print('max_min', clip_max.min())
                print('min_max', clip_min.max())
            
            if noise is None:
                noise = torch.sign(data['img'][0].grad.data.cpu().detach().clone()).squeeze(0).numpy().transpose(1, 2, 0)
            else:
                noise = momentum * noise + torch.sign(data['img'][0].grad.data.cpu().detach().clone()).squeeze(0).numpy().transpose(1, 2, 0)

            # if noise is None:
            #     noise = torch.sign(data['img'][0].grad.data.cpu().detach().clone()).squeeze(0)
            #     noise = (noise / torch.norm(noise, p=1)).numpy().transpose(1, 2, 0)
            # else:
            #     temp_noise = torch.sign(data['img'][0].grad.data.cpu().detach().clone()).squeeze(0)
            #     temp_noise = (temp_noise / torch.norm(temp_noise, p=1)).numpy().transpose(1, 2, 0)
            #     noise = torch.sign(torch.from_numpy(momentum * noise + temp_noise)).numpy()

            img = np.clip(img + noise, clip_min, clip_max)
            img_metas[0]['img_norm_cfg']['to_rgb'] = False
            loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  
            img = loader(mmcv.imnormalize(img, **img_metas[0]['img_norm_cfg'])).unsqueeze(0)
            data['img'][0] = img

        
        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result, tuple):
            bbox_results, mask_results = result
            encoded_mask_results = encode_mask_results(mask_results)
            result = bbox_results, encoded_mask_results
        results.append(result)

        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return results, cls_score, bbox_pred


def single_gpu_attack_patch(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        # with torch.no_grad():
        filename = data['img_metas'][0].data[0][0]['ori_filename']
        print(filename)
        original_img = None
        noise = None
        adversarial_degree = 8
        momentum = 1.0
        for attack_iter in range(1000):
            data['img'][0] = Variable(data['img'][0], requires_grad=True)
            print(data['img'][0].shape)
            result, cls_score, bbox_pred = model(return_loss=False, rescale=True, **data)
            loss = disappear_loss(cls_score)
            loss.backward()
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            print(cls_score.sum(), cls_score[cls_score>0.3].sum())
            # cls_v = (cls_score - cls_score.min()) * 255. / (cls_score.max() - cls_score.min())
            # print(cls_score.shape)
            # cv2.imwrite('/home/huanghao/mmdetection/attack_results/1.png', )

            img = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])[0]

            
            h, w, _ = img_metas[0]['img_shape']
            # print(img_metas[0]['img_shape'])
            img_show = img[:h, :w, :]
            ori_h, ori_w = img_metas[0]['ori_shape'][:-1]
            # print(img_show.shape)
            img_show = mmcv.imresize(img_show, (ori_w, ori_h))
            # print(img_show.shape)
            bboxs, labels = model.module.show_result(
                img_show,
                result,
                show=show,
                out_file=None,
                score_thr=show_score_thr)
            
            bbox_class = {}
            for labelIndex in range(len(labels)):
                if labels[labelIndex] not in bbox_class and bboxs[labelIndex][-1]>0.3:
                    bbox_class[labels[labelIndex]] = [bboxs[labelIndex][:-1]]
                elif bboxs[labelIndex][-1]>0.3:
                    bbox_class[labels[labelIndex]].append(bboxs[labelIndex][:-1])
            print(bbox_class)
      
            attack_m = np.zeros((img_show.shape[1], img_show.shape[0]))
            # print(attack_m.shape)
            for label in bbox_class:
                attack_label_m = np.zeros((img_show.shape[1], img_show.shape[0]))
                for bbox in bbox_class[0]:
                    attack_label_m[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])] += 1
                print(label, attack_label_m.min(), attack_label_m.max())
                attack_label_m[attack_label_m!=attack_label_m.max()] = 0
                attack_label_m[attack_label_m==attack_label_m.max()] = 1
                print(label, attack_label_m.min(), attack_label_m.max())
                attack_m += attack_label_m
            attack_m[attack_m>=1] = 1
            print('attack rate', attack_m[attack_m==1].sum()/np.size(attack_m))
            cv2.imwrite('/home/huanghao/mmdetection/attack_results/3.png', attack_m)



            cv2.imwrite('/home/huanghao/mmdetection/attack_results/{}.png'.format(filename.split('.')[0]), img)
            if original_img is None:
                original_img = np.array(img, dtype = np.int16)
                clip_min = np.clip(original_img - adversarial_degree, 0, 255)
                clip_max = np.clip(original_img + adversarial_degree, 0, 255)
                print('max_min', clip_max.min())
                print('min_max', clip_min.max())
            if noise is None:
                noise = torch.sign(data['img'][0].grad.data.cpu().detach().clone()).squeeze(0).numpy().transpose(1, 2, 0)
            else:
                noise = momentum * noise + torch.sign(data['img'][0].grad.data.cpu().detach().clone()).squeeze(0).numpy().transpose(1, 2, 0)
            img = np.clip(img + noise, clip_min, clip_max)
            img_metas[0]['img_norm_cfg']['to_rgb'] = False
            loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  
            img = loader(mmcv.imnormalize(img, **img_metas[0]['img_norm_cfg'])).unsqueeze(0)
            data['img'][0] = img

        
        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                bboxs, labels = model.module.show_result(
                    img_show,
                    result,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result, tuple):
            bbox_results, mask_results = result
            encoded_mask_results = encode_mask_results(mask_results)
            result = bbox_results, encoded_mask_results
        results.append(result)

        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return results, cls_score, bbox_pred

def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        
            # encode mask results
            if isinstance(result, tuple):
                bbox_results, mask_results = result
                encoded_mask_results = encode_mask_results(mask_results)
                result = bbox_results, encoded_mask_results
        results.append(result)

        if rank == 0:
            batch_size = (
                len(data['img_meta'].data)
                if 'img_meta' in data else len(data['img_metas'][0].data))
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
