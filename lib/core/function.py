# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os
from pycocotools.coco import COCO

import numpy as np
import torch
import torchvision
import random
import math
import cv2
import glob as _glob
import csv

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images

import gc 
import json

logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#시드고정
torch.manual_seed(0)
if device == 'cuda':
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)
random.seed(1)

csv.register_dialect(
    'mydialect',
    delimiter = ',',
    quotechar = '"',
    doublequote = True,
    skipinitialspace = True,
    lineterminator = '\r\n',
    quoting = csv.QUOTE_MINIMAL)

def writecsv(csvname,contents):
    f = open(csvname, 'a', newline='')
    wr = csv.writer(f)
    wr.writerow(contents)
    f.close()


def glob(dir, pats, recursive=False):  # faster than match, python3 only
    pats = pats if isinstance(pats, (list, tuple)) else [pats]
    matches = []
    for pat in pats:
        matches += _glob.glob(os.path.join(dir, pat), recursive=recursive)
    return matches

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()
    model=model.to(device)

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        input=input.type(torch.FloatTensor)
        input=input.to(device)
        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        
 
        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)
        random.seed(1)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model=model.to(device)

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            input=input.to(device)

            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)
        '''
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )
        '''
        model_name = config.MODEL.NAME
        '''
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)
        '''
        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            '''
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            '''
            writer_dict['valid_global_steps'] = global_steps + 1

    return losses.avg

def test(config, val_loader, val_dataset, model,  output_dir,TTA,TTA_VALT,start):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model=model.to(device)

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    
    file_path='./datasets/sample_submission.json'
    json_data = {}
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)
    
    #### 좌표 얼마나 차이나는지 확인용 실제 test 데이터 inference시에는 주석처리  #1
    annos=glob('datasets/train/annotations','*')
    annos.sort()
    annos=annos[8000:]
    #### 좌표 얼마나 차이나는지 확인용 실제 test 데이터 inference시에는 주석처리  #1
    
    with torch.no_grad():
        
        for i, (input, target, target_weight, center,scale,score,affine,origin_img,filename) in enumerate(val_loader):
            input=input.type(torch.FloatTensor)
            
            if TTA==True:
                input=input[0].to(device)
            else:
                input=input.to(device)
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            num_images = input.size(0)
            

        

            c = center.numpy()
            s = scale.numpy()
            score = score.numpy()
            
            # 예측값 웝본 사이즈에 맞춰서 다시 scaling 처리 
            if TTA==True:
                pred,_=get_max_preds_only_preds(output.cpu().numpy())
                
            else:
                pred,_=get_max_preds_only_preds(output.cpu().numpy())
  

  
            ## 좌표 반환 및 실제 사이즈가 아닌 input 사이즈 대비 좌표 반환 
            if TTA==True:
                if TTA_VALT=='AVG':
                    pred=np.expand_dims(np.mean(pred,axis=0),axis=0)
                    points = save_batch_image_only_preds(input[0:1,:,:,:],pred*4,'test'+str(i)+'.jpg',save=False)
                    
                elif TTA_VALT=='MEDIAN':
                    points=[]
                    for pred_val in pred:
                        points.append(save_batch_image_only_preds(input[0:1,:,:,:],np.expand_dims(pred_val,axis=0)*4,'test'+str(i)+'.jpg',save=False))
                    npdata = np.array(points)
                    npdata = np.median(npdata,axis=0)
                    points = npdata
                
            else:
                points = save_batch_image_only_preds(input,pred*4,'test'+str(i)+'.jpg',save=False)
            
            
            
            #### 좌표 얼마나 차이나는지 확인용 실제 test 데이터 inference시에는 주석처리  #2
            annot=COCO(annos[i])
            
            cow_annot=np.array(annot.dataset['label_info']['annotations'][0]['keypoints'])
            cow_annot=np.reshape(cow_annot,newshape=(17,3))[:17,:2]
            
            img=cv2.imread('datasets/train/images/'+annot.dataset['label_info']['image']['file_name'], cv2.IMREAD_COLOR)
            #### 좌표 얼마나 차이나는지 확인용 실제 test 데이터 inference시에는 주석처리  #2
            
            ## 실제 training 시에는 padding을 포함한 affine translation 으로 이미지를 translation and resize를 진행하기 떄문에 마지막 output에서 
            ## 이를 보정하여 좌표를 계산해야함 
            
            ## 좌표계산
            affine=affine.numpy()
            points=np.array(points)
            
            if TTA==True:
                points[:,0]=(points[:,0]-int(affine[0][0][2])) * (origin_img[0].shape[2]/(input[0].shape[2]-int(affine[0][0][2])*2))
                points[:,1]=(points[:,1]-int(affine[0][1][2])) * (origin_img[0].shape[1]/(input[0].shape[1]-int(affine[0][1][2])*2))
            else:
                points[:,0]=(points[:,0]-int(affine[0][0][2])) * (origin_img.shape[2]/(input.shape[3]-int(affine[0][0][2])*2))
                points[:,1]=(points[:,1]-int(affine[0][1][2])) * (origin_img.shape[1]/(input.shape[2]-int(affine[0][1][2])*2))
            
            
            #### 좌표 얼마나 차이나는지 확인용 실제 test 데이터 inference시에는 주석처리  #3
            test_imgssss=input[0].permute(1,2,0).cpu().numpy()[int(affine[0][1][2]):,int(affine[0][0][2]):,:]
            #save_test(img,points,'./test_folds/CA_WU_CP_N_C_B/'+str(i)+'.jpg')
            diff=np.abs(np.sum(cow_annot-points)) #단순 좌표값 차이의 합계 계산 
            jointbad=np.zeros(shape=17)
            #야메 PCK
            #기준 소의 전체 영역의 min max x,y  좌표를 가져와서 해당 좌표의 상하좌우 10% point 만큼 안에 들어오면 ok 안되면 fail로 xx/17로 score 계산 
            #n% point 만큼 조금씩 조정하면서 해보기 
            pck, out, tor=cal_pck(cow_annot,points,0.35)
            
            ls=json_data['annotations'][i]
            for idx in range(17):
                ls['joint_self'][idx]=points[idx].tolist()
                    
            del output,input,points,pred
            gc.collect()
            #if pck<1:
            #    save_bad(img,points,cow_annot,'./test_folds/bad_pck_20210805_MIDVAL/'+filename[0].split('/')[-1])
            #    for idx in out:
            #        jointbad[idx]+=1

            writecsv('valids_flip_with_pck_joint_W_ORIGIN.csv',[filename[0],diff,pck,tor])
            #writecsv('joint_bad_W_MIDVAL.csv',jointbad.tolist())
            #### 좌표 얼마나 차이나는지 확인용 실제 test 데이터 inference시에는 주석처리  #3
    end = time.time()
    print(end-start)
    json_data['latency']=end-start
    output_path='output_test.json'
    with open(output_path, 'w') as outfile:
        json.dump(json_data, outfile)
    print('end')

def cal_pck(anno,point,ratio):
    # baseline 기준 torso는 3번과 10번의 y좌표의 차이를 기준으로 하기 떄문에 똑같이 처리함 
    tor= np.linalg.norm(np.abs((anno[3][:]-anno[10][:])))*ratio
    
    diff=np.abs(point-anno)
    out=np.concatenate([np.where(diff[:,0]>tor)[0],np.where(diff[:,1]>tor)[0]])
    out=np.unique(out) 
    pck=1
    if out.shape[0]>0:
        pck=(17-out.shape[0])/17.0
    
    return pck, out, tor 

def save_bad(img,points,annos,file_name):
    for point in points:
        print((int(point[0]), int(point[1])))
        cv2.circle(img, (int(point[0]), int(point[1])), 2, [0, 0, 255], 2)
        
    for point in annos:
        print((int(point[0]), int(point[1])))
        cv2.circle(img, (int(point[0]), int(point[1])), 2, [0, 255, 0], 2)    
    cv2.imwrite(file_name, img)
    
    
#단일 좌표들만 찍을때
def save_test(img,points,file_name):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, [255, 0, 0], 2)
    cv2.imwrite(file_name, img)
            
def save_batch_image_only_preds(batch_image, batch_joints,
                                 file_name, nrow=8, padding=2,save=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    points =[]
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]

            for joint in joints:
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                
                cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
                points.append([int(joint[0]),int(joint[1])])
            k = k + 1
    if save==True:
        cv2.imwrite(file_name, ndarr)
    return points

def get_max_preds_only_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
