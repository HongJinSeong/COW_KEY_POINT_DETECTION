# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

import gc 

logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train,TTAUG, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []
        
        #test time augmentation 적용 여부 추가 
        self.TTAUG=TTAUG

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)
    
    def paste_cutout(self,img, ref_img, max_num_holes=8, min_h_size=30, max_h_size=32, min_w_size=30, max_w_size=32):    
        num_of_holes = random.randint(0, max_num_holes)    
        if not num_of_holes:
            return img    

        min_height = min(img.shape[0], ref_img.shape[0])
        min_width = min(img.shape[1], ref_img.shape[1])
        
        for i in range(num_of_holes):        
            hole_height = random.randint(min_h_size, max_h_size)
            hole_width = random.randint(min_w_size, max_w_size)
            y1, x1 = random.randint(0, min_height - hole_height), random.randint(0, min_width - hole_width)
            y2, x2 = y1 + hole_height, x1 + hole_width        
            img[ y1:y2, x1:x2,:] = ref_img[ y1:y2, x1:x2,:]
        return img
    
    #random noise
    def noisy(self,image):
        # 분포값을 랜덤하게 해주기 위함 
        randval = random.uniform(0.1, 0.5)
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**randval
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + gauss        
        result = np.clip(noisy.astype(np.uint8),0,255)
        return result
    
    #random contrast 
    def contrast(self,image):
        randval = random.uniform(0.7,1.3)
        new_image = cv2.convertScaleAbs(image,alpha=randval)
        return new_image
        
    #random brightness
    def brightness(self,image):
        randval = random.randrange(-50,50)
        new_image = cv2.convertScaleAbs(image,beta=randval)
        return new_image
    
    
    #TTA constast
    def contrast_brightness(self,image,al,be):
        new_image = cv2.convertScaleAbs(image,alpha=al,beta=be)
        return new_image
    
 
    
    #TTA용 
    def paste_cutout_TTA(self,image,ratio):    
        hh=int(image.shape[0]/2)
        hw=int(image.shape[1]/2)
        img=copy.deepcopy(image)
        img=np.expand_dims(img,axis=0)
        
        vacant_ratio=(1-ratio)*0.5 #여백비율
        
        #각 채널 별 평균값 취득 (이미지에 채워줄 값 취득한 것임)
        avgval=np.average(image,axis=(0,1))
        insert_img=np.ones(shape=(int(hh*ratio),int(hw*ratio),3))
        
        for i in range(3):
            insert_img[:,:,i]=insert_img[:,:,i]*avgval[i]
        
        ## 상하좌우 4방향으로 augmentation
        img1=copy.deepcopy(image)
        img1[int(hh*vacant_ratio):int(hh*vacant_ratio)+insert_img.shape[0],int(hw*vacant_ratio):int(hw*vacant_ratio)+insert_img.shape[1]]=insert_img
        img=np.concatenate([img,np.expand_dims(img1,axis=0)],axis=0)

        img2=copy.deepcopy(image)
        img2[int(hh*vacant_ratio)+hh:int(hh*vacant_ratio)+hh+insert_img.shape[0],int(hw*vacant_ratio):int(hw*vacant_ratio)+insert_img.shape[1]]=insert_img
        img=np.concatenate([img,np.expand_dims(img2,axis=0)],axis=0)
  
        img3=copy.deepcopy(image)
        img3[int(hh*vacant_ratio)+hh:int(hh*vacant_ratio)+hh+insert_img.shape[0],int(hw*vacant_ratio)+hw:int(hw*vacant_ratio)+hw+insert_img.shape[1]]=insert_img
        img=np.concatenate([img,np.expand_dims(img3,axis=0)],axis=0)
   
        img4=copy.deepcopy(image)
        img4[int(hh*vacant_ratio):int(hh*vacant_ratio)+insert_img.shape[0],int(hw*vacant_ratio)+hw:int(hw*vacant_ratio)+hw+insert_img.shape[1]]=insert_img
        img=np.concatenate([img,np.expand_dims(img4,axis=0)],axis=0)
        
        #총 이미지 5개에 대해서 밝기와 대비 고정 augmentation
        # 밝기 +25,-25
        # 대비 0.8,1.2
        
        img_output=copy.deepcopy(img)
        
        #밝기만
        img_output=np.concatenate([img_output,np.expand_dims(self.contrast_brightness(img[0],al=0,be=25),axis=0)],axis=0)
        img_output=np.concatenate([img_output,np.expand_dims(self.contrast_brightness(img[0],al=0,be=-25),axis=0)],axis=0)
        #대비만
        img_output=np.concatenate([img_output,np.expand_dims(self.contrast_brightness(img[0],al=0.8,be=0),axis=0)],axis=0)
        img_output=np.concatenate([img_output,np.expand_dims(self.contrast_brightness(img[0],al=1.2,be=0),axis=0)],axis=0)
        #밝기 UP 대비 UP
        img_output=np.concatenate([img_output,np.expand_dims(self.contrast_brightness(img[0],al=1.2,be=25),axis=0)],axis=0)
        #밝기 UP 대비 DOWN
        img_output=np.concatenate([img_output,np.expand_dims(self.contrast_brightness(img[0],al=0.8,be=25),axis=0)],axis=0)
        #밝기 DOWN 대비 UP
        img_output=np.concatenate([img_output,np.expand_dims(self.contrast_brightness(img[0],al=1.2,be=-25),axis=0)],axis=0)
        #밝기 DOWN 대비 DOWN
        img_output=np.concatenate([img_output,np.expand_dims(self.contrast_brightness(img[0],al=0.8,be=-25),axis=0)],axis=0)
        
        del img,img1,img2,img3,img4
        
        gc.collect()

        return img_output
    

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            #상하체 좌표값 가져와서 min max box 영역 가져오는 거
            '''
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body
            '''
            #2021-07-31 추가 (cut and paste 승백이 함수 적용)
            random.seed(idx)   #시드 변경안해주면 같은순서로 이미지 가져와서 작업함 
            aug_idx=random.randint(0, self.__len__())

            aug_img_nm=self.db[aug_idx]['image']
            aug_img = cv2.imread(
                aug_img_nm, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            if self.color_rgb:
                aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                
            # random cut & paste 적용
            if random.random() <= 0.5:    
                data_numpy=self.paste_cutout(data_numpy,aug_img)
            #2021-08-02  augmentation 추가적용
            # random noise 적용
            if  random.random() <= 0.5:
                data_numpy=self.noisy(data_numpy)
            #    print(np.max(data_numpy),np.min(data_numpy))
            # random contrast 적용
            if  random.random() <= 0.5:
                data_numpy=self.contrast(data_numpy)
               
            # random brightness 적용
            if  random.random() <= 0.5:
                data_numpy=self.brightness(data_numpy)
            
            
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0
        
            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1
        
        if self.TTAUG==True:
            data_numpy=self.paste_cutout_TTA(data_numpy,0.5)

        trans = get_affine_transform(c, s, r, self.image_size)
        
        if self.TTAUG==False:
            input = cv2.warpAffine(
                data_numpy,
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR
            )
        elif self.TTAUG==True:       
            input = np.zeros(shape=(data_numpy.shape[0],int(self.image_size[1]),int(self.image_size[0]),3), dtype=np.uint8)
            for i in range(data_numpy.shape[0]):
                input[i] = cv2.warpAffine(
                    data_numpy[i],
                    trans,
                    (int(self.image_size[0]), int(self.image_size[1])),
                    flags=cv2.INTER_LINEAR
                )
                
       
        if self.TTAUG==False:
            if self.transform:
                input = self.transform(input)
                
        elif self.TTAUG==True:
            if self.transform:                                
                input_data=self.transform(input[0])
                input_data=input_data.unsqueeze(0)
                for i in range(1,input.shape[0]):
                    input_data=torch.cat((input_data,self.transform(input[i]).unsqueeze(0)),0)
                input=input_data

        if joints_vis is not None:
            for i in range(self.num_joints):
                if joints_vis[i, 0] > 0.0:
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

            target, target_weight = self.generate_target(joints, joints_vis)

            target = torch.from_numpy(target)
            target_weight = torch.from_numpy(target_weight)
            
        else:
            target = 1
            target_weight = 1

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }
        
        if joints_vis is not None:
            return input, target, target_weight, meta
        else:
            return input, target, target_weight, meta['center'], meta['scale'], meta['score'],trans,data_numpy,image_file

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight
