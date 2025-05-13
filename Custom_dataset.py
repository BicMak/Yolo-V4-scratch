import os
import sys
import re
import cv2
import numpy as np
import torch
from anchor import DataEncoder

from torch.utils.data import Dataset

from PIL import Image
import albumentations as A
import random
from albumentations.pytorch import ToTensorV2

from CustomAugment.Cutmix import Cutmix
from CustomAugment.Mixup import MixUp
from CustomAugment.Mosaic import SimpleMosaic as Mosaic

class ListDataset(Dataset):
    '''
    make custom dataset for yolo v4

    Args:
        root_dir: (str) directory of image
        list_dir: (str) directory of annotation
        train: (boolean) train or test.
        transform: (bool) image transforms.
        input_size: (int) model input size.
    '''
    def __init__(self, 
                 image_dir:str, 
                 label_dir:str, 
                 classes, 
                 input_size:int = 450,
                 transform:bool = False):

        def natural_key(s):
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
        
        self.encoder = DataEncoder((input_size,input_size),classes=[0,1]) 
        self.image_dir = image_dir
        self.anno_dir = label_dir
        self.img_lst = sorted(os.listdir(self.image_dir),key = natural_key)
        self.anno_lst = sorted(os.listdir(self.anno_dir),key = natural_key)
        self.classes = classes
        self.num_samples = len(self.img_lst)
        self.input_size = input_size
        self.Transform = transform

        self.boxes = []
        self.labels = []
        self.size = []

        for i, file_name in enumerate(self.anno_lst):
            image = cv2.imread(os.path.join(self.image_dir, self.img_lst[i]))
            lines = open(self.anno_dir+file_name, 'r')
            lines = lines.readlines()
            box = []
            label = []
            for line in lines:
                line = list(map(float, line.strip().split()))
                clip_box = self.Clip_yoloformat(line[1],line[2],line[3],line[4])
                if clip_box is not None:
                    box.append(clip_box)
                    label.append(int(line[0]))
            self.boxes.append(box)
            self.labels.append(label)
    
    #In YOLOv4, every bounding box must be entirely inside the image.
    def Clip_yoloformat(self,cx,cy,w,h,eps=5):
        x1, x2 = np.clip([cx-w/2, cx+w/2],0,1)
        y1, y2 = np.clip([cy-h/2, cy+h/2],0,1)
        new_cx = (x2+x1)/2
        new_cy = (y2+y1)/2
        new_w = (x2-x1)
        new_h = (y2-y1)

        # Check boxes size, if it is smaller than eps, return None
        # eps is the minimun length of the boxes
        if (new_w*self.input_size) < eps or (new_h*self.input_size) < eps:
            return None  # 무효 bbox로 간주

        return [new_cx,new_cy,new_w,new_h]
        

        

    def add_agumentation(self,
                         basic_aug:A,
                         aug_para:dict,
                         finish_aug:A = None,
                         transform:bool = False):
        self.basic_aug = basic_aug
        self.mix_up = MixUp(size= self.input_size, alpha = aug_para["alpha"], prob= aug_para["prob"])
        self.four_mosaic = Mosaic(size = self.input_size)
        self.cut_mix =Cutmix(size = self.input_size,lamda = aug_para["lambda"], prob=aug_para["prob"])
        self.finish_aug = finish_aug
        self.Transform = transform

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        img = self.get_image(idx)

        boxes = self.boxes[idx]
        labels = self.labels[idx]
        #result = {"image":img,"bboxes":boxes,"class_labels":labels}
        result = self.basic_aug(image=img,bboxes=boxes,class_labels=labels)
        #result["bboxes"], result["class_labels"]=self.cut_bboxes(result["bboxes"], result["class_labels"])
        if self.Transform:
            result = self.mix_augmentation(idx,result)

        if self.finish_aug is not None:
            result = self.finish_aug(image=result['image'],bboxes=result['bboxes'],class_labels=result['class_labels'])
        
        #change the list to torch tensor
        return {
            "image":result['image'], 
            "bboxes":torch.tensor(result['bboxes'],dtype=torch.float32), 
            "class_labels":torch.tensor(result['class_labels'],dtype=torch.float32)
            }
    


    #albumutation 매서드 입력할때 arg 네임 정확하게 보셈셈
    def mix_augmentation(self,idx_1,auged_set1):
        prob = np.random.randint(0,3)
        remaining_indices = np.setdiff1d(np.arange(0, len(self.img_lst)), [idx_1])
        idx_2,idx_3,idx_4 = np.random.choice(remaining_indices, size=3, replace=False)

        auged_set2 = self.basic_aug(image=self.get_image(idx_2),bboxes=self.boxes[idx_2],class_labels=self.labels[idx_2])
        auged_set3 = self.basic_aug(image=self.get_image(idx_3),bboxes=self.boxes[idx_3],class_labels=self.labels[idx_3])
        auged_set4 = self.basic_aug(image=self.get_image(idx_4),bboxes=self.boxes[idx_4],class_labels=self.labels[idx_4])

        #delate over bboxes in image
        for auged_set in [auged_set1,auged_set2, auged_set3, auged_set4]:
            auged_set['bboxes'], auged_set['class_labels'] = self.cut_bboxes(
                auged_set['bboxes'], auged_set['class_labels']
            ) 
        for set_img in [auged_set1,auged_set2, auged_set3, auged_set4]:
            print("bboxes in img : {0}".format(set_img['bboxes']))    

        aug_names = {0:"mix up",1:"cut mix",2:"4 mosaic"}
        print("selected augmentation: {0}".format(aug_names[prob]))
        if prob == 0:
            result = self.mix_up(auged_set1,auged_set2)
        if prob == 1:
            result = self.cut_mix(auged_set1,auged_set2)
        if prob == 2:
            result = self.four_mosaic([auged_set1,auged_set2,auged_set3,auged_set4])
        result_img = np.clip(result[0],0,255)
        print("image type in mix augmetation function : {0}".format(result[0].dtype))

        new_bboxes, new_classLabel = self.cut_bboxes(result[1],result[2])
        return {"image":np.array(result_img,dtype=np.uint8), "bboxes":new_bboxes, "class_labels":new_classLabel}
    
    def cut_bboxes(self,bboxes,labels,min_wh=1e-3):
        new_bboxes = []
        new_labels = []
        for bbox, label in zip(bboxes, labels):
            x_c, y_c, w, h = bbox
            if (
                0.0 <= x_c <= 1.0 and
                0.0 <= y_c <= 1.0 and
                min_wh <= w <= 1.0 and
                min_wh <= h <= 1.0
            ):
                new_bboxes.append([x_c, y_c, w, h])
                new_labels.append(label)

        if new_bboxes:
            new_bboxes = np.clip(np.array(new_bboxes, dtype=np.float32), 0.0, 1.0)
        else:
            new_bboxes = np.zeros((0 , 4), dtype=np.float32) 
        return new_bboxes.tolist(), new_labels

    def get_image(self,idx):
        path = os.path.join(self.image_dir, self.img_lst[idx])
        img = cv2.imread(path)
        if img is None or np.prod(img.shape) == 0:
            print('cannot load image from path: ', path)
            sys.exit(-1)        
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (dict) of images, bboxes, class_lables.

        Returns:
          images, anchor_boxes, anchor_class_labels.
        '''
        imgs = [x['image'] for x in batch]
        boxes = [x['bboxes'] for x in batch]
        labels = [x['class_labels'] for x in batch]


        inputs = []
        loc_targets = []
        cls_targets = []

        for img, box, label in zip(imgs, boxes, labels):
            inputs.append(img)
            loc_target, cls_target = self.encoder.encoder(box, label)
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)

        inputs = torch.stack(inputs, dim=0).float()

        

        return {'image':inputs, 'bboxes':torch.stack(loc_targets), 'class_labels':torch.stack(cls_targets)}

    def __len__(self):
        return self.num_samples
    
