import os
import sys

import cv2
import numpy as np
import torch
from anchor import DataEncoder

from torch.utils.data import Dataset

from PIL import Image
import albumentations as A
import random
from albumentations.pytorch import ToTensorV2

class ListDataset(Dataset):
    def __init__(self, 
                 image_dir, 
                 label_dir, 
                 classes, 
                 input_size = 450,
                 transform = False):
        '''
        Args:
          root_dir: (str) directory of image
          list_dir: (str) directory of annotation
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.image_dir = image_dir
        self.anno_dir = label_dir
        self.img_lst = os.listdir(self.image_dir)
        self.anno_lst = os.listdir(self.anno_dir)
        self.classes = classes
        self.num_samples = len(self.img_lst)
        self.input_size = input_size
        self.Transform = transform
        self.encoder = DataEncoder(input_size = (input_size,input_size),classes = [0,1])

        self.boxes = []
        self.labels = []
        self.size = []

        for i, file_name in enumerate(self.anno_lst):
            image = cv2.imread(self.image_dir+self.img_lst[i])
            height, width = image.shape[:2] 
            lines = open(self.anno_dir+file_name, 'r')
            lines = lines.readlines()
            box = []
            label = []
            for line in lines:
                line = list(map(float, line.strip().split()))
                box.append([line[1],line[2],line[3],line[4]])
                label.append(int(line[0]))
            self.boxes.append(box)
            self.labels.append(label)
            self.size.append((width,height))

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

        result = self.basic_aug(image=img,bboxes=boxes,class_labels=labels)

        if self.Transform:
            result = self.mix_augmentation(idx,result)
        print(result['image'].dtype)
        result = self.finish_aug(image=result['image'],bboxes=result['bboxes'],class_labels=result['class_labels'])
        
        #change the list to torch tensor
        return {
            "image":result['image'], 
            "bboxes":torch.tensor(result['bboxes'],dtype=torch.float32), 
            "class_labels":torch.tensor(result['class_labels'],dtype=torch.float32)
            }
    
    def add_agumentation(self,
                         basic_aug:A,
                         aug_para:dict,
                         finish_aug:A):
        self.basic_aug = basic_aug
        self.mix_up = MixUp(size= self.input_size, alpha = aug_para["alpha"], prob= aug_para["prob"])
        self.four_mosaic = SimpleMosaic(size = self.input_size)
        self.cut_mix =Cutmix(size = self.input_size,lamda = aug_para["lambda"], prob=aug_para["prob"])
        self.finish_aug = finish_aug


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

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, w, h)
        loc_targets = []
        cls_targets = []

        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encoder(boxes[i], labels[i])
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)

        

        return {'image':inputs, 'bboxes':torch.stack(loc_targets), 'class_labels':torch.stack(cls_targets)}

    def __len__(self):
        return self.num_samples

class MixUp:
    '''
    Custom image augmentation with 2 image.
    mix each images on a array.
    
    Args:
      size: length of the side of the square
      alpha: weight of each image. the larger of alpha, the more vivid first image
      prob: probability augmentaion occur

    Returns:
      mixed images, boxes, label.
    '''
    def __init__(self, 
                 size:int=450,
                 alpha:float=0.7,
                 prob:float=0.5):
        self.alpha = alpha
        self.prob = prob
        self.size = size

    def __call__(self, 
                 dataset1:ListDataset,
                 dataset2:ListDataset):
        print("MIXUP BBOX CHECK")
        for b in dataset1['bboxes'] + dataset2['bboxes']:
            print("before cut:", b)
        print("-"*20)
        img1 = np.array(dataset1['image']) / 255
        img2 = np.array(dataset2['image']) / 255
        mixed_img = (self.alpha * img1 + (1-self.alpha) * img2)
        mixed_img = (mixed_img-np.min(mixed_img)) /  (np.max(mixed_img)-np.min(mixed_img))
        mixed_img *= 255
        mixed_boxes = dataset1['bboxes'] + dataset2['bboxes'] 
        mixed_labels = dataset1['class_labels'] + dataset2['class_labels']


        return mixed_img, mixed_boxes, mixed_labels
    
 
class SimpleMosaic(A.DualTransform):
    def __init__(self, size=450, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.image_size = size

    def apply(self, img, additional_targets=(), **params):
        return img  # dummy, 실질 로직은 `__call__`에서 처리

    def apply_to_bbox(self, bbox, **params):
        return bbox  # dummy, bbox 변환은 따로 처리함

    def get_transform_init_args_names(self):
        return ("image_size",)

    def __call__(self, dataset_lst,force_apply=False, ):

        size = self.image_size
        mosaic_img = np.full((size, size, 3), 114, dtype=np.uint8)
        xc, yc = np.random.randint(size*0.25,size*0.75,2)
        start_points = [[0,0],[yc,0],[0,xc],[yc,xc]]
        img_sizes = [[yc,xc],[size-yc,xc],[yc,size-xc],[size-yc,size-xc]]
        final_bboxes, final_labels = [], []

        for i in range(4):
            img_size = img_sizes[i]
            start_point = start_points[i]
            dataset = dataset_lst[i]
            resize_img = A.Compose(
                [A.Resize(height=img_size[0],width =img_size[1])]
                , bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            dataset = resize_img(image=dataset["image"],
                                 bboxes = dataset["bboxes"],
                                 class_labels = dataset["class_labels"])
            y1, x1 = start_point[0], start_point[1]
            y2, x2 = y1+img_size[0], x1+img_size[1]
            mosaic_img[y1:y2,x1:x2,:] = dataset["image"]


            for bbox, label in zip(dataset["bboxes"],dataset["class_label"]):
                x_c, y_c, bw, bh = bbox

                bw *= img_size[1]
                bh *= img_size[0]
                x_c = x_c * img_size[1]  + x1
                y_c = y_c * img_size[0]  + y1

                final_bboxes.append([
                    x_c / (size),
                    y_c / (size),
                    bw / (size),
                    bh / (size)
                ])
                final_labels.append(label)

        return mosaic_img,final_bboxes,final_labels

        
#일단 세이프티 하게 크롭이 됬다고 가정           
     
class Cutmix:
    '''
    Custom image augmentation with 2 image.
    insert reized image to original image
    
    Args:
      size: length of the side of the square
      alpha: weight of each image. the larger of alpha, the more vivid first image
      prob: probability augmentaion occur

    Returns:
      mixed images, boxes, label.
    '''
    def __init__(self, 
                 size:int=450,
                 lamda:float=0.7,
                 prob:float=0.5):
        self.lamda = lamda
        self.prob = prob
        self.size = size

    def __call__(self, 
                 dataset1:ListDataset,
                 dataset2:ListDataset):
        print("MIXUP BBOX CHECK")
        for b in dataset1['bboxes'] + dataset2['bboxes']:
            print("before cut:", b)
        print("-"*20)

        img1 = np.array(dataset1['image'])
        img2 = np.array(dataset2['image'])

        
        width_img2 = int(self.size*(np.sqrt(1-self.lamda)))
        height_img2 = width_img2


        #random 범위 지정정
        start,end = map(int, [0, self.size-width_img2])
        min_point = min(start,end)
        max_point = max(start,end)

        x1,y1 = np.random.randint(min_point,max_point,2,int)

        resize_img = A.Compose(
            [A.Resize(height=height_img2,width =width_img2)],
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
            )
        img2_resize = resize_img(image=dataset2['image'],
                                bboxes=dataset2['bboxes'],
                                class_labels=dataset2['class_labels'])
        
        img2 = img2_resize['image']
        x2 = x1 + width_img2 
        y2 = y1 + height_img2 
        x1, x2, y1, y2 = map(int, [x1, x2, y1, y2])
        #print("im2 start point  = {0}, {1}".format(x1,y1))
        #print("im2 end point  = {0}, {1}".format(x2,y2))

        dataset1['bboxes'], dataset1['class_labels'] = self.Remove_bbox(dataset1,
                                                                        (x1,y1,x2,y2),
                                                                        (self.size,self.size))
        img2_resize['bboxes'] = self.Move_box(img2_resize,
                                              (x1,y1,x2,y2),
                                              (width_img2,height_img2))

        img1[y1:y2,x1:x2,:] = img2
        mixed_boxes = dataset1['bboxes'] + img2_resize['bboxes'] 
        mixed_labels = dataset1['class_labels'] + img2_resize['class_labels']

        return img1, mixed_boxes, mixed_labels

    def Remove_bbox(self, 
                    dataset:ListDataset,
                    position:tuple,
                    size:tuple):
        
        new_bboxes = []
        new_labels = []
        
        for box, label in zip(dataset['bboxes'], dataset['class_labels']):
            x, y, w, h = box
            x = x * size[0]
            y = y * size[1]

            x_center = x
            y_center = y

            if not((position[0] < x_center< position[2]) and (position[1] < y_center < position[3])):
                    new_bboxes.append(box)
                    new_labels.append(label)

        return new_bboxes, new_labels
    
    def Move_box(self, 
                 dataset:ListDataset,
                 position:tuple,
                 size:tuple):
        
        new_bboxes = []
        width_img = size[0]
        height_img = size[1]
        print("size of image x, y= {0}, {1}".format(width_img,height_img))

        for box,label in zip(dataset['bboxes'],dataset['class_labels']):
            x, y, w, h = box  # yolo 형식이면
            # 1. YOLO → 절대좌표로 변환
            abs_x = int(x * width_img)
            abs_y = int(y * height_img)
            abs_w = w * width_img
            abs_h = h * height_img
            #print("size of bbox of im2 x, y= {0}, {1}".format(abs_x,abs_y))
            #print("BBbox size {0}*{1}".format(abs_w,abs_h))
            

            # 2. 이미지 patch가 삽입된 실제 위치로 좌표 이동
            #print("pivot position {0}, {1}".format(position[0],position[1]))

            new_x = (position[0] + abs_x) 
            new_y = (position[1] + abs_y) 
            # print("size of move x1, y1= {0}, {1}".format(new_x,new_y))

            # 3. 이제 박스를 오리지널 기준으로 
            new_x = new_x  / self.size
            new_y = new_y  / self.size
            new_w = abs_w / self.size
            new_h = abs_h / self.size
            #print("size of normalize = {0}, {1}".format(new_x,new_y))
            #print("size of normalize = {0}, {1}".format(new_w,new_h))
            
            new_bboxes.append([new_x, new_y, new_w, new_h])

        return new_bboxes
    
