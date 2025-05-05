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

    def __call__(self, dataset_lst,force_apply=False):
        print("4mosaic CHECK")
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
                , bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'],check_each_transform=False))
            dataset = resize_img(image=dataset["image"],
                                 bboxes = dataset["bboxes"],
                                 class_labels = dataset["class_labels"])
            y1, x1 = start_point[0], start_point[1]
            y2, x2 = y1+img_size[0], x1+img_size[1]
            
            mosaic_img[y1:y2,x1:x2,:] = dataset["image"]


            for bbox, label in zip(dataset["bboxes"],dataset["class_labels"]):
                x_c, y_c, bw, bh = bbox

                bw *= img_size[1] #width
                bh *= img_size[0] #height
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