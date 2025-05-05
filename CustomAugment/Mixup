import numpy as np
from dataset import ListDataset

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