
import numpy as np
import albumentations as A
from dataset import ListDataset
from albumentations.pytorch import ToTensorV2

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
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'],check_each_transform=False)
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

        for box,label in zip(dataset['bboxes'],dataset['class_labels']):
            x, y, w, h = box  # yolo 형식이면
            # 1. YOLO → 절대좌표로 변환
            abs_cx = x * width_img
            abs_cy = y * height_img
            abs_w = w * width_img
            abs_h = h * height_img
            #print("size of bbox of im2 x, y= {0}, {1}".format(abs_x,abs_y))
            #print("BBbox size {0}*{1}".format(abs_w,abs_h))
            

            # 2. 이미지 patch가 삽입된 실제 위치로 좌표 이동
            #print("pivot position {0}, {1}".format(position[0],position[1]))

            new_x = (position[0] + abs_cx) 
            new_y = (position[1] + abs_cy) 
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