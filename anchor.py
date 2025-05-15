
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import math

class DataEncoder:
    """
    Anchor-based encoder for object detection tasks.

    This class generates anchor boxes based on the input image size and predefined anchor settings.
    It encodes ground truth bounding boxes and class labels into regression and classification targets
    that can be used for training object detection models.

    Example:
        encoder = DataEncoder(input_size=(450, 450), classes=[0,1])
        loc_target, cls_target = encoder.encoder(gt_boxes, gt_labels)

    Parameters:
        input_size (tuple): Size of the input image (height, width).
        classes (list): List of class names, e.g., ['person', 'car', 'dog'].
        anchor_ratio (list, optional): Aspect ratios for anchors (width/height). Default is [0.5, 1, 1.5].
        anchor_areas (list, optional): Areas of anchor boxes for each feature map level. Default is [208*208, 104*104, 26*26].
        fm_lst (list, optional): List of feature map sizes corresponding to each anchor area. Default is [13, 26, 52].
        scale (list, optional): List of scale multipliers for anchor sizes. Default is [1].
    """
    def __init__(self, 
                 input_size:tuple,
                 classes:list,
                 anchor_ratio:list = [0.5, 1, 1.5],
                 anchor_areas:list = [208*208,104*104,20*20],
                 fm_lst:list = [14,28,56],
                 scale:list = [1],
                 normalize:bool = True
                 ):
        self.input_size = input_size
        self.anchor_areas = anchor_areas
        self.aspect_ratios = anchor_ratio
        self.scales = scale
        self.normalize = normalize
        num_fms = len(self.anchor_areas)
        fm_sizes = fm_lst
        self.anchor_boxes = []
        self.anchor_block = []

        for i, fm_size in enumerate(fm_sizes):
            anchors = self.generate_anchors(self.anchor_areas[i], self.aspect_ratios, self.scales)
            anchor_grid = self.generate_anchor_grid(input_size, self.anchor_areas[i],fm_size, anchors)
            self.anchor_boxes.append(anchor_grid)
            self.anchor_block.append(anchors)
        
        self.anchor_boxes = torch.cat(self.anchor_boxes, 0)
        self.anchor_block = torch.tensor(self.anchor_block)
        self.classes = classes

    def generate_anchors(self, input_size, aspect_ratios, scales):
        anchor = []
        anchor_size = math.sqrt(input_size)
        for ratio in aspect_ratios:
            for scale in scales:
                width = round(anchor_size*scale) * ratio
                height = round(anchor_size*scale)* (1/ratio)
                anchor.append([width,height])
        return anchor

    # Need to machine with CNN output layer
    # 정사각행렬 이면 굳이 CNN output layer에 맞출 필요 없음
    def generate_anchor_grid(self,input_size,anchor_areas,fm_size,anchors):
        grid_size = round(input_size[0]/fm_size)
        x_grid = torch.arange(0,fm_size) * fm_size
        y_grid = torch.arange(0,fm_size) * fm_size
        x, y = torch.meshgrid(x_grid,y_grid,indexing='xy')
        x = x.flatten()
        y = y.flatten()
        grid = []

        for anchor in anchors:
            x1 = torch.clamp(x-anchor[0]/2,0,input_size[0])
            y1 = torch.clamp(y-anchor[1]/2,0,input_size[1])
            x2 = torch.clamp(x+anchor[0]/2,0,input_size[0])
            y2 = torch.clamp(y+anchor[1]/2,0,input_size[1])
            result = torch.stack([x1,y1,x2,y2],dim=1)
            grid.append(result)  # [N, 4]


        # Normalize anchor boxes to [0, 1] range to match YOLO label format
        # This ensures anchors and GT boxes are in the same coordinate system.
        grid = torch.stack(grid,dim=1).view(-1,4)
        if self.normalize:
            grid =  grid / torch.tensor([self.input_size[1], self.input_size[0], 
                                        self.input_size[1], self.input_size[0]], 
                                        dtype=torch.float32)
        return grid


    #boxes를 yolo format으로 설정했다면 corner format으로 변경해서 계산해야됨

    def encoder(self,boxes,classes,iou_threshold=0.6):
        if boxes.shape[0] == 0:
            loc_target = torch.zeros((self.anchor_boxes.shape[0],4),dtype=torch.float32)
            cls_target = torch.zeros((self.anchor_boxes.shape[0]),dtype=torch.int64)
            return loc_target,cls_target


        xyxy_boxes = self.yolo_to_xyxy(boxes)
        iou = self.cal_iou(xyxy_boxes,self.anchor_boxes)
        iou, ids = iou.max(dim=1)
        loc_target = self.loc_offset_cal(boxes[ids],self.anchor_boxes)
        loc_target = self.xyxy_to_yolo(loc_target)
        cls_target = classes[ids]+1
        cls_target[iou < iou_threshold] = -1 #ignore
        cls_target[iou < (iou_threshold-0.1)] = 0 #backgroud
        return loc_target,cls_target

    def decoder(self,
                pred:torch.tensor,
                iou_threshold:float=0.6,
                class_threshold:float=0.5)-> list:
        # 1. pred_offset를 박스와 라벨로 분리
        # anchor는 conner format으로 되어있음
        # anchor을 yolo format으로 변경해야함
        pred_offsets = pred[:,:,:4]
        pred_label = pred[:,:,4:]
        anchors = self.xyxy_to_yolo(self.anchor_boxes)
        decoded_boxes = []

        for pred_offset,pred_label in zip(pred_offsets,pred_label):

            # 3. pred_offset_box를 앵커와 비교해서 절대위치로 변환
            pred_boxes = self.offset_to_box(pred_offset,anchors)
            idx = pred_label[:,0] > iou_threshold
            pred_boxes = pred_boxes[idx]
            pred_label = pred_label[idx]
            result = torch.cat([pred_boxes,pred_label],dim=1)
            filtered_result = self.non_max_suppression(result, pred_label[:,0])
            decoded_boxes.append(filtered_result)
        return decoded_boxes

    def non_max_suppression(self,
                            pred:torch.tensor, 
                            nms_threshold:float=0.4)-> torch.tensor:
        # pred : (N, 5) -> (x1,y1,x2,y2,score)
        nms_boxes = []
        _ , sorted_indices = torch.sort(pred[:, 4], descending=True)
        sorted_pred = pred[sorted_indices]

        while sorted_pred.shape[0] > 0:
            top_box = sorted_pred[0, :4]
            top_score = sorted_pred[0, 4]
            top_class = sorted_pred[0, 5:]

            nms_boxes.append(torch.cat([top_box, top_score,top_class]))  # Append the box to nms_boxes

            iou = self.cal_iou(sorted_pred[1:,:],top_box)
            filtered_indices = iou < nms_threshold
            sorted_pred = sorted_pred[1:][filtered_indices]

        nms_boxes = torch.stack(nms_boxes, dim=0)
        return nms_boxes

    def offset_to_box(self,offsets, anchors):
        # anchors는 yolo format으로 컨버젼 해서 입력되고있음
        center_offsets = offsets[:, :2]
        box_offsets = offsets[:, 2:]
        anchor_wh = anchors[:, :2]
        anchor_ctr = anchors[:, :2]

        box_center = anchor_ctr+ center_offsets*anchor_wh
        box_size = torch.exp(box_offsets) * anchor_wh

        return torch.cat([box_center,box_size],dim = 1)

    def yolo_to_xyxy(self,bbox:torch.tensor):  # (4,) -> (1, 4)
        cx, cy, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)  
    
    def xyxy_to_yolo(self,bbox:torch.tensor):
        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        cx = (x2+x1) / 2
        cy = (y2+y1) / 2
        w = (x2-x1) 
        h = (y2-y1) 
        return torch.stack([cx, cy, w, h], dim=1)

    def cal_iou(self,boxes,anchor_box):
        p1 = torch.max(anchor_box[:,None,:2],boxes[:,:2])
        p2 = torch.min(anchor_box[:,None,2:],boxes[:,2:])
        inter = torch.prod((p2-p1+1).clamp(0),2)
        gt_area = torch.prod(boxes[:, 2:] - boxes[:, :2] + 1, 1)
        anchor_area = torch.prod(anchor_box[:, 2:] - anchor_box[:, :2] + 1, 1)
        iou = inter / (gt_area+anchor_area[:,None] - inter)
        return iou
    
    def loc_offset_cal(self, boxes, anchors):
        boxes_wh = boxes[:, 2:] - boxes[:, :2] + 1
        boxes_ctr  = boxes[:, :2] + 0.5*boxes_wh
        anchor_wh = anchors[:, 2:] - anchors[:, :2] + 1
        anchor_ctr  = anchors[:, :2] + 0.5*anchor_wh

        ctr_offset = (boxes_ctr-anchor_ctr)/anchor_wh
        wh_offset = torch.log(boxes_wh/anchor_wh)

        return torch.cat([ctr_offset,wh_offset],dim = 1)


