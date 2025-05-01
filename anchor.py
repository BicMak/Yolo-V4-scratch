
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import math

class DataEncoder:
    def __init__(self, 
                 input_size,
                 classes,
                 anchor_ratio = [0.5, 1, 1.5],
                 anchor_areas = [208*208,104*104,26*26],
                 fm_lst = [13,26,52],
                 scale = [1],
                 ):
        self.input_size = input_size
        self.anchor_areas = anchor_areas
        self.aspect_ratios = anchor_ratio
        self.scales = scale
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

        grid = torch.stack(grid,dim=1)
        return grid.view(-1,4)

    def encoder(self,boxes,classes,iou_threshold=0.5):
        iou = self.cal_iou(boxes,self.anchor_boxes)
        iou, ids = iou.max(dim=1)
        loc_target = self.loc_offset_cal(boxes[ids],self.anchor_boxes)
        cls_target = classes[ids]+1
        cls_target[iou < iou_threshold] = -1 #ignore
        cls_target[iou < (iou_threshold-0.1)] = 0 #backgroud
        return loc_target,cls_target

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


