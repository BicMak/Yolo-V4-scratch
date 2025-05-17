import math

import torch
import numpy

class MeanAveragePrecision:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.pred_boxes = []
        self.pred_scores = []
        self.true_boxes = []
        self.true_labels = []
        self.image_ids = []

    def yolo_to_xyxy(self,boxes):
        x1 = boxes[:,:,0] - boxes[:,:,2]/2
        y1 = boxes[:,:,1] - boxes[:,:,3]/2
        x2 = boxes[:,:,0] + boxes[:,:,2]/2
        y2 = boxes[:,:,1] + boxes[:,:,3]/2
        return torch.stack([x1,y1,x2,y2],dim = 2)

    def cal_iou(self,gt_box,pred_box):    
        gt_box = self.yolo_to_xyxy(gt_box)
        pred_box = self.yolo_to_xyxy(pred_box)
        p1 = torch.max(gt_box[:,:,:2],pred_box[:,:,:2])
        p2 = torch.min(gt_box[:,:,2:],pred_box[:,:,2:])

        inter = torch.prod((p2-p1+1).clamp(0),dim=2)
        gt_area = torch.prod(torch.abs(gt_box[:,:, 2:] - gt_box[:,:, :2]+ 1) , dim =2)
        anchor_area = torch.prod(torch.abs(pred_box[:,:, 2:] - pred_box[:,:, :2]+ 1) , dim=2)

        epsilion = 1e-7
        iou_loss = inter / (gt_area+anchor_area - inter+ epsilion)
        return iou_loss
    
    def update(self, 
               pred_boxes:torch.tensor,
               pred_scores:torch.tensor,
               true_boxes:torch.tensor,
               true_labels:torch.tensor)-> None:
        '''
        update the state with the predictions and targets.

        parameters
            pred_boxes: (tensor) predicted boxes, sized [batch_size, anchors, 4].
            pred_scores: (tensor) predicted class confidences and ignore object score, sized [batch_size, anchors, classes].
            true_boxes: (tensor) encoded target boxes, sized [batch_size, anchors, 4].
            true_labels: (tensor) encoded target labels, sized [batch_size, anchors].
        '''
        
        self.pred_boxes.append(pred_boxes)
        #class gt class information만 체크함함
        pred_softmax = torch.softmax(pred_scores, dim=2)
        self.pred_scores = torch.argmax(pred_softmax, dim=2)
        self.true_boxes.append(true_boxes)
        self.true_labels.append(true_labels)

    def compute(self):
        iou = self.cal_iou(self.true_boxes, self.pred_boxes)
        iou_range = list(range(0, 1.1, 0.1))
        
        for i in range(self.pred_boxes.shape[0]):
            
            ap_points =[]
            # check every image in the batch
            for iou_thres in iou_range :
                pred_score = self.pred_scores[i]
                true_box_num = (self.true_boxes[i]).shape(0)
                true_label = self.true_labels[i]
                iou = iou[i]

                # Calculate True Positives and False Positives
                tp = (pred_score == true_label) & (iou > iou_thres)
                fp = (pred_score != true_label) & (iou > iou_thres)

                # Calculate Precision and Recall
                precision = tp.sum() / (tp.sum() + fp.sum())
                recall = tp.sum() / true_box_num.size(0)

                # Calculate Average Precision
                ap_points.append([precision,recall])
        


