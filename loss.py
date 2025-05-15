import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, 
                preds:torch.tensor,
                targets:torch.tensor)-> dict:
        """
        loc_preds:   [B,N,4] predicted bbox in [cx, cy, w, h]
        obj_preds:   [B,N] predicted objectness logits
        cls_preds:   [B,N,num_classes] predicted class logits
        loc_targets: [B,N,4] target bbox in [cx, cy, w, h]
        cls_targets: [B,N] predicted class logits
        """

        loc_preds = preds[:,:,:4]
        obj_preds = preds[:,:,4]
        cls_preds = preds[:,:,5:]
        loc_targets = targets[:,:,:4]
        cls_targets = targets[:,:,4]

        # 1. Box CIoU Loss
        box_loss = self.cal_ciou(loc_targets, loc_preds)

        # 2. Objectness Loss
        # Encoding 과정에서 계산된 클래스 기준으로 구분 진행행
        obj_gt = cls_targets
        obj_gt = torch.where(obj_gt > 0, torch.tensor(1), torch.tensor(0)) 
        obj_loss = self.compute_objectness_loss(obj_preds, obj_gt)

        # 3. Classification Loss

        cls_loss = self.compute_classification_loss(cls_preds, cls_targets)

        # 4. Total Loss (가중치 조절하고 싶으면 여기서 weight 줘도 됨)
        total_loss = box_loss + obj_loss + cls_loss

        return {
            'total': total_loss,
            'ciou': box_loss,
            'obj': obj_loss,
            'cls': cls_loss,
        }

    def compute_objectness_loss(self,pred_obj, target_obj):
        """
        Compute objectness loss using binary cross-entropy.

        parameters:
            pred_obj: [B, N, 1] 예측된 objectness score (Sigmoid 전 or 후 가능)
            target_obj: [B, N, 1] 각 anchor에 대해 GT와의 IoU (최대값) -> Encoding 에서 획득한걸로 봐야될듯?

        """
        obj_target = torch.where(target_obj > 0, torch.tensor(1), torch.tensor(0))   # GT와 매칭: 1, 나머지: 0
        obj_loss = F.binary_cross_entropy_with_logits(pred_obj, obj_target, reduction='none')
        obj_loss = obj_loss * obj_target  # 무시된 것들에 대해 loss 줄임

        return obj_loss.mean()

    def compute_classification_loss(self,pred_cls, gt_cls):
        """
        pred_cls: [B, N, num_classes] → sigmoid score for each class
        gt_cls:   [B, N, num_classes] → one-hot encoded GT class vector (only for positive anchors)

        N = matched anchor 수 (positive only)
        """
        cls_target = F.one_hot(gt_cls, num_classes=self.num_classes+2)
        target_obj = cls_target[:,:,2:]
        target_mask = (gt_cls <= 0).float()  #gt_cls가 0보다 같거나 작은걸 0으로바꿔줌줌
        cls_loss = F.binary_cross_entropy_with_logits(pred_cls, target_obj, reduction='mean')
        return (cls_loss* target_mask).mean()

    
    def cal_ciou(self,gt_loc,pred_loc):
        #cal_d_loss
        iou = self.cal_iou(gt_loc,pred_loc) # 1-Iou
        c = self.cal_EuclidDist(gt_loc,pred_loc)
        d = self.cal_reclength(gt_loc,pred_loc)
        print(f"iou shape: {iou.shape}")
        print(f"c shape: {c.shape}")
        print(f"d shape: {d.shape}")

        # cal_c)loss value
        v, alpha = self.cal_v_alpha(gt_loc,pred_loc,iou)
        print(f"v shape: {v.shape}")
        print(f"alpha shape: {alpha.shape}")

        result= 1- iou + (c/d)**2 + (alpha*v)
        return result.mean()

    def yolo_to_xyxy(self,box):
        x1 = box[:,:,0] - box[:,:,2]/2
        y1 = box[:,:,1] - box[:,:,3]/2
        x2 = box[:,:,0] + box[:,:,2]/2
        y2 = box[:,:,1] + box[:,:,3]/2
        return torch.stack([x1,y1,x2,y2],dim = 2)

    def cal_iou(self,gt_box,pred_box):
        gt_box = self.yolo_to_xyxy(gt_box)
        pred_box = self.yolo_to_xyxy(pred_box)
        p1 = torch.max(gt_box[:,:,:2],pred_box[:,:,:2])
        p2 = torch.min(gt_box[:,:,2:],pred_box[:,:,2:])

        inter = torch.prod((p2-p1+1).clamp(0),dim=2)
        gt_area = torch.prod(gt_box[:,:, 2:] - gt_box[:,:, :2] + 1, dim =2)
        anchor_area = torch.prod(pred_box[:,:, 2:] - pred_box[:,:, :2] + 1, dim=2)

        iou_loss = inter / (gt_area+anchor_area - inter)
        return iou_loss

    def cal_EuclidDist(self,gt_box,pred_box):       
        dist = torch.abs(gt_box[:,:,:2] - pred_box[:,:,:2])
        euclid_dist = torch.sqrt(torch.sum(dist ** 2, dim=2))
        return euclid_dist

    def cal_reclength(self,gt_box,pred_box):
        gt_box = self.yolo_to_xyxy(gt_box)
        pred_box = self.yolo_to_xyxy(pred_box)
        p1 = torch.min(gt_box[:,:,:2],pred_box[:,:,:2])
        p2 = torch.max(gt_box[:,:,2:],pred_box[:,:,2:])
        result = self.cal_EuclidDist(p1,p2)
        return result

    def cal_v_alpha(self,gt_boxes, pred_boxes, iou):
        """
        gt_boxes, pred_boxes: (N, 4) in format [cx, cy, w, h]
        iou: (N,) tensor
        return: v, alpha tensors of shape (N,)
        """
        w_gt = gt_boxes[:, :, 2]
        h_gt = gt_boxes[:, :, 3]
        w_pred = pred_boxes[:, :, 2]
        h_pred = pred_boxes[:, :, 3]

        atan_gt = torch.atan(w_gt / h_gt)
        atan_pred = torch.atan(w_pred / h_pred)
        
        v = (4 / (math.pi ** 2)) * (atan_gt - atan_pred).pow(2)
        print(f"v shape: {v.shape}")
        alpha = v / (1 - iou + v + 1e-7)  # small epsilon added for numerical stability

        return v, alpha 