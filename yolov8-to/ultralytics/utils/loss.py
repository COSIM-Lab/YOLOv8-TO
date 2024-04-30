# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from .metrics import bbox_iou
from .tal import bbox2dist
import numpy as np
import sys
# torch.autograd.set_detect_anomaly(True)

def display_shape(input_item, prefix=""):
    # If the input_item is a list or a tuple, iterate through its elements
    if isinstance(input_item, (list, tuple)):
        for idx, item in enumerate(input_item):
            # For nested lists or tuples, add an additional level to the prefix
            new_prefix = f"{prefix}[{idx}]"
            display_shape(item, new_prefix)
    # If the input_item is a tensor, print its shape
    elif isinstance(input_item, torch.Tensor):
        print(f"{prefix}: {input_item.shape}")
    else:
        print(f"Unsupported type {type(input_item)} at {prefix}")


def Heaviside(phi, alpha, epsilon):
    device = phi.device  # Get the device of phi

    # For values outside of [-epsilon, epsilon]
    H_positive = torch.ones_like(phi, device=device) 
    H_negative = alpha * torch.ones_like(phi, device=device)

    # For values inside [-epsilon, epsilon]
    default = 3 * (1 - alpha) / 4 * (phi / epsilon - phi**3 / (3 * epsilon**3)) + (1 + alpha) / 2

    # Construct Heavisidve using conditions
    H = torch.where(phi > epsilon, H_positive, torch.where(phi < -epsilon, H_negative, default))

    return H
def smooth_heaviside(phi, alpha, epsilon):
    # Scale and shift phi for the sigmoid function
    scaled_phi = (phi - alpha) / epsilon
    
    # Apply the sigmoid function
    H = torch.sigmoid(scaled_phi)

    return H

def calc_Phi(variable, LSgrid):
    device = variable.device  # Get the device of the variable

    x0 = variable[0]
    y0 = variable[1]
    L = variable[2]
    t = variable[3]  # Constant thickness
    angle = variable[4]

    # Rotation
    st = torch.sin(angle)
    ct = torch.cos(angle)
    x1 = ct * (LSgrid[0][:, None].to(device) - x0) + st * (LSgrid[1][:, None].to(device) - y0) 
    y1 = -st * (LSgrid[0][:, None].to(device) - x0) + ct * (LSgrid[1][:, None].to(device) - y0)

    # Regularized hyperellipse equation
    a = L / 2  # Semi-major axis
    b = t / 2  # Constant semi-minor axis
    small_constant = 1e-9  # To avoid division by zero
    temp = ((x1 / (a + small_constant))**6) + ((y1 / (b + small_constant))**6)

    # # Ensuring the hyperellipse shape
    allPhi = 1 - (temp + small_constant)**(1/6)

    # # Call Heaviside function with allPhi
    alpha = torch.tensor(0.0, device=device, dtype=torch.float32)
    epsilon = torch.tensor(0.001, device=device, dtype=torch.float32)
    H_phi = smooth_heaviside(allPhi, alpha, epsilon)
    return allPhi, H_phi


class CustomDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CustomDiceLoss, self).__init__()
        self.size_average = size_average

    def forward(self, inputs, targets, smooth=1):
        
        # If your model contains a sigmoid or equivalent activation layer, comment this line
        #inputs = F.sigmoid(inputs)       

        # Check if the input tensors are of expected shape
        if inputs.shape != targets.shape:
            raise ValueError("Shape mismatch: inputs and targets must have the same shape")

        # Flatten tensors
        inputs_flat = inputs.view(inputs.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)

        # Compute intersections and unions
        intersection = (inputs_flat * targets_flat).sum(1)
        union = inputs_flat.sum(1) + targets_flat.sum(1)

        # Compute Dice
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice

        # If you want the average loss over the batch to be returned
        if self.size_average:
            return dice_loss.mean()
        else:
            # If you want individual losses for each sample in the batch
            return dice_loss
        
class CustomTverskyLoss(nn.Module):
    def __init__(self, alpha, beta, size_average=True):
        super(CustomTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.size_average = size_average

    def forward(self, inputs, targets, smooth=1):
        
        # If your model contains a sigmoid or equivalent activation layer, uncomment this line
        # inputs = F.sigmoid(inputs)       

        # Check if the input tensors are of expected shape
        if inputs.shape != targets.shape:
            raise ValueError("Shape mismatch: inputs and targets must have the same shape")

        # Flatten tensors
        inputs_flat = inputs.view(inputs.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)

        # Compute true positives, false positives, and false negatives
        true_pos = (inputs_flat * targets_flat).sum(1)
        false_neg = ((1 - inputs_flat) * targets_flat).sum(1)
        false_pos = (inputs_flat * (1 - targets_flat)).sum(1)

        # Compute the Tversky index
        tversky_index = (true_pos + smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + smooth)
        tversky_loss = 1 - tversky_index

        # If you want the average loss over the batch to be returned
        if self.size_average:
            return tversky_loss.mean()
        else:
            # If you want individual losses for each sample in the batch
            return tversky_loss


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367."""

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).mean(1).sum()
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, ):
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / (torch.sum(kpt_mask != 0) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return kpt_loss_factor * ((1 - torch.exp(-e)) * kpt_mask).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.nm = model.model[-1].nm  # number of masks
        try:
            self.overlap = model.args.overlap_mask
        except:
            self.overlap =False
        self.diceloss = CustomDiceLoss()
        #self.bceloss = nn.BCELoss()
        #self.tversky_loss= CustomTverskyLoss(alpha=0.7, beta=0.3)

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl
        if len(preds) ==3:
            feats, pred_masks, proto = preds 
        elif len(preds) ==4:
            feats, pred_masks, proto, regression_tensor = preds
            #Let's describe each variables:
            #display_shape(preds)
        else:
            feats, pred_masks, proto, regression_tensor = preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

        
        
        if 'regression_vars' in batch:
            max_objects = 300  # Set the fixed maximum number of objects
            padded_vars = [np.pad(item, ((0, max_objects - len(item)), (0, 0)), mode='constant') for item in batch['regression_vars']]
            regression_targets = torch.tensor(np.stack(padded_vars)).to(self.device).float()
            # pboxes
            pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
            test_labels, target_bboxes, target_scores, fg_mask, target_gt_idx, regression_scores = self.assigner(
                pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt, regression_targets)
        else:
            # regression targets will only contain zeros:
            regression_targets = torch.zeros((batch_size, 6, 300), device=self.device).float()
            pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
            test_labels, target_bboxes, target_scores, fg_mask, target_gt_idx, regression_scores = self.assigner(
                pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt, gt_regression= None)


        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way

        REG_LOSS = 'pixels'
        # if 'regression_vars' in batch:
        if REG_LOSS == 'direct':
            # Assuming fg_mask has shape (b, h*w)
            # Expand the dimensions of fg_mask to match regression_tensor
            fg_regression_mask = fg_mask.unsqueeze(1).expand(-1, 6, -1)  # fg_regression_mask now has shape (BS, 6, 8400)
            filtered_predictions = regression_tensor[fg_regression_mask]
            #check if there are nans in regression_tensor:
            

            filtered_target = regression_scores[fg_regression_mask.permute(0,2,1).contiguous()]
            # Now create masked versions of your regression tensor and regression scores
            # Compute MSE loss on masked tensors

            regression_loss = F.mse_loss(filtered_predictions, filtered_target,reduction="mean")

        if (REG_LOSS == 'pixels' or REG_LOSS=="level") and self.hyp.reg_gain > 0:
            # if torch.isnan(regression_tensor).any():
            #     print("There are nans in regression_tensor")
            #     sys.exit()
            DW = 1.0
            DH = 1.0

            nelx = int(200 * DW)
            nely = int(200 * DH)

            x, y = torch.meshgrid(torch.linspace(0, DW, nelx+1), torch.linspace(0, DH, nely+1))
            LSgrid = torch.stack((y.flatten(), x.flatten()), dim=0)

            
            
        #print(pred_scores.shape,target_scores.shape) torch.Size([8, 8400, 1]) torch.Size([8, 8400, 1])
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE   

        if fg_mask.sum():
            # bbox loss
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)
            # masks loss
            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            for i in range(batch_size):
                if fg_mask[i].sum():
                    mask_idx = target_gt_idx[i][fg_mask[i]]
                    if self.overlap:
                        gt_mask = torch.where(masks[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                    xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                    
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    loss[1] += self.single_mask_loss(gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy, marea)  # seg
                    test_bboxes = pred_bboxes*stride_tensor

                    test_bboxes = test_bboxes[i][fg_mask[i]]  / imgsz[[1, 0, 1, 0]]
                    # clip the test_bboxes between 0 and 1:
                    test_bboxes = torch.clip(test_bboxes,0,1)

                    filtered_regression = regression_tensor[i][:,fg_mask[i]]
           
                    # Check if CUDA is available

                    # Create the constant tensors on the specified device
                    constant_tensor_02 = torch.full((test_bboxes.shape[0],), 0.2, device=self.device)
                    constant_tensor_00 = torch.full((test_bboxes.shape[0],), 0.001, device=self.device)
                    # Stack the tensors and move them to the specified device
                    xmax = torch.stack([test_bboxes[:,2], test_bboxes[:,3], test_bboxes[:,2], test_bboxes[:,3], constant_tensor_02], dim=1).to(self.device)
                    xmin = torch.stack([test_bboxes[:,0], test_bboxes[:,1], test_bboxes[:,0], test_bboxes[:,1], constant_tensor_00], dim=1).to(self.device)
                    unnormalized_preds = filtered_regression.T * (xmax - xmin) + xmin
                    
                    # # # The design variables are infered from the two endpoints and the two thicknesses:
                    x_center = (unnormalized_preds[:, 0] + unnormalized_preds[:, 2]) / 2
                    y_center = (unnormalized_preds[:, 1] + unnormalized_preds[:, 3]) / 2

                    L = torch.sqrt((unnormalized_preds[:, 0] - unnormalized_preds[:, 2])**2 + 
                                (unnormalized_preds[:, 1] - unnormalized_preds[:, 3])**2)

                    L = L+1e-4
                    t_1 = unnormalized_preds[:, 4]

                    epsilon = 1e-10
                    y_diff = unnormalized_preds[:, 3] - unnormalized_preds[:, 1] + epsilon
                    x_diff = unnormalized_preds[:, 2] - unnormalized_preds[:, 0] + epsilon
                    theta = torch.atan2(y_diff, x_diff)
                    formatted_variables = torch.cat((x_center.unsqueeze(1), 
                                        y_center.unsqueeze(1), 
                                        L.unsqueeze(1), 
                                        t_1.unsqueeze(1), 
                                        theta.unsqueeze(1)), dim=1)

                    pxyxy = test_bboxes * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)

                    if (REG_LOSS == "pixels" or REG_LOSS=="level") and self.hyp.reg_gain > 0:

                        # filtered_predictions = formatted_variables[i][:,fg_mask[i]]
                        filtered_predictions = formatted_variables
                        pred_phi , H_phi = calc_Phi(filtered_predictions.T,LSgrid.to(self.device))
                        if REG_LOSS == "level":
                            pred_phi= torch.reshape(pred_phi,(nely+1,nelx+1,H_phi.shape[-1]))
                            normalized = (pred_phi - pred_phi.min()) / (pred_phi.max() - pred_phi.min())
                            cropped_gt_mask = crop_mask(gt_mask,pxyxy)

                            normalized = normalized.permute(2, 0, 1).unsqueeze(1)  # Now the shape is ([80, 1, 51, 51])
                            normalized = F.interpolate(normalized, size=cropped_gt_mask.shape[-2:], mode='nearest')
                            
                            level_loss = F.mse_loss(normalized.squeeze(1), cropped_gt_mask, reduction="mean")
                

                            loss[4]+=level_loss

                        else:
                            H_phi= torch.reshape(H_phi,(nely+1,nelx+1,H_phi.shape[-1]))
                        
                            # Rearrange H_phi to the shape ([batch_size, channels, height, width])
                            H_phi = H_phi.permute(2, 0, 1).unsqueeze(1)  # Now the shape is ([80, 1, 51, 51])
                            cropped_gt_mask = crop_mask(gt_mask,pxyxy)
                            # Use interpolate to resize
                            H_phi_resized = F.interpolate(H_phi, size=cropped_gt_mask.shape[-2:], mode='nearest')
                            # Rearrange H_phi_resized back to the shape ([height, width, batch_size])
                            H_phi_resized = H_phi_resized.squeeze(1)  # Now the shape is ([80, 160, 160])
                            #Threhsold the H_phi_resized
                            #H_phi_resized = torch.where(H_phi_resized > 0.5, 1.0, 0.0)
                            dice = self.diceloss(H_phi_resized, cropped_gt_mask)
                            #tversky = self.tversky_loss(H_phi_resized, cropped_gt_mask)
                            #mse = F.mse_loss(H_phi_resized, cropped_gt_mask, reduction="mean")
                            loss[4]+= dice


                # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
                else:
                    loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
                    loss[4] += 0.0
        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
            loss[4] += 0.0
        if REG_LOSS =='direct':
            loss[4] = regression_loss
        else:
            loss[4] *= self.hyp.reg_gain / batch_size

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box / batch_size  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain
        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        """Mask loss for one image."""
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n, 32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()
    
    def single_reg_loss(self, gt_mask, pred, xyxy, area):
        """Mask loss for one image."""
        loss = F.binary_cross_entropy_with_logits(pred, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            for i in range(batch_size):
                if fg_mask[i].sum():
                    idx = target_gt_idx[i][fg_mask[i]]
                    gt_kpt = keypoints[batch_idx.view(-1) == i][idx]  # (n, 51)
                    gt_kpt[..., 0] /= stride_tensor[fg_mask[i]]
                    gt_kpt[..., 1] /= stride_tensor[fg_mask[i]]
                    area = xyxy2xywh(target_bboxes[i][fg_mask[i]])[:, 2:].prod(1, keepdim=True)
                    pred_kpt = pred_kpts[i][fg_mask[i]]
                    kpt_mask = gt_kpt[..., 2] != 0
                    loss[1] += self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss
                    # kpt_score loss
                    if pred_kpt.shape[-1] == 3:
                        loss[2] += self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose / batch_size  # pose gain
        loss[2] *= self.hyp.kobj / batch_size  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='sum') / 64
        loss_items = loss.detach()
        return loss, loss_items
