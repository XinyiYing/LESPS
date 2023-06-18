from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
from skimage import measure
import os
from model import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        
        if model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')  
        elif model_name == 'ACM':
            self.model = ACM()
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        
        self.cal_loss = FocalLoss()
        
    def forward(self, img):
        pred = self.model(img)
        return pred

    def loss(self, pred, gt_mask):
        target_mask, avg_factor = gt_mask, max(1, (gt_mask.eq(1)).sum())
        loss = self.cal_loss(pred, target_mask, avg_factor=avg_factor)
        return loss
        
    def update_gt(self, pred, gt_mask, thresh_Tb, thresh_k, size):
        bs, c, feat_h, feat_w = pred.shape
        background_length = 33
        target_length = 3
        gt_mask_pos = (gt_mask>0.5).float()
        updated_gt_mask = gt_mask.clone()
        
        label_image = measure.label((gt_mask_pos[0,0,:,:]).cpu())
        for region in measure.regionprops(label_image, cache=False):
            ### get neighborhood pixels of images and labels by a given a GT point
            cur_point_mask = pred.new_zeros(bs, c, feat_h, feat_w)
            cur_point_mask[0, 0, int(region.centroid[0]), int(region.centroid[1])] = 1
            nbr_mask = ((F.conv2d(cur_point_mask, weight=torch.ones(1,1,background_length,background_length).to(gt_mask.device), stride=1, padding=background_length//2))>0).float()
            targets_mask = ((F.conv2d(cur_point_mask, weight=torch.ones(1,1,target_length,target_length).to(gt_masks.device), stride=1, padding=target_length//2))>0).float()
            pred_nbr = pred.clone() * nbr_mask
            gt_mask_nbr = gt_mask.clone() * nbr_mask
            
            ### Candidate Pixels Extraction
            max_limitation = size[0] * size[1] * 0.0015
            adaptive_thresh = (pred_nbr ).max()* (thresh_Tb + thresh_k*(1-thresh_Tb)*((len(region.coords)/max_limitation)).to(gt_mask.device))
            thresh_mask = (pred_nbr > adaptive_thresh).float()
            
            ### False Alarm Elimination
            label_image = measure.label((thresh_mask[0,:,:,:]>0).cpu())
            if label_image.max() > 1:
                for num_cur in range(label_image.max()):
                    curr_mask = thresh_mask * torch.tensor(label_image == (num_cur + 1)).float().unsqueeze(0).to(gt_mask.device) 
                    if (curr_mask * targets_mask).sum() == 0:
                        thresh_mask = thresh_mask - curr_mask
            
            ### Average Weighted Summation
            updated_gt_mask_nbr = gt_mask_nbr * (1-thresh_mask) + ((gt_mask_nbr  + pred_nbr)/2)* thresh_mask
            updated_gt_mask = torch.max(updated_gt_mask, updated_gt_mask_nbr)
        
        ### Ensure initial GT point label
        updated_gt_mask = torch.max(updated_gt_mask, (gt_mask==1).float())

        return updated_gt_mask

