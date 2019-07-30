import torch
from torch import nn

import pdb, os
from shapely.geometry import *
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

import random
import string

all_types = [[1,2,3,4],[1,2,4,3],[1,3,2,4],[1,3,4,2],[1,4,2,3],[1,4,3,2],\
                    [2,1,3,4],[2,1,4,3],[2,3,1,4],[2,3,4,1],[2,4,1,3],[2,4,3,1],\
                    [3,1,2,4],[3,1,4,2],[3,2,1,4],[3,2,4,1],[3,4,1,2],[3,4,2,1],\
                    [4,1,2,3],[4,1,3,2],[4,2,1,3],[4,2,3,1],[4,3,1,2],[4,3,2,1]]


class kePostProcessor(nn.Module):
    def __init__(self, keer=None, cfg=None):
        super(kePostProcessor, self).__init__()
        self.keer = keer
        self.cfg = cfg

    def forward(self, ft_x, ft_y, mty, boxes):
        ke_prob_x = ft_x
        ke_prob_y = ft_y
        mty_prob = mty

        boxes_per_image = [box.bbox.size(0) for box in boxes]
        ke_prob_x = ke_prob_x.split(boxes_per_image, dim=0)
        ke_prob_y = ke_prob_y.split(boxes_per_image, dim=0)
        mty_prob = mty_prob.split(boxes_per_image, dim=0)

        results = []
        for prob_x, prob_y, prob_mty, box in zip(ke_prob_x, ke_prob_y, mty_prob, boxes):
            bbox = BoxList(box.bbox, box.size, mode='xyxy')
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))

            if self.keer: 
                prob_x, rescores_x = self.keer(prob_x, box)
                prob_y, rescores_y = self.keer(prob_y, box)
                rescores = (rescores_x+rescores_y)*0.5
            
            if self.cfg.MODEL.ROI_KE_HEAD.RESCORING:
                bbox.add_field('scores', rescores)
                
            prob = torch.cat((prob_x,prob_y), dim = -2)
            prob = prob[..., :1]
            prob = textKES(prob, box.size)
            bbox.add_field('ke', prob)
            bbox.add_field('mty', prob_mty)
            results.append(bbox)

        return results


# TODO remove and use only the keer
import numpy as np
import cv2

def scores_to_probs(scores):
    """Transforms CxHxW of scores to probabilities spatially."""
    channels = scores.shape[0]
    for c in range(channels):
        temp = scores[c, :, :]
        max_score = temp.max()
        temp = np.exp(temp - max_score) / np.sum(np.exp(temp - max_score))
        scores[c, :, :] = temp
    return scores

def kes_decode(kes):
    # BDN decode 
    for ix, i in enumerate(kes):
        mnd = i[0, 0]
        nkes = i.shape[1]-2
        kes[ix][0, 1:5] = kes[ix][0, 1:5]*2 - mnd
    return kes

def heatmaps_to_kes(maps, rois, scores, cfg):
    """Extract predicted ke locations from heatmaps. Output has shape
    (#rois, 4, #kes) with the 4 rows corresponding to (x, y, logit, prob)
    for each ke.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous ke coordinate. We maintain
    # consistency with kes_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = np.maximum(widths, 1)
    heights = np.maximum(heights, 1)
    widths_ceil = np.ceil(widths)
    heights_ceil = np.ceil(heights)

    
    resol = cfg.MODEL.ROI_KE_HEAD.RESOLUTION # cfg.mo... 56

    if maps.shape[-2:] == (1, resol):
        xory_mode = 0 # x mode
    elif maps.shape[-2:] == (resol, 1):
        xory_mode = 1 # y mode
    else:
        assert(0), 'invalid mode.'

    # print("maps", maps.shape, maps[0,0], maps[0,1])
    # NCHW to NHWC for use with OpenCV
    maps = np.transpose(maps, [0, 2, 3, 1])
    min_size = 0 # cfg
    num_kes = int(cfg.MODEL.ROI_KE_HEAD.NUM_KES/2)+2
    d_preds = np.zeros(
        (len(rois), 2, num_kes), dtype=np.float32)
    d_scores = np.zeros(scores.shape,  dtype=np.float32)

    assert(len(rois) == maps.shape[0]), 'shape mismatch {}, {}, {}, {}'.format(str(len(rois)), \
                                                                    str(rois.shape), \
                                                                    str(maps.shape[0]), \
                                                                    str(maps.shape))

    normal = 0
    innormal = 0                                                                    
    for i in range(len(rois)):
        if min_size > 0:
            roi_map_width = int(np.maximum(widths_ceil[i], min_size))
            roi_map_height = int(np.maximum(heights_ceil[i], min_size))
        else:
            roi_map_width = widths_ceil[i]
            roi_map_height = heights_ceil[i]
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height

        np.set_printoptions(suppress=True)
        # print(i, "stop", maps.shape, np.around(maps[i][0, :, :], decimals=2))

        if not xory_mode:
            roi_map = cv2.resize(
                maps[i], (roi_map_width, 1), interpolation=cv2.INTER_CUBIC)
        else:
            roi_map = cv2.resize(
                maps[i], (1, roi_map_height), interpolation=cv2.INTER_CUBIC)

        # print(roi_map.shape, np.around(roi_map[0, :, :], decimals=2))
        # Bring back to CHW
        roi_map = np.transpose(roi_map, [2, 0, 1])
        roi_map_probs = scores_to_probs(roi_map.copy())
        
        # kescore visulize.
        map_vis = np.transpose(maps[i], [2, 0, 1])
        map_vis = scores_to_probs(map_vis.copy())   

        sum_score = []

        if cfg.MODEL.ROI_KE_HEAD.RESCORING:
            for k in range(num_kes):
                if map_vis[k].shape[0] == 1:
                    x = np.arange(0, len(map_vis[k][0]), 1)  
                    y = map_vis[k][0]
                else:
                    x = np.arange(0, len(map_vis[k][:, 0]), 1)  
                    y = map_vis[k][:, 0]

                top = y.max()
                atop = y.argmax()

                # lf2&1
                lf2 = max(atop-2, 0)
                lf1 = max(atop-1, 0)
                rt2 = min(atop+2, 55)
                rt1 = min(atop+1, 55)

                sum_score.append(top+y[lf2]+y[lf1]+y[rt1]+y[rt2])

                kes_score_mean = sum(sum_score)*1.0/len(sum_score)
                gama = cfg.MODEL.ROI_KE_HEAD.RESCORING_GAMA
                final_score = (scores[i]*(2.0-gama)+gama*kes_score_mean)*0.5

                # rescore 
                d_scores[i] = final_score
        else:
            d_scores[i] = scores[i]

        w = roi_map.shape[2]
        for k in range(num_kes):
            pos = roi_map[k, :, :].argmax()
            x_int = pos % w
            y_int = (pos - x_int) // w
            assert (roi_map_probs[k, y_int, x_int] ==
                    roi_map_probs[k, :, :].max())
            x = (x_int + 0.5) * width_correction
            y = (y_int + 0.5) * height_correction
            if not xory_mode:
                d_preds[i, 0, k] = x + offset_x[i]
                d_preds[i, 1, k] = roi_map_probs[k, y_int, x_int]
            else:
                d_preds[i, 0, k] = y + offset_y[i]
                d_preds[i, 1, k] = roi_map_probs[k, y_int, x_int]


    out_kes_d = kes_decode(d_preds)
    return np.transpose(out_kes_d, [0, 2, 1]), d_scores

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.ke import textKES
class KEer(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """
    def __init__(self, padding=0, cfg =None):
        self.padding = padding
        self.cfg =cfg

    def compute_flow_field_cpu(self, boxes):
        im_w, im_h = boxes.size
        boxes_data = boxes.bbox
        num_boxes = len(boxes_data)
        device = boxes_data.device

        TO_REMOVE = 1
        boxes_data = boxes_data.int()
        box_widths = boxes_data[:, 2] - boxes_data[:, 0] + TO_REMOVE
        box_heights = boxes_data[:, 3] - boxes_data[:, 1] + TO_REMOVE

        box_widths.clamp_(min=1)
        box_heights.clamp_(min=1)

        boxes_data = boxes_data.tolist()
        box_widths = box_widths.tolist()
        box_heights = box_heights.tolist()

        flow_field = torch.full((num_boxes, im_h, im_w, 2), -2)

        # TODO maybe optimize to make it GPU-friendly with advanced indexing
        # or dedicated kernel
        for i in range(num_boxes):
            w = box_widths[i]
            h = box_heights[i]
            if w < 2 or h < 2:
                continue
            x = torch.linspace(-1, 1, w)
            y = torch.linspace(-1, 1, h)
            # meshogrid
            x = x[None, :].expand(h, w)
            y = y[:, None].expand(h, w)

            b = boxes_data[i]
            x_0 = max(b[0], 0)
            x_1 = min(b[2] + 0, im_w)
            y_0 = max(b[1], 0)
            y_1 = min(b[3] + 0, im_h)
            flow_field[i, y_0:y_1, x_0:x_1, 0] = x[(y_0 - b[1]):(y_1 - b[1]),(x_0 - b[0]):(x_1 - b[0])]
            flow_field[i, y_0:y_1, x_0:x_1, 1] = y[(y_0 - b[1]):(y_1 - b[1]),(x_0 - b[0]):(x_1 - b[0])]

        return flow_field.to(device)

    def compute_flow_field(self, boxes):
        return self.compute_flow_field_cpu(boxes)

    # TODO make it work better for batches
    def forward_single_image(self, masks, boxes):
        boxes = boxes.convert('xyxy')
        if self.padding:
            boxes = BoxList(boxes.bbox.clone(), boxes.size, boxes.mode)
            masks, scale = expand_masks(masks, self.padding)
            boxes.bbox = expand_boxes(boxes.bbox, scale)

        flow_field = self.compute_flow_field(boxes)
        result = torch.nn.functional.grid_sample(masks, flow_field)
        return result

    def to_points(self, masks):
        height, width = masks.shape[-2:]
        m = masks.view(masks.shape[:2] + (-1,))
        scores, pos = m.max(-1)
        x_int = pos % width
        y_int = (pos - x_int) // width

        result = torch.stack([x_int.float(), y_int.float(), torch.ones_like(x_int, dtype=torch.float32)], dim=2)
        return result

    def __call__(self, masks, boxes):
        # TODO do this properly
        if isinstance(boxes, BoxList):
            boxes = [boxes]

        if isinstance(masks, list):
            masks = torch.stack(masks, dim=0)
            assert(len(masks.size()) == 4)

        scores = boxes[0].get_field("scores")

        result, rescores = heatmaps_to_kes(masks.detach().cpu().numpy(), boxes[0].bbox.cpu().numpy(), scores.cpu().numpy(), self.cfg)
        return torch.from_numpy(result).to(masks.device), torch.from_numpy(rescores).to(masks.device)


def make_roi_ke_post_processor(cfg):
    if cfg.MODEL.ROI_KE_HEAD.POSTPROCESS_KES:
        keer = KEer(padding=0, cfg=cfg)
    else:
        keer = None
    ke_post_processor = kePostProcessor(keer,cfg)
    return ke_post_processor
