# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat

from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)

from maskrcnn_benchmark.structures.ke import kes_to_heat_map

import os, time

DEBUG = 0

def project_kes_to_heatmap(kes, mty, proposals, discretization_size):
    proposals = proposals.convert('xyxy')
    out_x, out_y, valid_x, valid_y, out_mty, valid_mty = kes_to_heat_map(kes.kes_x, kes.kes_y, mty.mty, proposals.bbox, discretization_size)
    return out_x, out_y, valid_x, valid_y, out_mty, valid_mty

def _within_box(points_x, points_y, boxes):
    """Validate which kes are contained inside a given box.
    points: NxKx2
    boxes: Nx4
    output: NxK
    """
    x_within = (points_x[..., :, 0] >= boxes[:, 0, None]) & (points_x[..., :, 0] <= boxes[:, 2, None])
    y_within = (points_y[..., :, 0] >= boxes[:, 1, None]) & (points_y[..., :, 0] <= boxes[:, 3, None])
    return x_within & y_within

_TOTAL_SKIPPED = 0

class KERCNNLossComputation(object):
    def __init__(self, proposal_matcher, fg_bg_sampler, discretization_size, cfg):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.discretization_size = discretization_size
        self.cfg = cfg.clone()

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # KE RCNN needs "labels" and "kes "fields for creating the targets
        target = target.copy_with_fields(["labels", "kes", "mty"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_ke_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        kes = []
        mty = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_ke_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            # TODO check if this is the right one, as BELOW_THRESHOLD
            neg_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[neg_inds] = 0

            kes_per_image = matched_targets.get_field('kes')
            mty_per_image = matched_targets.get_field('mty')
            # TODO remove conditional  when better support for zero-dim is in
 
            within_box = _within_box(kes_per_image.kes_x, kes_per_image.kes_y, matched_targets.bbox)
            is_visible = (within_box).sum(1) > 0

            labels_per_image[~is_visible] = -1

            labels.append(labels_per_image)
            kes.append(kes_per_image)
            mty.append(mty_per_image)

        return labels, kes, mty

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, kes, mty = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, kes_per_image, mty_per_image, proposals_per_image in zip(
            labels, kes, mty, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field("kes", kes_per_image)
            proposals_per_image.add_field("mty", mty_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            # img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            img_sampled_inds = torch.nonzero(pos_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, proposals, ke_logits_x, ke_logits_y, ke_logits_mty):
        heatmaps_x = []
        heatmaps_y = []
        valid_x = []
        valid_y = []
        mtyS = []
        valid_mty = []
        for proposals_per_image in proposals:
            ke = proposals_per_image.get_field("kes")
            mty = proposals_per_image.get_field("mty")
            heatmaps_per_image_x, heatmaps_per_image_y, valid_per_image_x, valid_per_image_y, heatmaps_per_image_mty, valid_per_image_mty = \
                        project_kes_to_heatmap(ke, mty, proposals_per_image, self.discretization_size)
            heatmaps_x.append(heatmaps_per_image_x.view(-1))
            heatmaps_y.append(heatmaps_per_image_y.view(-1))
            valid_x.append(valid_per_image_x.view(-1))
            valid_y.append(valid_per_image_y.view(-1))
            valid_mty.append(valid_per_image_mty.view(-1))
            mtyS.append(heatmaps_per_image_mty.view(-1))

        ke_targets_x = cat(heatmaps_x, dim=0)
        ke_targets_y = cat(heatmaps_y, dim=0)
        valid_x = cat(valid_x, dim=0).to(dtype=torch.uint8)
        valid_x = torch.nonzero(valid_x).squeeze(1)
        valid_y = cat(valid_y, dim=0).to(dtype=torch.uint8)
        valid_y = torch.nonzero(valid_y).squeeze(1)
        valid_mty = cat(valid_mty, dim=0).to(dtype=torch.uint8)
        valid_mty = torch.nonzero(valid_mty).squeeze(1)
        ke_targets_mty = cat(mtyS, dim=0)

        MIN_KE_COUNT_FOR_VALID_MINIBATCH = 6
        num_valid_x = len(valid_x)
        num_valid_y = len(valid_y)

        # torch.mean (in binary_cross_entropy_with_logits) does'nt
        # accept empty tensors, so handle it sepaartely
        out_loss_x = 0
        out_loss_y = 0
        out_loss_mty = 0
        if ke_targets_x.numel() == 0 or valid_x.numel() == 0 or num_valid_x <= MIN_KE_COUNT_FOR_VALID_MINIBATCH or\
            ke_targets_y.numel() == 0 or valid_y.numel() == 0 or num_valid_y <= MIN_KE_COUNT_FOR_VALID_MINIBATCH:
            global _TOTAL_SKIPPED
            _TOTAL_SKIPPED += 1
            print("Non valid, skipping {}", _TOTAL_SKIPPED)
            out_loss_x = ke_logits_x.sum() * 0
            out_loss_y = ke_logits_y.sum() * 0
            out_loss_mty = ke_logits_mty.sum() * 0
            return (out_loss_x+out_loss_y, out_loss_mty) 

        N_x, K_x, H_x, W_x = ke_logits_x.shape
        ke_logits_x = ke_logits_x.view(N_x * K_x, H_x * W_x)

        N_y, K_y, H_y, W_y = ke_logits_y.shape
        ke_logits_y = ke_logits_y.view(N_y * K_y, H_y * W_y)

        ke_loss_x = self.cfg.MODEL.ROI_KE_HEAD.KE_WEIGHT * F.cross_entropy(
                ke_logits_x[valid_x], ke_targets_x[valid_x])
        ke_loss_y = self.cfg.MODEL.ROI_KE_HEAD.KE_WEIGHT * F.cross_entropy(
                ke_logits_y[valid_y], ke_targets_y[valid_y])


        N_mty, K_mty, H_mty, W_mty = ke_logits_mty.shape
        ke_logits_mty = ke_logits_mty.view(N_mty, K_mty)

        if ke_logits_mty.numel() == 0 or valid_mty.numel() == 0 :
            ke_loss_mty = ke_logits_mty.sum() * 0
        else:
            ke_loss_mty = self.cfg.MODEL.ROI_KE_HEAD.MTY_WEIGHT * F.cross_entropy(
                    ke_logits_mty, ke_targets_mty)

        ke_loss = ke_loss_x + ke_loss_y 
        mty_loss = ke_loss_mty

        return ke_loss, mty_loss

def make_roi_ke_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    loss_evaluator = KERCNNLossComputation(
        matcher, fg_bg_sampler, cfg.MODEL.ROI_KE_HEAD.RESOLUTION, cfg
    )

    return loss_evaluator
