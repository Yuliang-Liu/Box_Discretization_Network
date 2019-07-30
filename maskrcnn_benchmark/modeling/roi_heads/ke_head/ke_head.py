# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .roi_ke_feature_extractors import make_roi_ke_feature_extractor
from .roi_ke_predictors import make_roi_ke_predictor
from .inference import make_roi_ke_post_processor
from .loss import make_roi_ke_loss_evaluator


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class ROIKEHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIKEHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_ke_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_ke_predictor(cfg)
        self.post_processor = make_roi_ke_post_processor(cfg)
        self.loss_evaluator = make_roi_ke_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        x = self.feature_extractor(features, proposals)
        ke_outputs_x, ke_outputs_y, mty = self.predictor(x)

        if not self.training:
            result = self.post_processor(ke_outputs_x, ke_outputs_y, mty, proposals)
            return x, result, {}, {}

        loss_ke, loss_mty = self.loss_evaluator(proposals, ke_outputs_x, ke_outputs_y, mty)

        return x, proposals, dict(loss_ke=loss_ke), dict(loss_mty=loss_mty)


def build_roi_ke_head(cfg, in_channels):
    return ROIKEHead(cfg, in_channels)
