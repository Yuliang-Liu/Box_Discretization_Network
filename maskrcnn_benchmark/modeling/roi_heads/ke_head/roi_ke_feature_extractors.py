# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

# from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3


class KERCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(KERCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_KE_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_KE_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_KE_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            deformable=cfg.MODEL.RESNETS.DEFORM_POOLING
        )
        input_size = in_channels
        self.pooler = pooler

        layers = cfg.MODEL.ROI_KE_HEAD.CONV_LAYERS
        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "ke_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x


_ROI_KE_FEATURE_EXTRACTORS = {
    "KERCNNFPNFeatureExtractor": KERCNNFPNFeatureExtractor,
}


def make_roi_ke_feature_extractor(cfg, in_channels):
    func = _ROI_KE_FEATURE_EXTRACTORS[cfg.MODEL.ROI_KE_HEAD.FEATURE_EXTRACTOR]
    return func(cfg, in_channels)
