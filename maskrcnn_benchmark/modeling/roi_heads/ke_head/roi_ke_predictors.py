# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d

from maskrcnn_benchmark import layers

class KERCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(KERCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_KE_HEAD.CONV_LAYERS[-1]

        self.resol = cfg.MODEL.ROI_KE_HEAD.RESOLUTION

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        assert(cfg.MODEL.ROI_KE_HEAD.NUM_KES%2==0), 'require plural but got {}'.format(str(cfg.MODEL.ROI_KE_HEAD.NUM_KES))
        NumPred = int(cfg.MODEL.ROI_KE_HEAD.NUM_KES/2 + 2)

        
        self.ke_input_xy = Conv2d(num_inputs, num_inputs, 1, 1, 0)
        nn.init.kaiming_normal_(self.ke_input_xy.weight,
                mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.ke_input_xy.bias, 0)

        self.conv5_ke_xy = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        nn.init.kaiming_normal_(self.conv5_ke_xy.weight,
                mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv5_ke_xy.bias, 0)


        self.conv5_ke_x_shrink = Conv2d(dim_reduced, NumPred, (self.resol, 1), 1, 0) # H W
        nn.init.kaiming_normal_(self.conv5_ke_x_shrink.weight,
                mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv5_ke_x_shrink.bias, 0)

        self.conv5_ke_y_shrink = Conv2d(dim_reduced, NumPred, (1, self.resol), 1, 0) # H W
        nn.init.kaiming_normal_(self.conv5_ke_y_shrink.weight,
                mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv5_ke_y_shrink.bias, 0)

        # mt branch
        self.cat_trans = Conv2d(dim_reduced, cfg.MODEL.ROI_KE_HEAD.NUM_KES, 1, 1, 0)
        nn.init.kaiming_normal_(self.cat_trans.weight,
                mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.cat_trans.bias, 0)

        self.mty = Conv2d(cfg.MODEL.ROI_KE_HEAD.NUM_KES, cfg.MODEL.ROI_KE_HEAD.NUM_MATCHTYPE, (int(self.resol/2), int(self.resol/2)), 1, 0)
        nn.init.kaiming_normal_(self.mty.weight,
                mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.mty.bias, 0)

        self.up_scale=2

    def forward(self, ft):
        ft = self.ke_input_xy(ft)
        ft = self.conv5_ke_xy(ft)

        ft_2x = layers.interpolate(ft, scale_factor=self.up_scale, mode='bilinear', align_corners=True)
        x = self.conv5_ke_x_shrink(ft_2x)

        y = self.conv5_ke_y_shrink(ft_2x)

        assert(x.size()[2:] == torch.Size([1, self.resol]) and \
                y.size()[2:] == torch.Size([self.resol, 1])), "x y: " +str(x.size())+'  '+str(y.size()) 

        # mty
        # mty = torch.cat((x_tc, y_tc), dim=1)
        mty = self.cat_trans(ft)
        mty = self.mty(mty)
        assert(mty.size()[1:] == torch.Size([24,1,1])), "mty w h should be 1, but got {}".format(str(mty.size()))

        return x, y, mty


_ROI_KE_PREDICTOR = {"KERCNNC4Predictor": KERCNNC4Predictor}


def make_roi_ke_predictor(cfg):
    func = _ROI_KE_PREDICTOR[cfg.MODEL.ROI_KE_HEAD.PREDICTOR]
    return func(cfg)
