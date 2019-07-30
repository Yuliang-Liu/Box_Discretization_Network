// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "SigmoidFocalLoss.h"
#include "dcn_v2.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
  m.def("sigmoid_focalloss_forward", &SigmoidFocalLoss_forward, "SigmoidFocalLoss_forward");
  m.def("sigmoid_focalloss_backward", &SigmoidFocalLoss_backward, "SigmoidFocalLoss_backward");
  m.def("dcn_v2_forward", &dcn_v2_forward, "dcn_v2_forward");
  m.def("dcn_v2_backward", &dcn_v2_backward, "dcn_v2_backward");
  m.def("dcn_v2_psroi_pooling_forward", &dcn_v2_psroi_pooling_forward, "dcn_v2_psroi_pooling_forward");
  m.def("dcn_v2_psroi_pooling_backward", &dcn_v2_psroi_pooling_backward, "dcn_v2_psroi_pooling_backward");
}
