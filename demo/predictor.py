# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
from torchvision import transforms as T

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util

from maskrcnn_benchmark.modeling.roi_heads.ke_head.inference import KEer

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import *
import copy
import itertools

def paraToQuad_v3(kes, mt):
    ms = (kes[0,0], kes[6,0])
    xs = [kes[i,0] for i in range(1,5)]
    ys = [kes[i,0] for i in range(7,11)]
    crs = (kes[5,0], kes[11,0])
    ms = Point(ms)
    crs = Point(crs)
    vp = []
    all_types = [[1,2,3,4],[1,2,4,3],[1,3,2,4],[1,3,4,2],[1,4,2,3],[1,4,3,2],\
                 [2,1,3,4],[2,1,4,3],[2,3,1,4],[2,3,4,1],[2,4,1,3],[2,4,3,1],\
                 [3,1,2,4],[3,1,4,2],[3,2,1,4],[3,2,4,1],[3,4,1,2],[3,4,2,1],\
                 [4,1,2,3],[4,1,3,2],[4,2,1,3],[4,2,3,1],[4,3,1,2],[4,3,2,1]]
    all_types = [[all_types[iat][0]-1,all_types[iat][1]-1,all_types[iat][2]-1,all_types[iat][3]-1] for iat in range(24)]

    tpe = all_types[mt]
    p1 = Point((xs[0], ys[tpe[0]]))
    p2 = Point((xs[1], ys[tpe[1]]))
    p3 = Point((xs[2], ys[tpe[2]]))
    p4 = Point((xs[3], ys[tpe[3]]))
    pts = [p1,p2,p3,p4]
    scs = [0,1,2,3]
    for it in itertools.permutations(scs, 4):
        poly =Polygon([(pts[it[0]].x, pts[it[0]].y), (pts[it[1]].x, pts[it[1]].y), \
                        (pts[it[2]].x, pts[it[2]].y), (pts[it[3]].x, pts[it[3]].y)])
        if poly.is_valid and ms.within(poly) and crs.within(poly):
            quad = [(pts[it[0]].x, pts[it[0]].y), (pts[it[1]].x, pts[it[1]].y), \
                        (pts[it[2]].x, pts[it[2]].y), (pts[it[3]].x, pts[it[3]].y)]
            lr = LinearRing(quad)
            if lr.is_ccw:
                return [(int(iq[0]), int(iq[1])) for iq in quad]
            else:
                quad =  [quad[0], quad[3], quad[2], quad[1]]
                return [(int(iq[0]), int(iq[1])) for iq in quad]

            return [(int(iq[0]), int(iq[1])) for iq in quad]

    return None

def paraToQuad_v2(kps, roRect, maskPoly):
    ms = (kps[0,0], kps[1,0])
    xs = kps[0, 1:5]
    xs = xs*2-ms[0]
    ys = kps[1, 5:9]
    ys = ys*2-ms[1]
    crs = (kps[0,9], kps[1,9])
    ms = Point(ms)
    crs = Point(crs)
    roRectpoly = Polygon([(roRect[0], roRect[1]), (roRect[2], roRect[3]), (roRect[4], roRect[5]), (roRect[6], roRect[7])]) # mask rect
    assert(maskPoly.shape[2] == 2), 'x y correspond'
    maskpts = [(maskPoly[i][0][0], maskPoly[i][0][1]) for i in xrange(maskPoly.shape[0])]
    try:
        mask_poly = Polygon(maskpts) # mask rect
    except:
        return None
    if not ((ms.within(mask_poly) and crs.within(mask_poly)) or (roRectpoly.is_valid and ms.within(roRectpoly) and crs.within(roRectpoly))):
        return None
    idx = range(4)
    valid_polys = []
    vp = []
    for i1 in xrange(4):
        for i2 in xrange(4):
            for i3 in xrange(4):
                if i2 != i1 != i3:
                    i4 = [idx[i] for i in xrange(4) if not i in [i1,i2,i3]]
                    vas1s2 = np.array([0, 0])
                    try:
                        poly_s1 = Polygon([(xs[0], ys[i1]), (xs[1], ys[i2]), (xs[2], ys[i3]), (xs[3], ys[i4])])
                    except Exception as e:
                        vas1s2[0] = 1
                    try:
                        poly_s2 = Polygon([(xs[0], ys[i1]), (xs[1], ys[i2]), (xs[3], ys[i3]), (xs[2], ys[i4])])
                    except Exception as e:
                        vas1s2[1] = 1
                    if vas1s2.any() == 1:
                        continue
                    if poly_s1.is_valid and ms.within(poly_s1) and crs.within(poly_s1):
                        valid_polys.append(poly_s1)
                        vp.append([(xs[0], ys[i1]), (xs[1], ys[i2]), (xs[2], ys[i3]), (xs[3], ys[i4])])
                    if poly_s2.is_valid and ms.within(poly_s2) and crs.within(poly_s2):
                        valid_polys.append(poly_s2)
                        vp.append([(xs[0], ys[i1]), (xs[1], ys[i2]), (xs[3], ys[i3]), (xs[2], ys[i4])])

    valid_intersect_areas =[]
    if valid_polys:
        for ixp, ip in enumerate(valid_polys):
            is_calmask = 1
            try:
                ip_ins_area = np.abs(ip.intersection(mask_poly).area)
                ip_un_area = np.abs(ip.union(mask_poly).area)
            except:
                ip_ins_area = np.abs(ip.intersection(roRectpoly).area)
                ip_un_area = np.abs(ip.union(roRectpoly).area)
                is_calmask = 0

            if ip_un_area>1e-5:
                if is_calmask:
                    valid_intersect_areas.append(ip_ins_area/ip_un_area + ip_ins_area/mask_poly.area)
                else:
                    valid_intersect_areas.append(ip_ins_area/ip_un_area)
                # valid_intersect_areas.append(ip_ins_area/(mask_poly.area+ip.area) + ip_ins_area/mask_poly.area) # next item is necessary
            else:
                valid_intersect_areas.append(0) # discard this poly
        via = np.array(valid_intersect_areas)
        # ================ change ccw ==================
        lr_checkCCW = LinearRing(vp[via.argmax()])
        if lr_checkCCW.is_ccw:
            return vp[via.argmax()]
        else:
            vp[via.argmax()] =  [vp[via.argmax()][0], vp[via.argmax()][3], vp[via.argmax()][2], vp[via.argmax()][1]]
            return vp[via.argmax()]
    else:
        return None

class COCODemo(object):
    # COCO categories for pretty print
    CATEGORIES = [
        "__background",
        "text",
    ]

    def __init__(
        self,
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        self.keer =KEer()

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        result = image.copy()
        predictions = self.compute_prediction(image)

        top_predictions = self.select_top_predictions(predictions)

        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        # result = self.overlay_boxes(result, top_predictions)
        # if self.cfg.MODEL.MASK_ON:
        #     result = self.overlay_mask(result, top_predictions)
        # if self.cfg.MODEL.KEYPOINT_ON:
        #     result = self.overlay_keypoints(result, top_predictions)
        # result = self.overlay_class_names(result, top_predictions)
        if self.cfg.MODEL.KE_ON:
            result = self.overlay_ke(result, top_predictions)
        ext = 't'
        result = self.overlay_class_names(result, top_predictions, ext)

        return result

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        height, width = original_image.shape[:-1]
        with torch.no_grad():
            predictions = self.model(image_list)

        predictions = [o.to(self.cpu_device) for o in predictions]
        # always single image is passed at a time
        prediction = predictions[0]
        # reshape prediction (a BoxList) into the original image size
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)

        if prediction.has_field("ke") and self.cfg.MODEL.KE_ON:
            kes = prediction.get_field("ke")
            mty = prediction.get_field("mty")
            prediction.add_field("ke", kes.kes)
            prediction.add_field("mty", mty)

        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            # tuple(color)
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), (255,255,0), 2
            )

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite


    def scores_to_probs(self, scores):
        temp = scores[:]
        max_score = temp.max()
        temp = np.exp(temp - max_score) / np.sum(np.exp(temp - max_score))
        scores[:] = temp
        return scores

    def overlay_ke(self, image, predictions):
        kes = predictions.get_field("ke").numpy()
        labels = predictions.get_field("labels")
        mtys = predictions.get_field("mty").view(-1, 24).numpy()
        boxes = predictions.bbox
        colors = [(255,0,0), (255,128,0), (255,255,0), (0,255,0), (0,255,255), (0,0,255), (0,128,255), (128,0,255)]
        masks = predictions.get_field("mask").numpy()

        for box, ke, mty, mask  in zip(boxes, kes, mtys, masks):
            mt = mty[:].argmax()
            mt_prob = self.scores_to_probs(mty)
            quad = paraToQuad_v3(ke, mt)

            if quad:
                for i in range(4):
                    cv2.line(image, quad[i], quad[(i+1)%4], (0,0,255), 2)
            else:
                print("find none quad", ke, mt)
                thresh = mask[0, :, :, None]
                # print("predictor overlay maks2", thresh.shape)
                contours, hierarchy = cv2.findContours(
                    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                image = cv2.drawContours(image, contours, -1, (0,255,0), 2)

        composite = image

        return composite

    def overlay_keypoints(self, image, predictions):
        keypoints = predictions.get_field("keypoints")
        kps = keypoints.keypoints
        scores = keypoints.get_field("logits")
        kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
        for region in kps:
            image = vis_keypoints(image, region.transpose((1, 0)))
        return image

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions, ext):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(ext, score)
            if ext == 'c':
                cv2.putText(
                    image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1
                )
            elif ext == 't':
                cv2.putText(
                    image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1
                )

        return image

import numpy as np
import matplotlib.pyplot as plt
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints = PersonKeypoints.NAMES
    kp_lines = PersonKeypoints.CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
