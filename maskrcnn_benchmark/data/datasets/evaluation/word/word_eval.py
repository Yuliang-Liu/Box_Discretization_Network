import logging
import tempfile
import os
import torch
from collections import OrderedDict
import itertools
from tqdm import tqdm

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.modeling.roi_heads.ke_head.inference import KEer
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

from maskrcnn_benchmark.config import cfg
from shapely.geometry import *
import cv2
import numpy as np


def do_coco_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    if box_only:
        logger.info("Evaluating bbox proposals")
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        res = COCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(
                    predictions, dataset, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res.results["box_proposal"][key] = stats["ar"].item()
        logger.info(res)
        check_expected_results(res, expected_results, expected_results_sigma_tol)
        if output_folder:
            torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
        return
    logger.info("Preparing results for COCO format")
    coco_results = {}
    if "bbox" in iou_types:
        logger.info("Preparing bbox results")
        coco_results["bbox"] = prepare_for_coco_detection(predictions, dataset)
    if "segm" in iou_types:
        logger.info("Preparing segm results")
        coco_results["segm"] = prepare_for_coco_segmentation(predictions, dataset)
    if "kes" in iou_types:
        logger.info("Preparing kes results")
        coco_results["kes"] = prepare_for_kes(predictions, dataset)

    logger.info("Do not apply evaluating predictions")
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                if not os.path.isdir(output_folder):
                    print('creating dir: '+output_folder)
                    os.mkdir(output_folder)
                file_path = os.path.join(output_folder, iou_type + ".json")
            res = evaluate_predictions_on_coco(
                dataset.coco, coco_results[iou_type], file_path, iou_type
            )

    return None


def prepare_for_coco_detection(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

def contour_to_xys(cnt, image_shape):
    """Convert rect to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid, i.e., within image area
    """
    rect = cv2.minAreaRect(cnt)
    h, w = image_shape[0:2]
    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x
    
    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y
    
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    for i_xy, (x, y) in enumerate(points):
        x = get_valid_x(x)
        y = get_valid_y(y)
        points[i_xy, :] = [x, y]
    points = np.reshape(points, -1)
    return points

def mask_to_roRect(mask, img_shape):
    ## convert mask into rotated rect
    e = mask[0, :, :]
    countours, hier = cv2.findContours(e.clone().numpy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Aarlog

    if len(countours) == 0:
        return np.zeros((1,8))
    t_c = countours[0].copy()

    quad = contour_to_xys(t_c, img_shape)
    return quad


def prepare_for_coco_segmentation(predictions, dataset):
    import pycocotools.mask as mask_util
    import numpy as np

    masker = Masker(threshold=0.5, padding=1)
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in tqdm(enumerate(predictions)):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")

        # Masker is necessary only if masks haven't been already resized.
        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
            masks = masks[0]

        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        rects = [mask_to_roRect(mask, [image_height, image_width]) for mask in masks]

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        esd = []
        for k, rect in enumerate(rects):
            if rect.all() == 0:
                continue
            else:
                esd.append(
                        {
                            "image_id": original_id,
                            "category_id": mapped_labels[k],
                            "seg_rorect": rect.tolist(),
                            "score": scores[k],
                        }
                )
        coco_results.extend(esd)

    return coco_results

def ke_to_quad(ke, mty, img_shape):
    mt = mty[:].argmax()
    quad = paraToQuad_v3(ke, mt)
    return quad

# polynms
def py_cpu_pnms(dets, scores, thresh):
    pts = []
    for det in dets:
        pts.append([[det[i][0], det[i][1]] for i in range(len(det))])
    order = scores.argsort()[::-1]
    areas = np.zeros(scores.shape)
    order = scores.argsort()[::-1]
    inter_areas = np.zeros((scores.shape[0], scores.shape[0]))
    for il in range(len(pts)):
        poly = Polygon(pts[il])
        areas[il] = poly.area 
        for jl in range(il, len(pts)):
            polyj = Polygon(pts[jl])
            try:
                inS = poly.intersection(polyj) 
            except:
                print(poly, polyj)
            inter_areas[il][jl] = inS.area
            inter_areas[jl][il] = inS.area

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = inter_areas[i][order[1:]] / (areas[i] + areas[order[1:]] - inter_areas[i][order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def esd_pnms(esd, pnms_thresh):
    scores = []
    dets = []
    for ele in esd:
        score = ele['score']
        quad = ele['ke_quad']
        det = np.array([[quad[0][0], quad[0][1]], [quad[1][0], quad[1][1]],[quad[2][0], quad[2][1]],[quad[3][0], quad[3][1]]])
        scores.append(score)
        dets.append(det)
    scores = np.array(scores)
    dets = np.array(dets)
    keep = py_cpu_pnms(dets, scores, pnms_thresh)
    return keep

def prepare_for_kes(predictions, dataset):
    import numpy as np

    masker = KEer()
    coco_results = []
    for image_id, prediction in tqdm(enumerate(predictions)):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))
        kes = prediction.get_field("ke")
        kes = kes.kes.numpy()

        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()
        mtys = prediction.get_field("mty").view(-1, 24).numpy()
        quads = []
        for ik, ke in enumerate(kes):
            quads.append(ke_to_quad(ke, mtys[ik], [image_height, image_width]))

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        esd = []
        for k, quad in enumerate(quads):
            if not quad:
                continue
            else:
                esd.append(
                        {
                            "image_id": original_id,
                            "category_id": mapped_labels[k],
                            # "segmentation": rle,
                            "ke_quad": quad,
                            "score": scores[k],
                        }
                )
        # chan 2019/4/11
        if cfg.PROCESS.PNMS:
            pnms_thresh = cfg.PROCESS.NMS_THRESH
            keep = esd_pnms(esd, pnms_thresh)        
            new_esd = []
            for i in keep:
                new_esd.append(esd[i])
            coco_results.extend(new_esd)
        else:
            coco_results.extend(esd)

    return coco_results

# inspired from Detectron
def evaluate_box_proposals(
    predictions, dataset, thresholds=None, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction.get_field("objectness").sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def evaluate_predictions_on_coco(
    coco_gt, coco_results, json_result_file, iou_type="bbox"
):
    import json

    print('writing results to '+json_result_file)
    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    # from pycocotools.cocoeval import COCOeval

    # coco_dt = coco_gt.loadRes(str(json_result_file))
    # # coco_dt = coco_gt.loadRes(coco_results)
    # coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()
    # return coco_eval
    return None


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoint": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)


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

