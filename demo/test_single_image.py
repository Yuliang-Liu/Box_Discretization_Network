# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import os

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import glob
import time

import random

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--img",
        type=str,
        help="path to the target image",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="path to the target image",
        default=None,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    if not ((os.path.exists(args.img) or os.path.isdir(args.img))):
        assert(0), "Image or Dir: {} not found.".format(args.img)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    if os.path.isdir(args.img):
        imgset = []
        imgset.extend(glob.glob(os.path.join(args.img, '*.jpg')))
        imgset.extend(glob.glob(os.path.join(args.img, '*.JPG')))
        imgset.extend(glob.glob(os.path.join(args.img, '*.png')))
        imgset.extend(glob.glob(os.path.join(args.img, '*.PNG')))
        random.shuffle(imgset) 
        for iximg, img in enumerate(imgset):
            start_time = time.time()
            imgr = cv2.imread(img)
            composite = coco_demo.run_on_opencv_image(imgr)
            print("Time: {:.2f} s / img".format(time.time() - start_time))
            composite = cv2.resize(composite, (1280, 720))
            cv2.imwrite(args.output_dir+os.path.basename(img), composite)
    else:
        start_time = time.time()
        img = cv2.imread(args.img)
        composite = coco_demo.run_on_opencv_image(img)
        print("Time: {:.2f} s / img".format(time.time() - start_time))
        composite = cv2.resize(composite, (1280, 720))
        cv2.imshow("COCO detections", composite)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
