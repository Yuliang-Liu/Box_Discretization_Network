# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

from maskrcnn_benchmark.structures.ke import textKES
from maskrcnn_benchmark.structures.mty import MTY
import numpy as np
from PIL import Image
from shapely.geometry import *
import cv2

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if isinstance(target, list):
            target = [t.resize(image.size) for t in target]
        else:
            target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            if isinstance(target, list):
                target = [t.transpose(0) for t in target]
            else:
                target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomCrop(object):
    """Random crop with repeatedly expanding the range to included box borders."""
    def __init__(self, prob, init_crop_size=(0.5, 1.0)):

        if (not isinstance(init_crop_size, list)) and (not isinstance(init_crop_size, tuple)):
            raise ValueError('Paremeter init_crop_size should be a list or tuple!')
        elif len(init_crop_size) != 2:
            raise ValueError('Length of init_crop_size should be 2!')
        elif not (init_crop_size[0] <= 1 and init_crop_size[0] >= 0 and init_crop_size[1] <= 1 and init_crop_size[1] >= 0):
            raise ValueError('Elements of init_crop_size should be within [0, 1]!')
        self.prob = prob
        self.init_crop_size = init_crop_size

    def __call__(self, image, target):
        if random.random() >= self.prob:
            return image, target

        if isinstance(target, list):
            target0 = target[0]
        else:
            target0 = target
        while True:
            # Initial Crop Region
            crop_region = self.initial_crop_region(image)

            # Adjust Crop Region
            crop_region, keep_target = self.adjust_crop_region(crop_region, target0)
            if crop_region is None and keep_target is None:
                continue

            if isinstance(target, list):
                # check empty char
                new_t1 = target[1].crop(crop_region)
                if len(new_t1) < 1: return image, target

            image = image.crop(crop_region.numpy())
            if isinstance(target, list):
                target0 = keep_target.crop(crop_region)
                others = [t.crop(crop_region, remove_empty=True) for t in target[1:]]
                target = [target0] + others
            else:
                target = keep_target.crop(crop_region)

            return image, target

    def initial_crop_region(self, image):
        width, height = image.size
        ratio_w, ratio_h = torch.empty(2).uniform_(self.init_crop_size[0], self.init_crop_size[1])
        crop_width, crop_height = int(width*ratio_w), int(height*ratio_h)
        crop_xmin = torch.randint(width-crop_width, (1,))
        crop_ymin = torch.randint(height-crop_height, (1,))
        crop_xmax = crop_xmin + crop_width
        crop_ymax = crop_ymin + crop_height
        crop_region = torch.Tensor([crop_xmin, crop_ymin, crop_xmax, crop_ymax])
        return crop_region

    def intersect_area(self, bbox, bboxes):
        inter_xmin = torch.max(bbox[0], bboxes[:, 0])
        inter_ymin = torch.max(bbox[1], bboxes[:, 1])
        inter_xmax = torch.min(bbox[2], bboxes[:, 2])
        inter_ymax = torch.min(bbox[3], bboxes[:, 3])
        inter_width = torch.max(torch.Tensor([0]), inter_xmax-inter_xmin)
        inter_height = torch.max(torch.Tensor([0]), inter_ymax-inter_ymin)
        return inter_width*inter_height

    def adjust_crop_region(self, crop_region, target):
        keep_indies_ = torch.zeros((len(target)), dtype=torch.uint8)
        while True:
            inter_area = self.intersect_area(crop_region, target.bbox)
            keep_indies = (inter_area > 0)
            if torch.sum(keep_indies) == 0:
                return None, None
            keep_target = target[keep_indies]
            if keep_indies.equal(keep_indies_):
                return crop_region, keep_target
            keep_bbox = keep_target.bbox
            crop_xmin = torch.min(crop_region[0], torch.min(keep_bbox[:, 0]))
            crop_ymin = torch.min(crop_region[1], torch.min(keep_bbox[:, 1]))
            crop_xmax = torch.max(crop_region[2], torch.max(keep_bbox[:, 2]))
            crop_ymax = torch.max(crop_region[3], torch.max(keep_bbox[:, 3]))
            crop_region = torch.Tensor([crop_xmin, crop_ymin, crop_xmax, crop_ymax])
            keep_indies_ = keep_indies


class RandomRotation(object):
    def __init__(self, prob = 0.3, degree = 5):
        self.prob = prob
        self.degree = degree

    def kes_encode(self, kes):
        kes_encode = []
        for i in kes:
            mnx = i[0]
            mny = i[1]
            assert(len(i)%3 == 0)
            npts = int(len(i)/3-2)
            for index in range(npts):
                i[3+index*3]  = (i[3+index*3]+mnx)/2
                i[4+index*3]  = (i[4+index*3]+mny)/2
            kes_encode.append(i)
        return kes_encode
    
    def kes_gen(self, kes):
        kes_gen_out = []
        for i in kes:
            mnx = i[0]
            mny = i[1]
            cx= i[27]
            cy= i[28]
            assert(len(i)%3 == 0)
            ot = [mnx, i[3],i[6],i[9],i[12], cx,\
                  mny, i[16],i[19],i[22],i[25], cy]
            kes_gen_out.append(ot)
        return kes_gen_out

    def __call__(self, image, target):
        if random.random() < self.prob:
            image1 = image
            target1 = target
            img = np.array(image)
            w = image.size[0]
            h = image.size[1]
            pri_points = []
            for i in range(len(target.extra_fields['masks'].instances)):
                assert(len(target.extra_fields['masks'].instances[i].polygons)==1), 'one text instance should have only one polygon.'
                tensor_box = target.extra_fields['masks'].instances[i].polygons[0].polygons

                points_x = np.array([tensor_box[0][0],tensor_box[0][2],tensor_box[0][4],tensor_box[0][6]])
                points_y = np.array([tensor_box[0][1],tensor_box[0][3],tensor_box[0][5],tensor_box[0][7]])
                smaller_x = np.where(points_x <= 0)
                larger_x = np.where(points_x >= w)
                smaller_y = np.where(points_y <= 0)
                larger_y = np.where(points_y >= h)
                points_x[smaller_x] = 1
                points_x[larger_x] = w - 1
                points_y[smaller_y] = 1
                points_y[larger_y] = h -1
                pri_points.append((int(points_x[0]),int(points_y[0])))
                pri_points.append((int(points_x[1]),int(points_y[1])))
                pri_points.append((int(points_x[2]),int(points_y[2])))
                pri_points.append((int(points_x[3]),int(points_y[3])))

            #get the transform image and points  
            height, width = img.shape[:2]
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), random.uniform(-self.degree,self.degree), 1.0)
            cos = np.abs(matrix[0,0])
            sin = np.abs(matrix[0,1])
            new_W = int((height * sin) + (width * cos))
            new_H = int((height * cos) + (width * sin))
            matrix[0,2] += (new_W/2) - width/2
            matrix[1,2] += ((new_H/2)) - height/2
            img = cv2.warpAffine(img, matrix, (new_W,new_H))

            change_points = []
            for i in range(int(len(pri_points))):
                x_r,y_r = cv2.transform(np.array([[pri_points[i]]]),matrix).squeeze()
                change_points.append([x_r,y_r])
        
            image = Image.fromarray(img)

            keypoints_len = len(change_points)
            tran_boxes = []
            n = keypoints_len/4

            for i in range(int(n)):
                tran_boxes.append(change_points[0 + i*4: 4 + i*4])

            tran_boxes = np.array(tran_boxes).reshape(-1,2)                       
            tran_x = []
            tran_y = []
            for k in range(len(tran_boxes)):
                tran_x.append(int(tran_boxes[k][0]))
                tran_y.append(int(tran_boxes[k][1]))
            max_x = max(tran_x)
            min_x = min(tran_x)
            max_y = max(tran_y)
            min_y = min(tran_x)
            ctr_x = new_W / 2
            ctr_y = new_H / 2
            origin_xmin = ctr_x - width / 2
            origin_xmax = ctr_x + width / 2
            origin_ymin = ctr_y - height / 2
            origin_ymax = ctr_y + height / 2
            cut_xmax = origin_xmax
            cut_xmin = origin_xmin
            cut_ymax = origin_ymax
            cut_ymin = origin_ymin
            if max_x >= origin_xmax:
                cut_xmax = max_x
            if min_x <= origin_xmin:
                cut_xmin = min_x
            if max_y >= origin_ymax:
                cut_ymax = max_y
            if min_y <= origin_ymin:
                cut_ymin = min_y
            for i in range(len(tran_boxes)):
                tran_x[i] = tran_x[i] - cut_xmin 
                tran_y[i] = tran_y[i] - cut_ymin 
            image = image.crop((cut_xmin,cut_ymin,cut_xmax,cut_ymax))
            tran_x = np.array(tran_x)
            tran_y = np.array(tran_y)

            boxes = []
            masks = []
            mty = []
            kes = []
            #GET FORMAT OF BOXES,MASKS,MTY,KES
            for idx in range(int(tran_x.size/4)):
                x_points = [tran_x[4 * idx], tran_x[4*idx+1],tran_x[4*idx+2],tran_x[4*idx+3]]
                y_points = [tran_y[4 * idx], tran_y[4*idx+1],tran_y[4*idx+2],tran_y[4*idx+3]]

                l1 = LineString([(x_points[0], y_points[0]), (x_points[2], y_points[2])])
                l2 = LineString([(x_points[1], y_points[1]), (x_points[3], y_points[3])])
                p_l1l2 = l1.intersection(l2)
                poly1 = Polygon([(x_points[0], y_points[0]), (x_points[1], y_points[1]),
                                (x_points[2], y_points[2]), (x_points[3], y_points[3])])
                if not poly1.is_valid:
                    continue
                if not p_l1l2.within(poly1):
                    continue
                if poly1.area <= 10:
                    continue
                x_min = min(x_points)
                x_max = max(x_points)
                y_min = min(y_points)
                y_max = max(y_points)
                width = max(0, x_max - x_min + 1)
                height = max(0, y_max - y_min + 1)
                if width == 0 or height == 0:
                    continue
                boxes.append([x_min,y_min,width,height])

                #get mask format
                one_point = [[tran_x[4*idx],tran_y[4*idx],tran_x[4*idx+1],tran_y[4*idx+1],tran_x[4*idx+2],tran_y[4*idx+2],tran_x[4*idx+3],tran_y[4*idx+3]]]
                masks.append(one_point)

                #get matchtype format
                mean_x = np.mean(x_points)
                mean_y = np.mean(y_points)
                xt_sort = np.sort(x_points)
                yt_sort = np.sort(y_points)
                xt_argsort = list(np.argsort(x_points))
                yt_argsort = list(np.argsort(y_points))
                ldx = []
                for ildx in range(4):
                    ldx.append(yt_argsort.index(xt_argsort[ildx]))
                all_types = [[1,2,3,4],[1,2,4,3],[1,3,2,4],[1,3,4,2],[1,4,2,3],[1,4,3,2],\
                                [2,1,3,4],[2,1,4,3],[2,3,1,4],[2,3,4,1],[2,4,1,3],[2,4,3,1],\
                                [3,1,2,4],[3,1,4,2],[3,2,1,4],[3,2,4,1],[3,4,1,2],[3,4,2,1],\
                                [4,1,2,3],[4,1,3,2],[4,2,1,3],[4,2,3,1],[4,3,1,2],[4,3,2,1]]
                all_types = [[all_types[iat][0]-1,all_types[iat][1]-1,all_types[iat][2]-1,all_types[iat][3]-1] for iat in range(24)]
                match_type = all_types.index(ldx)
                mty.append(match_type)
                
                half_x = (xt_sort + mean_x) / 2
                half_y = (yt_sort + mean_y) / 2

                keypoints = []
                keypoints.append(mean_x)
                keypoints.append(mean_y)
                keypoints.append(2)
                for i in range(4):
                    keypoints.append(half_x[i])
                    keypoints.append(mean_y)
                    keypoints.append(2)
                for i in range(4):
                    keypoints.append(mean_x)
                    keypoints.append(half_y[i])
                    keypoints.append(2)
                try:
                    keypoints.append(int(p_l1l2.x))
                    keypoints.append(int(p_l1l2.y))
                    keypoints.append(2)
                except Exception as e:
                    continue
                kes.append(keypoints)
                
            #IF ENCOUNTER THAT NO BOX IN A TRANSFORMED IMAGE, RETURN PRIMARY IMAGE AND TARGET
            if kes == []:
                image = image1
                target = target1
                return image,target
            classes = []
            for i in range(len(boxes)):
                classes.append(1)
            classes = torch.tensor(classes)
            #GET NEW TARGET
            boxes = torch.as_tensor(boxes).reshape(-1, 4)  
            target = BoxList(boxes, image.size, mode="xywh").convert("xyxy")

            target.add_field("labels",classes)

            masks = SegmentationMask(masks, image.size)
            target.add_field("masks", masks)

            kes = self.kes_gen(kes)
            kes = textKES(kes, image.size)
            target.add_field("kes", kes)

            mty = MTY(mty, image.size)
            target.add_field("mty", mty)

        return image,target