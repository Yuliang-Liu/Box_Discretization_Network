import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

from maskrcnn_benchmark.structures.ke import textKES
from maskrcnn_benchmark.structures.mty import MTY

DEBUG = 0

class WordDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(WordDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

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

    def __getitem__(self, idx):
        img, anno = super(WordDataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        if DEBUG: print('len(boxes)', len(boxes), boxes[0])
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        if DEBUG: print('len(classes)', len(classes), classes[0])
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        if DEBUG: print('len(masks)', len(masks), masks[0])
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        if anno and 'keypoints' in anno[0]:
            kes = [obj["keypoints"] for obj in anno]
            kes = self.kes_gen(kes)
            if DEBUG: print('len(kes)', len(kes), kes[0])
            kes = textKES(kes, img.size)
            target.add_field("kes", kes)

        if anno and 'match_type' in anno[0]:
            mty = [obj["match_type"] for obj in anno]
            mty = MTY(mty, img.size)
            target.add_field("mty", mty)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
