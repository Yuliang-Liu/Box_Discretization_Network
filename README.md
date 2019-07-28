# Box_Discretization_Network
This repository is built on the **pytorch [[maskrcnn_benchmark]](https://github.com/facebookresearch/maskrcnn-benchmark)**. Full code will be released soon.

# Description
To produce quadrilateral bounding box, segmentation-based methods need post-processing step to group pixels; non-segmentation methods are sensitive to annotating sequence, which undermining detection performance. This method can directly produce quadrilateral bounding box without these drawbacks, and it thus substantially improve the mask r-cnn performance in scene text datasets. 

**Paper [[link]](https://arxiv.org/abs/1906.02371)**. If you have questions or confuses about the paper, please open an issue. We welcom any discussion.

This method is also served as the foundation for our recent ICDAR 2019 ReCTs competition method [[link]](https://rrc.cvc.uab.es/?ch=12), which won the first place of the detection task.

