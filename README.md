# Box_Discretization_Network
This repository is built on the **pytorch [[maskrcnn_benchmark]](https://github.com/facebookresearch/maskrcnn-benchmark)**. 

# Description
**Paper [[link]](https://arxiv.org/abs/1906.02371)**. 

This method is served as the foundation for our recent ICDAR 2019 ReCTs competition method [[link]](https://rrc.cvc.uab.es/?ch=12), which won the first place of the detection task.


# Getting Started

An easy guide for training.

## Install anaconda 

Link：https://pan.baidu.com/s/1TGy6O3LBHGQFzC20yJo8tg psw：vggx

## Step-by-step install
 ```shell
conda create --name mb
conda activate mb
conda install ipython
pip install ninja yacs cython matplotlib tqdm scipy shapely
conda install pytorch=1.0 torchvision=0.2 cudatoolkit=9.0 -c pytorch
conda install -c menpo opencv
export INSTALL_DIR=$PWD
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
cd $INSTALL_DIR
git clone https://github.com/Yuliang-Liu/bdn.git
cd bdn
python setup.py build develop
```

## Pretrained model：

[[Link]](https://drive.google.com/file/d/1pBQ53ZNvsdu8byFKDST-de30X5pEFI7C/view?usp=sharing)
unzip under project_root

## ic15 data

Prepare data follow COCO format.
[[Link]](https://drive.google.com/file/d/16rpK9Ql4mZydl1CGPMQXf0Q8YQqtXnX7/view?usp=sharing)
unzip under project_root

## Train

After downloading data and model, run
  ```shell
  ./quick_train_guide.sh
 ```

## Test

Run 
 ```shell
 ./my_test.sh
```

Put kes.json to ic15_TIoU_metric/
inside ic15_TIoU_metric/
Run 
 ```shell
 python2 to_eval.py
```

## Visualization 

Run 
 ```shell
 ./single_image_demo.sh
```

# Citation
If you find our metric useful for your reserach, please cite
'''
@article{liu2019omnidirectional,
  title={Omnidirectional Scene Text Detection with Sequential-free Box Discretization},
  author={Liu, Yuliang and Zhang, Sheng and Jin, Lianwen and Xie, Lele and Wu, Yaqiang and Wang, Zhepeng},
  journal={arXiv preprint arXiv:1906.02371},
  year={2019}
}
'''

## Feedback 
Suggestions and opinions of this metric (both positive and negative) are greatly welcome. Please contact the authors by sending email to 
  `liu.yuliang@mail.scut.edu.cn` or `yuliang.liu@adelaide.edu.au`.
