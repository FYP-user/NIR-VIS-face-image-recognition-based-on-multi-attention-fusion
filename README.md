## Our Works

## Title
Lightweight NIR-VIS face image recognition using deep learning and multi-attention fusion

## Dataset Information and ‎Description
https://github.com/ZhaoJ9014/face.evoLVe#Data-Zoo
The CASIA NIR-VIS 2.0 face dataset is one of the most prominent NIR-VIS face recognition datasets.
The BUAA-VISNIR face dataset is a widely used heterogeneous face database, often employed for 376
evaluating domain adaptation.
The Oulu-Casia NIR&VIS dataset is a near-infrared face recognition dataset designed to address 382
illumination variations.
The LFW (Labled Faces in the Wild) dataset is a commonly used test set for face recognition.
The CASIA NIR-VIS 2.0 face dataset that support the findings of this study is available at
http://www.cbsr.ia.ac.cn/english/NIR-VIS-2.0-Database.html.
The BUAA-VisNir dataset was created by researchers at Beihang University. Access can be requested directly from the original creators (Contact:Prof. Di Huang) or via their laboratory website. 
The Oulu-Casia NIR&VIS dataset is available athttps://www.oulu.fi/en/university/faculties-and-units/faculty-information-technology-and-electrical-engineering/center-for-machine-vision-and-signal-analysis\#off-canvas. 
The LFW dataset is publicly available at the official website (\url{ http://vis-www.cs.umass.edu/lfw/}). An alternative repository is also hosted on Kaggle (\url{https://www.kaggle.com/datasets/jessicali9530/lfw-dataset}).

## ‎Requirements – Any dependencies
- pip install torch torchvision
- pip install pillow
- pip install numpy
- pip install caffe2

## Code Information
## Prerequisites
- Python 3.7.0 & PyTorch 1.5.0 & Torchvision 0.6.0
- Download LightCNN-29 [[Google Drive](https://drive.google.com/file/d/1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS/view)] pretrained on MS-Celeb-1M.
- Download Identity Sampler [[Google Drive](https://drive.google.com/file/d/1kezDpwqA4a3WGq5PfS1kDrASf3-bN4js/view?usp=sharing)] pretrained on MS-Celeb-1M.
- Put the above two models in `./pre_train`

## Usage Instructions
## Train the generator
`train.py`:
Fill out options of '--img_root' and '--train_list', which are the image root and training list of the heterogeneous data, respectively.
An example of the training list:
```
NIR/s2_NIR_10039_001.jpg 232
VIS/s1_VIS_00134_010.jpg 133
NIR/s1_NIR_00118_011.jpg 117
```
Here we use 'NIR' and 'VIS' in the training list to distinguish the modalities of images. If your list has other distinguishable marks,
please change them correspondingly in `./data/dataset.py` (lines 28, 38, 66, and 68).
```
python train_generator.py --gpu_ids 0
```

## Generate images
`generate.py`:
Fill out options of '--img_root' and '--train_list' that are the same as the above options.
```
python gen_samples.py --gpu_ids 0
```
The generated images will be saved in `./gen_images`

## Train the recognition model
`train.py`:
Fill out options of 'num_classes', '--img_root_A', and '--train_list_A', where the last two options are the same as the above options.
```
python train_ligthcnn.py --gpu_ids 0,1
```

## Citation
```
@inproceedings{li2013casia,
title={The casia nir-vis 2.0 face database},
author={Li, Stan Z and Yi, Dong and Lei, Zhen and Liao, Shengcai},
booktitle={Computer Vision and Pattern Recognition Workshops (CVPRW), 2013 IEEE Conference on},
pages={348--353},
year={2013},
organization={IEEE}
 }

@article{The BUAA-VisNir,
  title={The BUAA-VisNir face database instructions},
  author={Huang, Di and Sun, Jia and Wang, Yunhong},
  journal={School Comput. Sci. Eng., Beihang Univ., Beijing, China, Tech. Rep. IRIP-TR-12-FR-001},
  volume={3},
  number={3},
  pages={8},
  year={2012}
}

@TechReport{LFWTech,
    author = {Gary B. Huang and Manu Ramesh and Tamara Berg and Erik Learned-Miller},
    title = {Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments},
    institution = {University of Massachusetts, Amherst},
    year = 2007,
    number = {07-49},
    month = {October}
}

@inproceedings{Oulu-Casia NIR&VIS,
  title={Learning mappings for face synthesis from near infrared to visual light images},
  author={Chen, Jie and Yi, Dong and Yang, Jimei and Zhao, Guoying and Li, Stan Z and Pietikainen, Matti},
  booktitle={2009 IEEE conference on computer vision and pattern recognition},
  pages={156--163},
  year={2009},
  organization={IEEE}
}
```








