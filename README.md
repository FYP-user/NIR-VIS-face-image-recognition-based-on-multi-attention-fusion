## Our Works

##Title
Lightweight NIR-VIS face image recognition using deep learning and multi-attention fusion

##Dataset Introduction and Dataset Source
https://github.com/ZhaoJ9014/face.evoLVe#Data-Zoo
The CASIA NIR-VIS 2.0 face dataset is one of the most prominent NIR-VIS face recognition datasets.
The BUAA-VISNIR face dataset is a widely used heterogeneous face database, often employed for 376
evaluating domain adaptation.
The Oulu-Casia NIR&VIS dataset is a near-infrared face recognition dataset designed to address 382
illumination variations.
The LFW (Labled Faces in the Wild) dataset is a commonly used test set for face recognition.
The CASIA NIR-VIS 2.0 face dataset that support the findings of this study is available at
http://www.cbsr.ia.ac.cn/english/NIR-VIS-2.0-Database.html.
The BUAA-VisNir dataset was created by researchers at Beihang University. Access can be requested directly from the original creators (Contact:
Prof. Di Huang) or via their laboratory website. 
The Oulu-Casia NIR&VIS dataset is available athttps://www.oulu.fi/en/university/faculties-and-units/faculty-information-technology-and-electrical-engin
eering/center-for-machine-vision-and-signal-analysis\#off-canvas. 
The LFW database is available at http://vis-www.cs.umass.edu/lfw/.

## Prerequisites
- Python 3.7.0 & PyTorch 1.5.0 & Torchvision 0.6.0
- Download LightCNN-29 [[Google Drive](https://drive.google.com/file/d/1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS/view)] pretrained on MS-Celeb-1M.
- Download Identity Sampler [[Google Drive](https://drive.google.com/file/d/1kezDpwqA4a3WGq5PfS1kDrASf3-bN4js/view?usp=sharing)] pretrained on MS-Celeb-1M.
- Put the above two models in `./pre_train`

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
@article{fu2021dvg,
  title={DVG-face: Dual variational generation for heterogeneous face recognition},
  author={Fu, Chaoyou and Wu, Xiang and Hu, Yibo and Huang, Huaibo and He, Ran},
  journal={IEEE TPAMI},
  year={2021}
}

@inproceedings{fu2019dual,
  title={Dual Variational Generation for Low-Shot Heterogeneous Face Recognition},
  author={Fu, Chaoyou and Wu, Xiang and Hu, Yibo and Huang, Huaibo and He, Ran},
  booktitle={NeurIPS},
  year={2019}
}

@article{fu2022towards,
  title={Towards Lightweight Pixel-Wise Hallucination for Heterogeneous Face Recognition},
  author={Fu, Chaoyou and Zhou, Xiaoqiang and He, Weizan and He, Ran},
  journal={IEEE TPAMI},
  year={2022}
}

@inproceedings{duan2020cross,
  title={Cross-spectral face hallucination via disentangling independent factors},
  author={Duan, Boyan and Fu, Chaoyou and Li, Yi and Song, Xingguang and He, Ran},
  booktitle={CVPR},
  year={2020}
}

```








