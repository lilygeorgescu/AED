## A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah.
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.

Official URL: https://ieeexplore.ieee.org/document/9410375

ArXiv URL: https://arxiv.org/abs/2008.12328

This is the official repository of "A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video".

### In order to receive our code fill in this [form](./SecurifAI-form-and-license-PAMI-2021.pdf) and send a copy to georgescu_lily@yahoo.com and raducu.ionescu@gmail.com. The form must be sent from your academic email.

### License
The source code of our model is released under the SecurifAI’s NonCommercial Use & No Sharing International Public License. The details of this license are presented in SecurifAI-license-v1.0.pdf.

### Citation
Please cite our work if you use any material released in this repository.

```
@ARTICLE{Georgescu-TPAMI-2021, 
  author={Georgescu, Mariana Iuliana and Ionescu, Radu Tudor and Khan, Fahad Shahbaz and Popescu, Marius and Shah, Mubarak}, 
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},  
  title={A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3074805}}
```

### This repo contains: 
 - the form that must be filled in order to obtain our source code.
 - the track annotations for the ShanghaiTech Campus dataset, released under the CC BY-NC-ND 4.0 license (use the tracks to compute the new RBDC and TBDC metrics introduced in "Bharathkumar Ramachandra, Michael Jones. Street Scene: A new dataset and evaluation protocol for video anomaly detection. WACV, 2020").
 - the region and track annotations for the Subway dataset, released under the CC BY-NC-ND 4.0 license (use the annotations to compute the pixel-level AUC and the new RBDC and TBDC metrics introduced in "Bharathkumar Ramachandra, Michael Jones. Street Scene: A new dataset and evaluation protocol for video anomaly detection. WACV, 2020").
 - annotated videos obtained using our abnormal event detection method. 
 - the unofficial evaluation code for the new RBDC and TBDC metrics from: 

## Street Scene: A new dataset and evaluation protocol for video anomaly detection,
released under the CC BY-NC-ND 4.0 license.

### Pseudo-abnormal data set for appearance AE
We use the following data sets to form the Pseudo-abnormal data set for appearance AE:
- UIUCTex (reference 68 in the paper)
- Oxford Flowers (reference 69 in the paper)
- Anime face images https://www.kaggle.com/splcher/animefacedataset
- The following categories from Tiny ImageNet:  
{'n0176824', 'n0216545', 'n02666196', 'n0227997', 'n018827', 'n077151', 'n04067472', 'n077476', 'n01944390', 'n020565', 'n092564', 'n01644900', 'n04597913', 'n01774384', 'n019107', 'n04179913', 'n0439938', 'n0177438', 'n0220685', 'n02226429', 'n076145', 'n0758306', 'n020027', 'n017682', 'n0775359', 'n07875152', 'n07614500', 'n019172', 'n01629819', 'n023215', 'n023954', 'n017421', 'n07871810', 'n04328186', 'n0223148', 'n022799', 'n043980', 'n018556', 'n01698640', 'n0439804', 'n07715103', 'n0299941', 'n026661', 'n016986', 'n0771510', 'n0191074', 'n04399382', 'n022814', 'n0426527', 'n0226844', 'n019443', 'n07583066', 'n021901', 'n07720875', 'n0185567', 'n01882714', 'n078718', 'n077208', 'n0757978', 'n02999410', 'n0769574', 'n0200272', 'n040674', 'n022068', 'n02165456', 'n07579787', 'n0205657', 'n01984695', 'n014435', 'n0787181', 'n0772087', 'n0188271', 'n02236044', 'n02074367', 'n0774958', 'n03544143', 'n019846', 'n017703', 'n035441', 'n0219016', 'n0228140', 'n0232152', 'n043281', 'n01774750', 'n0393754', 'n02268443', 'n022333', 'n09256479', 'n02321529', 'n0191728', 'n045967', 'n01768244', 'n0427554', 'n0174217', 'n0169864', 'n01443537', 'n02231487', 'n0198469', 'n0787515', 'n02206856', 'n01770393', 'n0198348', 'n04275548', 'n022360', 'n022314', 'n0432818', 'n034243', 'n01945685', 'n02058221', 'n016449', 'n0207436', 'n0178467', 'n0459791', 'n0239540', 'n0925647', 'n042755', 'n0340023', 'n04596742', 'n0776869', 'n02056570', 'n0144353', 'n075830', 'n0177039', 'n07747607', 'n0406747', 'n0164490', 'n0380474', 'n07753592', 'n017747', 'n0342432', 'n022684', 'n02395406', 'n029994', 'n0194568', 'n01910747', 'n0222642', 'n0164157', 'n0162981', 'n03424325', 'n07873807', 'n02190166', 'n019507', 'n039375', 'n020743', 'n0354414', 'n041799', 'n0459674', 'n022264', 'n04398044', 'n034002', 'n01950731', 'n075797', 'n0417991', 'n02281406', 'n0223333', 'n016415', 'n016298', 'n078738', 'n02002724', 'n01742172', 'n077495', 'n019834', 'n077535', 'n077347', 'n07768694', 'n0774760', 'n0205822', 'n038047', 'n01641577', 'n017846', 'n0177475', 'n01855672', 'n045979', 'n037062', 'n01917289', 'n078751', 'n019456', 'n0194439', 'n07734744', 'n02279972', 'n02233338', 'n0195073', 'n043993', 'n07695742', 'n077686', 'n042652', 'n0223604', 'n0266619', 'n0370622', 'n017743', 'n03937543', 'n04265275', 'n0787380', 'n076957', 'n0761450', 'n03804744', 'n020582', 'n03400231', 'n0773474', 'n07749582', 'n03706229', 'n021654', 'n01784675', 'n01983481'}

### Brief  Description 
Our framework is composed of an object detector, a set of appearance and motion auto-encoders, and a set of classifiers.
Since our framework only looks at object detections, it can be applied to different scenes, provided that normal 
events are defined identically across scenes and that the single main factor of variation is the background. 
To overcome the lack of abnormal data during training, we propose an adversarial learning strategy for the auto-encoders. 
We create a scene-agnostic set of out-of-domain pseudo-abnormal examples, which are correctly reconstructed 
by the auto-encoders before applying gradient ascent on the pseudo-abnormal examples. 
We further utilize the pseudo-abnormal examples to serve as abnormal examples when training appearance-based 
and motion-based binary classifiers to discriminate between normal and abnormal latent features and reconstructions.

This is the pipeline of our framework:
![pipeline](figs/Pipeline.png)

Object reconstructions before and after adversarial training:
![rec](figs/Prelim.png)
