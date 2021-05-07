## A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah.
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.

Official URL: https://ieeexplore.ieee.org/document/9410375

ArXiv URL: https://arxiv.org/abs/2008.12328

This is the official repository of "A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video".

### In order to receive our code fill in this [form](./SecurifAI-form-and-license-PAMI-2021.pdf) and send a copy to georgescu_lily@yahoo.com and raducu.ionescu@gmail.com

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
 - the unofficial evaluation code for the new RBDC and TBDC metrics, released under the CC BY-NC-ND 4.0 license.

### Short Abstract 
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
