## Code for computing the new TBDC and RBDC metrics introduced in "Ramachandra et al. Street Scene: A new dataset and evaluation protocol for video anomaly detection. WACV, 2020"

This is an unofficial implementation of the TBDC and RBDC metrics. In order to implement these metrics, we followed the instructions given in this paper: "Bharathkumar Ramachandra, Michael Jones. Street Scene: A new dataset and evaluation protocol for video anomaly detection. WACV, 2020". Additionally, we consulted with Ramachandra Bharathkumar, confirming that the implementation is correct.

In order to run our script use: 
```
python compute_tbdc_rbdc.py --tracks-path=toy_tracks --anomalies-path=toy_anomalies --num-frames=10
```
where
- ```tracks-path``` is the path to the folder containing the tracks for all videos.
    - The tracks are organized as follows:
        - for each video, we have a txt file containing all the regions with the following format:
        
            track_id, frame_id, x_min, y_min, x_max, y_max
        
        - the track_id and frame_id must be in ascending order
- ```anomalies-path``` is the path to the folder containing the detected anomaly regions for all videos.
    - The anomaly regions are organized as follows:
        - for each video, we have a txt file containing all the detected regions with the following format:
        
            frame_id, x_min, y_min, x_max, y_max, anomaly_score
- ```num-frames``` is the total number of frames in the videos.
- The name of the video tracks must match the name of the detected region per video.

When running the above command on the toy dataset the results of RDBC and TBDC should be 1.

### License for the evaluation code
Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)

You are free to:

##### Share — copy and redistribute the material in any medium or format

Under the following terms:

##### Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

##### NonCommercial — You may not use the material for commercial purposes.

##### NoDerivatives — If you remix, transform, or build upon the material, you may not distribute the modified material.

### Citation
Please cite the following work, if you use this software:
```
@ARTICLE{Georgescu-TPAMI-2021, 
  author={Georgescu, Mariana Iuliana and Ionescu, Radu and Khan, Fahad Shahbaz and Popescu, Marius and Shah, Mubarak}, 
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},  
  title={A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3074805}}
```

