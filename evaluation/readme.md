## Code for computing the new TBDC and RBDC metrics introduced metrics introduced in "Ramachandra Bharathkumar, Jones Michael, Street Scene: A new dataset and evaluation protocol for video anomaly detection, WACV 2020"

This is the unofficial implementation of the TBDC and RBDC metrics. We follow the instructions given in this paper "Ramachandra Bharathkumar, Jones Michael, Street Scene: A new dataset and evaluation protocol for video anomaly detection, WACV 2020" to implement these metrics.

In order to run our script use: 
```
python compute_tbdc_rbdc.py --tracks-path=toy_tracks --anomalies-path=toy_anomalies --num-frames=10
```
where
- ```tracks-path``` is the path to the folder containing the tracks for all videos.
    - The tracks are organized as follows:
        - for each video, we have a txt file containing all the regions with the following format:
        [track_id, frame_id, x_min, y_min, x_max, y_max]
        - the track_id must be in ascending order
- ```anomalies-path``` is the path to the folder containing the detected anomaly regions for all videos.
    - The anomaly regions are organized as follows:
        - for each video, we have a txt file containing all the detected regions with the following format:
        [frame_id, x_min, y_min, x_max, y_max, anomaly_score] 
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

