#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)
#  (https://creativecommons.org/licenses/by-nc-nd/4.0/)

"""
How to run:
    python compute_tbdc_rbdc.py --tracks-path=toy_tracks --anomalies-path=toy_anomalies --num-frames=10
"""

import argparse
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


class ContinuousTrack:

    def __init__(self, start_idx=0, end_idx=None, masks=0, video_name=""):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.bboxes = {}
        self.masks = masks
        self.video_name = video_name

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class Region:

    def __init__(self, frame_idx, bbox, score, video_name, track_id=-1):
        self.frame_idx = frame_idx
        self.bbox = bbox
        self.score = score
        self.video_name = video_name
        self.track_id = track_id

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def get_matching_gt_indices(pred_anomaly, gt_anomalies_per_frame, beta):
    indices = []
    for index, gt_anomaly in enumerate(gt_anomalies_per_frame):
        iou = bb_intersection_over_union(gt_anomaly.bbox, pred_anomaly.bbox)
        if iou >= beta:
            indices.append(index)

    return indices


def compute_tbdr(gt_tracks, num_matched_detections_per_track, alpha):
    percentages = np.array([x / len(y.bboxes) for x, y in zip(num_matched_detections_per_track, gt_tracks)])
    return np.sum(percentages >= alpha) / len(gt_tracks)


def compute_fpr_rbdr(pred_anomalies_detected: [Region], gt_anomalies: [Region], all_gt_tracks,
                     num_frames, alpha=0.1, beta=0.1):

    num_tracks = len(all_gt_tracks)
    num_matched_detections_per_track = [0] * num_tracks

    # TODO: add pixel level IOU
    num_detected_anomalies = len(pred_anomalies_detected)
    gt_anomaly_video_per_frame_dict = {}
    found_gt_anomaly_video_per_frame_dict = {}

    for anomaly in gt_anomalies:
        anomalies_per_frame = gt_anomaly_video_per_frame_dict.get((anomaly.video_name, anomaly.frame_idx), None)
        if anomalies_per_frame is None:
            gt_anomaly_video_per_frame_dict[(anomaly.video_name, anomaly.frame_idx)] = [anomaly]
            found_gt_anomaly_video_per_frame_dict[(anomaly.video_name, anomaly.frame_idx)] = [0]
        else:
            gt_anomaly_video_per_frame_dict[(anomaly.video_name, anomaly.frame_idx)].append(anomaly)
            found_gt_anomaly_video_per_frame_dict[(anomaly.video_name, anomaly.frame_idx)].append(0)

    tp = np.zeros(num_detected_anomalies)
    fp = np.zeros(num_detected_anomalies)
    tbdr = np.zeros(num_detected_anomalies)

    pred_anomalies_detected.sort(key=lambda anomaly_detection: anomaly_detection.score, reverse=True)
    for idx, pred_anomaly in enumerate(pred_anomalies_detected):
        gt_anomalies_per_frame = gt_anomaly_video_per_frame_dict.get((pred_anomaly.video_name, pred_anomaly.frame_idx),
                                                                     None)

        if gt_anomalies_per_frame is None:
            fp[idx] = 1
        else:
            matching_gt_bboxes_indices = get_matching_gt_indices(pred_anomaly, gt_anomalies_per_frame, beta)
            if len(matching_gt_bboxes_indices) > 0:
                non_matched_indices = []
                for matched_ind in matching_gt_bboxes_indices:
                    if found_gt_anomaly_video_per_frame_dict.get((pred_anomaly.video_name,
                                                                  pred_anomaly.frame_idx))[matched_ind] == 0:
                        non_matched_indices.append(matched_ind)
                        found_gt_anomaly_video_per_frame_dict.get((pred_anomaly.video_name, pred_anomaly.frame_idx))[
                                matched_ind] = 1
                        num_matched_detections_per_track[gt_anomalies_per_frame[matched_ind].track_id] += 1

                tp[idx] = len(non_matched_indices)
            else:
                fp[idx] = 1

        tbdr[idx] = compute_tbdr(all_gt_tracks, num_matched_detections_per_track, alpha)

    cum_false_positive = np.cumsum(fp)
    cum_true_positive = np.cumsum(tp)

    # add the point (0, 0) for each vector
    cum_false_positive = np.concatenate(([0], cum_false_positive))
    cum_true_positive = np.concatenate(([0], cum_true_positive))
    tbdr = np.concatenate(([0], tbdr))

    rbdr = cum_true_positive / len(gt_anomalies)
    fpr = cum_false_positive / num_frames

    idx_1 = np.where(fpr <= 1)[0][-1] + 1

    if fpr[idx_1 - 1] != 1:
        print('fpr does not reach 1')
        rbdr = np.insert(rbdr, idx_1, rbdr[idx_1 - 1])
        tbdr = np.insert(tbdr, idx_1, tbdr[idx_1 - 1])
        fpr = np.insert(fpr, idx_1, 1)
        idx_1 += 1

    tbdc = metrics.auc(fpr[:idx_1], tbdr[:idx_1])
    rbdc = metrics.auc(fpr[:idx_1], rbdr[:idx_1])

    print(f'tbdc = {tbdc}')
    print(f'rbdc = {rbdc}')

    plt.plot(fpr, rbdr, '-')
    plt.xlabel('FPR')
    plt.ylabel('RBDR')
    plt.show()

    plt.plot(fpr, tbdr, '-')
    plt.xlabel('FPR')
    plt.ylabel('TBDR')
    plt.show()


def read_tracks(tracks_path) -> [ContinuousTrack]:
    """
    :param tracks_path - the path to the tracks in txt
    /path/to/tracks/01.txt - 01 is the name of the video
    /path/to/tracks/02.txt - 02 is the name of the video
    """

    # !!! the regions of the tracks must be written in the correct order
    # i.e. [track 1 region1, track 1 region 2, .. track 2 region 1, ..]
    # NOT  [track 1 region1, track 2 region 1, .. track 1 region 2, ..]
    def create_tracks(tracks_array_) -> [ContinuousTrack]:
        num_tracks = int(tracks_array_[-1][0] + 1)  # the last track-id
        tracks: [ContinuousTrack] = []
        for track_id in range(num_tracks):
            regions_per_track = tracks_array_[tracks_array_[:, 0] == track_id]
            track = ContinuousTrack(start_idx=regions_per_track[0][1],
                                    end_idx=regions_per_track[-1][1],
                                    video_name=video_name[:-4])

            for region in regions_per_track:
                assert region[0] == track_id
                frame_id = region[1]
                bbox = region[2:]
                track.bboxes[frame_id] = list(bbox)
            tracks.append(track)
        return tracks

    video_names = os.listdir(tracks_path)
    all_tracks = []
    for video_name in video_names:
        tracks_array = np.loadtxt(os.path.join(tracks_path, video_name), delimiter=',')
        all_tracks += create_tracks(tracks_array)
    return all_tracks


def get_gt_regions(tracks: [ContinuousTrack]):
    gt_regions: [Region] = []

    for track_id, track in enumerate(tracks):
        for frame_idx, bbox in track.bboxes.items():
            gt_regions.append(Region(frame_idx=frame_idx,
                                     bbox=list(bbox),
                                     score=1,
                                     video_name=track.video_name,
                                     track_id=track_id))

    return gt_regions


def read_detected_anomalies(anomalies_path) -> [Region]:
    """
    :param anomalies_path - the path to the detected anomalies with the format
     [frame_id, x_min, y_min, x_max, y_max, anomaly_score]
    /path/to/tracks/01.txt - 01 is the name of the video
    /path/to/tracks/02.txt - 02 is the name of the video
    """
    predicted_regions: [Region] = []
    video_names = os.listdir(anomalies_path)

    for video_name in video_names:
        detected_anomalies = np.loadtxt(os.path.join(anomalies_path, video_name), delimiter=',')
        # detected_anomaly = [frame_id, x_min, y_min, x_max, y_max, anomaly_score]
        for detected_anomaly in detected_anomalies:
            predicted_regions.append(Region(frame_idx=detected_anomaly[0],
                                            bbox=list(detected_anomaly[1:5]),
                                            score=detected_anomaly[5],
                                            video_name=video_name[:-4]))
    return predicted_regions


def compute_metrics(tracks_path, anomalies_path, num_frames_in_video):

    all_gt_tracks = read_tracks(tracks_path)
    gt_regions = get_gt_regions(all_gt_tracks)
    predicted_anomalies = read_detected_anomalies(anomalies_path)
    compute_fpr_rbdr(predicted_anomalies, gt_regions, all_gt_tracks, num_frames_in_video)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--tracks-path', required=True, type=str, help="The path to the tracks.")
parser.add_argument('--anomalies-path', required=True, type=str, help="The path to the detected anomalies.")
parser.add_argument('--num-frames', required=True, type=int, help="The total number of frames in videos.")

args = parser.parse_args()

if __name__ == '__main__':
    compute_metrics(tracks_path=args.tracks_path,
                    anomalies_path=args.anomalies_path,
                    num_frames_in_video=args.num_frames)

"""
How to run:
    python compute_tbdc_rbdc.py --tracks-path=toy_tracks --anomalies-path=toy_anomalies --num-frames=5
"""