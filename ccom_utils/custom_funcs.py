import os, cv2
import torch, csv, glob
import numpy as np
import json
import utils.general as gen
from utils.metrics import bbox_iou


def frames2distances(csvpath):
    # takes path to csv of frame nums and their corresponding distances
    # returns two corresponding arrays of frame numbers and the interpolated distances
    frames = []
    dists = []
    with open(csvpath,'r') as csvfile:
        data = csv.reader(csvfile)
        for lines in data:
            if lines != ['frame','distance'] and lines != '':
                # try:
                lines = list(map(int,lines))
                f, d = lines
                frames.append(f)
                dists.append(d)
                # except Exception as e:
                #     print(e)
                #     print(lines, csvpath)
    f_min = min(frames)
    f_max = max(frames)
    dif = f_max - f_min
    new_frames = np.round(np.linspace(f_min, f_max, dif))
    new_frames = new_frames.astype(int)
    interp_dists = np.interp(new_frames, frames, dists)
    return(new_frames, interp_dists)

class Inference:
    def __init__(self, pred_data):
        self.times = pred_data['Time Stats']
        self.detects = [Detect(i) for i in pred_data['Detections']]
        self.imgsz = pred_data['Img Dimensions']
    
class Detect:
    def __init__(self,detects):
        if detects:
            self.xywhn = detects[0:4]
            self.conf = detects[4]
            self.class_id = detects[5]
            self.class_name = detects[6]
