import os, cv2
import torch, csv, glob
import numpy as np
import json
import utils.general as gen
from utils.metrics import bbox_iou


#stale script

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
