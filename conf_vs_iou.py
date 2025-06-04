import matplotlib.pyplot as plt
import os, torch, cv2, shutil, json, csv
import numpy as np
import pandas as pd
from utils.metrics import bbox_iou
from utils.metrics import bbox_iou

'''
For each detection in each video, 
calculate iou with groundtruth annotations
and record confidence, plot histogram of iou
by 10% increments of confidence
'''


# establish video paths
video_dir = '/home/field/jennaDir/dive_tools/annotated_vids'
expr = 'bitrate_exp'
buoys = ['lobsterPot', 'mooringBall', 'redBall']
bitrates = [250, 500, 1000, 2000, 3000] # in kbps


for i in range(0,len(buoys)):
    buoy = buoys[i]

    # load ground truth labels and inference labels
    gt_file = 'gt_labels_' + buoy + '.json'
    g = os.path.join(video_dir, expr, buoy, gt_file)
    with open(g,'r') as f:
        gt_lbls = json.load(f)

    inf_json_file = 'inference_' + buoy + '.json'
    inf_path = os.path.join(video_dir, expr, buoy, inf_json_file)
    with open(inf_path,'r') as f:
        buoy_inf = json.load(f)
    
    per_buoy = {}

    # for each bitrate
    for bitrate in bitrates:
        per_bitrate = {}
        inf = buoy_inf[str(bitrate) + ' kbps']

        for key in inf: # for every detect in the video, look up ground truth
            detects = inf[key]['Detections']
            if detects != []:
                try:
                    gt = gt_lbls[key]
                except KeyError as e:
                    print(f"Couldn't find a ground truth label for frame {key}")
                    gt = []
                for j in range(0,len(detects)): # compare every     
                    for q in range(0,len(gt)):
                        gt_xywhn = gt[q][1:]
                        for detect in detects:
                            det_xywhn = detect[0:4]

                            # do iou
                            if cf.compare_labels(gt_xywhn, det_xywhn, miniou):
                                val[key] = inf[key]
        val_buoy[str(bitrate) + ' kbps'] = val

    # save results for each buoy
    save_val_json = 'validated_' + buoy + '.json'
    save_val_path = os.path.join(video_dir, expr, buoy, save_val_json)
    with open(save_val_path, 'w') as jfile:
        json.dump(val_buoy,jfile)
