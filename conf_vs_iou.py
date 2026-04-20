import matplotlib.pyplot as plt
import os, torch, cv2, shutil, json, csv
import numpy as np
import pandas as pd
from utils.metrics import bbox_iou
from utils.metrics import bbox_iou
import ccom_yolov5.ccom_utils.custom_funcs as cf
from val import process_batch
from utils import general
'''
For each detection in each video, 
calculate iou with groundtruth annotations
and record confidence, plot histogram of iou
by 10% increments of confidence
'''

dhufish_path = '/home/field/jennaDir/ccom_yolov5'
yolo_dir = dhufish_path
model_name = 'model4_3_best'
model_path = 'model_testing/model4_3_best.pt'
weights_path= os.path.join(yolo_dir, model_path)
img_sz = 1280
# model = torch.hub.load(yolo_dir, 'custom', path = weights_path, source='local',force_reload=True)

# establish video paths
video_dir = '/home/field/jennaDir/dive_tools/annotated_vids'
expr = 'bitrate_exp'
lbls = 'labels'
mdl = 'model4_3_best'
m_abrv = 'm4'
buoys = ['lobsterPot', 'mooringBall', 'redBall']
bitrates = [250, 500, 1000, 2500, 5000, 'native'] # in kbps


for i in range(0,len(buoys)):
    buoy = buoys[i]

    # load ground truth labels and inference labels
    gt_file = m_abrv + '_gt_' + buoy + '.json'
    gt_path = os.path.join(video_dir, expr, buoy, lbls, gt_file)
    with open(gt_path,'r') as f:
        gt_lbls = json.load(f)

    inf_json_file = 'inference_' + buoy + '.json'
    inf_path = os.path.join(video_dir, expr, buoy, mdl, inf_json_file)
    with open(inf_path,'r') as f:
        buoy_inf = json.load(f)
    
    correct_buoy = {}

    # for each bitrate
    for bitrate in bitrates:
        correct_video = {}
        inf_vid = buoy_inf[str(bitrate) + ' kbps']
        csv_path = os.path.join(video_dir,expr,buoy,'frames_dist_corr.csv')
        frames, dists = cf.frames2distances(csv_path)

        for frame in inf_vid: # for every detect in the video, look up ground truth
            detects = inf_vid[frame]['Detections'] # detects = N x [x, y, w, h, conf, class]
            if detects != []:
                # FUTURE WORK add loop here for detect in detects. change output to contain frame info and then detects which contains iou and class t/f
                try:
                    gt = gt_lbls[frame] # gt = M x [class, x, y, w, h]
                except KeyError as e:
                    print(f"Couldn't find a ground truth label for frame {frame}")
                    # create empty gt
                    gt = np.array([])
                else:
                    # separate xywh and class + conf from detect and gt
                    det_xywhn = np.array(np.array(detects)[:,:4],dtype=float)
                    gt_xywhn = np.array(gt, dtype=float)[:,1:] 
                    det_confc = np.array(detects)[:,4:6] # np.arr(N X [ conf, class_id])
                    gt_c = np.array(gt, dtype=float)[:,0:1].astype(float) # np.arr( M x [class])

                    # calc euclidean distance btwn gt center and detection center
                    pix_d = det_xywhn[:,0:2] * np.array([1920,1088])
                    pix_gt = gt_xywhn[0,0:2] * np.array([1920,1088]
                                                        )
                    euc = np.linalg.norm(pix_d - pix_gt)

                    # convert to xyxy
                    det_xyxy = general.xywhn2xyxy(det_xywhn.astype(float), w = 1280, h = 720) # assumed no padding on the image
                    gt_xyxy = general.xywhn2xyxy(gt_xywhn.astype(float),w=1280,h=720) # assumed no padding on the image

                    # recombine
                    d = np.hstack((det_xyxy, det_confc))
                    g = np.hstack((gt_c, gt_xyxy))
                    d = d.astype(float)
                    g = g.astype(float)                    # convert detects to np array (N,6) [x1, y1, x2, y2, conf, class]
                    # convert gt to np array (m, 6) [class, x1, y1, x2, y2]
                    iouv = torch.linspace(0.05,0.95,19,device=None) # range of iou values

                    correct_bb, correct_class, iou = process_batch(torch.Tensor(d), torch.Tensor(g), iouv)

                    # pull distance associated with frame
                    idx = np.where(frames == int(frame))
                    dist = dists[idx]

                    correct_video[frame] = {
                'Distance': list(dist),
                'IoU 0.05:0.95': correct_bb.tolist(),
                'IoU': iou.tolist(),
                'Class Match': correct_class,
                'Euclidean Distance': euc,
                'Detections': detects,
                'Ground Truth': gt                
            }
   
        correct_buoy[str(bitrate) + ' kbps'] = correct_video


    # save results for each buoy
    save_val_json = 'iouStats_' + buoy + '.json'
    save_val_path = os.path.join(video_dir, expr, buoy, save_val_json)
    with open(save_val_path, 'w') as jfile:
        json.dump(correct_buoy, jfile)
