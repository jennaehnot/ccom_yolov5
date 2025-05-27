import cv2 
import custom_funcs as cf
import matplotlib.pyplot as plt
import os, torch, cv2, shutil, json, csv
import numpy as np
import pandas as pd
from utils.metrics import bbox_iou
import utils.general as gen
from matplotlib.lines import Line2D

'''
file structure and naming convention for videos:
video_dir
    -buoy1
        -buoy1_bitrate1.mp4
        -buoy1_bitrate2.mp4
        -buoy1_bitrate3.mp4
        ...
        -inference_buoy1.json # <-- output !

    -buoy2
        -buoy2_bitrate1.mp4
        -buoy2_bitrate2.mp4
        -buoy2_bitrate3.mp4
        .
        .
        .
        and so on

'''

# establish video paths
video_dir = '/home/jennaehnot/Desktop/annotated_videos'
expr = 'bitrate_exp'
buoys = ['lobsterPot']
bitrates = [1000] # in kbps

# establish yolo path and variables
vm_path = '/home/jennaehnot/Desktop/ccom_yolov5'
yolo_dir = vm_path
model_name = 'model3_4_best'
model_path = 'model_testing/model3_4_best.pt'
weights_path= os.path.join(yolo_dir, model_path)
img_sz = 1280

# other administrative business
imgs = 'images'
lbls = 'labels'
inf = 'inference'

# load model !
model = torch.hub.load(yolo_dir, 'custom', path = weights_path, source='local',force_reload=True)

### Run inference on each video frame by frame, log to dict, save as json for each video ###

for buoy in buoys:
    #save name for output json
    save_json_file = 'inference_' + buoy + '.json'
    inf_save_path = os.path.join(video_dir, expr, buoy, save_json_file)
    
    # dictionary for results
    buoy_inf_output = {}

    for bitrate in bitrates:
        # assemble path to buoy, video name, and path to video
        buoy_path = os.path.join(video_dir, expr, buoy)
        video_name = buoy + '_' + str(bitrate) + '.mp4'
        video_path = os.path.join(buoy_path, video_name)
        frame_inf_output = {}

        # okay now get in to it !
        try:
            vid = cv2.VideoCapture(video_path)
            count = 1 # frist frame number 

            while vid.isOpened():
                ret,frame = vid.read() # read next frame
                if frame is not None:
                    # perform validation
                    results = model(frame, size = img_sz)
                    # results.save()
                    times = list(results.t)
                    times.append(sum(times)) # times = [prep_t, infer_t, nms_t, total]
                    detects = results.pandas().xywhn[0]
                    detects = detects.values.tolist()
                    _, _, img_w, img_h = results.s
                    frame_inf_output[count] = {
                        'Img Dimensions': [img_sz, img_w, img_h],
                        'Time Stats': times,
                        'Detections': detects
                    }

                    count = count + 1
                else: 
                    # close video if there's no frames left
                    vid.release()
                    cv2.destroyAllWindows() 
            buoy_inf_output[str(bitrate) + ' kbps' ] = frame_inf_output
    
        except Exception as e:
            print(e)
        
    # save results
    with open(inf_save_path,'w') as json_file:
        json.dump(buoy_inf_output,json_file)

### COMPARE TO GROUNDTRUTH ANNOTATIONS ###

# define vars
miniou = 0.2
gt_paths = ['/home/jennaehnot/Desktop/annotated_videos/lobsterPot_140cmH/gt_labels_lobsterPot.json']

for i in range(0,len(buoys)):
    buoy = buoys[i]
    g = gt_paths[i]
    val_buoy = {}
    
    # load ground truth labels and inference labels
    with open(g,'r') as f:
        gt_lbls = json.load(f)
    with open(inf_save_path,'r') as f:
        buoy_inf = json.load(f)
    
    # for each bitrate
    for bitrate in bitrates:
        val = {}
        inf = buoy_inf[str(bitrate) + ' kbps']
        for key in inf: # for every detect in the video, look up ground truth
            detects = inf[key]['Detections']
            try:
                gt = gt_lbls[key] # future bug fix, lobsterPot frames are off by one
            except KeyError as e:
                print(f"Couldn't find a ground truth label for frame {key}")
                gt = []
            
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


  

