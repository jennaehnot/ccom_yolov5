import cv2 
import custom_funcs as cf
import matplotlib.pyplot as plt
import os, torch, cv2, shutil, json, csv
import numpy as np
import pandas as pd
from utils.metrics import bbox_iou
import utils.general as gen
from matplotlib.lines import Line2D
import seaborn as sb

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

# lob_gt = '/home/field/jennaDir/dive_tools/annotated_vids/lobsterPot_140cmH/labels'
# lob_save = '/home/field/jennaDir/dive_tools/annotated_vids/bitrate_exp/lobsterPot/gt_labels_lobsterPot.json'
# moor_gt = '/home/field/jennaDir/dive_tools/annotated_vids/mooringBall_115cmH/labels'
# moor_save = '/home/field/jennaDir/dive_tools/annotated_vids/bitrate_exp/mooringBall/gt_labels_mooringBall.json'
# red = '/home/field/jennaDir/dive_tools/annotated_vids/redBall_150cmH/labels'
# red_save = '/home/field/jennaDir/dive_tools/annotated_vids/bitrate_exp/redBall/gt_labels_redBall.json'

# cf.create_gt_json(lob_gt,lob_save)
# cf.create_gt_json(moor_gt, moor_save)
# cf.create_gt_json(red, red_save)

# establish video paths
video_dir = '/home/field/jennaDir/dive_tools/annotated_vids'
expr = 'bitrate_exp'
buoys = ['lobsterPot', 'mooringBall', 'redBall']
bitrates = [250, 500, 1000, 2000, 3000] # in kbps

# establish yolo path and variables
vm_path = '/home/jennaehnot/Desktop/ccom_yolov5'
dhufish_path = '/home/field/jennaDir/ccom_yolov5'
yolo_dir = dhufish_path
model_name = 'model3_4_best'
model_path = 'model_testing/model3_4_best.pt'
weights_path= os.path.join(yolo_dir, model_path)
img_sz = 1280

# other administrative business
imgs = 'images'
lbls = 'labels'
inf = 'inference'
run_inf = False
run_val = True
run_plot = True
# load model !
# model = torch.hub.load(yolo_dir, 'custom', path = weights_path, source='local',force_reload=True)

### Run inference on each video frame by frame, log to dict, save as json for each video ###

for buoy in buoys:
    #save name for output json
    save_json_file = 'inference_' + buoy + '.json'
    inf_save_path = os.path.join(video_dir, expr, buoy, save_json_file)
    
    # dictionary for results
    buoy_inf_output = {}
    bitrates.sort(reverse=True)

    for bitrate in bitrates:
        # assemble path to buoy, video name, and path to video
        buoy_path = os.path.join(video_dir, expr, buoy)
        video_name = buoy + '_' + str(bitrate) + '.mp4'
        video_path = os.path.join(buoy_path, video_name)
        frame_inf_output = {}

        # okay now get in to it !
        if run_inf == True:
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
    if run_inf == True:        
        # save results
        with open(inf_save_path,'w') as json_file:
            json.dump(buoy_inf_output,json_file)

### COMPARE TO GROUNDTRUTH ANNOTATIONS ###

# define vars
miniou = 0.0
if run_val == True:
    for i in range(0,len(buoys)):
        buoy = buoys[i]

        gt_file = 'gt_labels_' + buoy + '.json'
        g = os.path.join(video_dir, expr, buoy, gt_file)

        inf_json_file = 'inference_' + buoy + '.json'
        inf_path = os.path.join(video_dir, expr, buoy, inf_json_file)
        
        # load ground truth labels and inference labels
        with open(g,'r') as f:
            gt_lbls = json.load(f)
        with open(inf_path,'r') as f:
            buoy_inf = json.load(f)
        
        val_buoy = {}

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


### PLOT THE HOUSE DOWN BOOTS ###
if run_plot == True:
    csvname = 'frames_dist_corr.csv'
    for buoy in buoys:
        # pull distance and frame info
        plot_save_path = os.path.join(video_dir, expr, buoy, 'results' + str(miniou)+ '.png')
        csv_path = os.path.join(video_dir,expr,buoy,csvname)
        frames, dists = cf.frames2distances(csv_path)
        

        # load validated data
        val_json = os.path.join(video_dir, expr, buoy, 'validated_' + buoy + '.json')
        with open(val_json, 'r') as jsonfile:
            val_data = json.load(jsonfile)
        
        # define some plot values
        plt.figure(figsize=(10,5))
        plt.xlabel('Buoy Distance to Camera')
        plt.ylabel('Confidence')

        ax = plt.gca()
        
        minor_ticks = np.arange(0,100,1)
        major_ticks = np.linspace(5,100,20)
        ax.set_xticks(minor_ticks, minor=True)
        ax.grid(True, which='minor', linestyle='-.', linewidth=0.5, alpha = 0.6)
        ax.set_xticks(major_ticks)
        ax.grid(True, which = 'major', linestyle='-',linewidth='1.0',color='darkgray',alpha=0.6)
        ax.set_ylim(0,1)
        ax.set_xlim(0,100)
        #ax.invert_xaxis()

        color_map = sb.color_palette('tab10',len(bitrates))
        markers = ['o', '^', '*', 's', '>']

        for i in range(0,len(bitrates)):
            bitrate = bitrates[i]
            inf = val_data[str(bitrate) + ' kbps']
            conf = np.empty((0,len(frames)))
            times = np.empty((0,len(frames)))
            clss = np.empty((0,len(frames)))
            for frame in frames:
                try:
                    det = inf[str(frame)]['Detections'] 
                    c = det[0][4]
                    conf = np.append(conf,c)

                    id = det[0][5]
                    clss = np.append(clss,id)

                    t = inf[str(frame)]['Time Stats']
                    total_t = t[3]
                    times = np.append(times,total_t)

                except KeyError:
                    conf = np.append(conf, np.nan)
                    times = np.append(times,np.nan)
                    clss = np.append(clss, -1)
                
            plt.plot(dists,conf, color= color_map[i],linewidth=0.5, alpha = 0.3)    
            plt.scatter(dists, conf, s= 10, color = color_map[i], marker = markers[i], label = bitrate)
        #plt.grid(True, which = 'both', linestyle= '-', linewidth = '0.75')
        plt.legend()
        plt.savefig(plot_save_path)


            

                

            
            