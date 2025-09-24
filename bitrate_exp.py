import cv2 
import ccom_utils.custom_funcs as cf
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
video_dir = '/home/field/jennaDir/dive_tools/annotated_vids'
expr = 'bitrate_exp'
buoys = [ 'lobsterPot','mooringBall', 'redBall']
bitrates = [250, 500, 1000, 2500, 5000, 'native'] # in kbps

# establish yolo path and variables
vm_path = '/home/jennaehnot/Desktop/ccom_yolov5'
dhufish_path = '/home/field/jennaDir/ccom_yolov5'
yolo_dir = dhufish_path
model_path = 'model_testing/model4_3_best.pt'
weights_path= os.path.join(yolo_dir, model_path)
img_sz = 1920
iou_abrv = '05'

# other administrative business
plot_titles = ['White Mooring Ball'] #[ 'Lobster Pot', 'White Mooring Ball', 'Red Mooring Ball']
imgs = 'images'
lbls = 'labels'
inf = 'inference'
iou_abrv = '05'
model_name = 'model4_3_best'
model_abrv = 'm4'
date = '06232025'
run_inf = True
run_ver = False
run_plot = False


### Run inference on each video frame by frame, log to dict, save as json for each video ###
if run_inf == True:   
    # load model !
    model = torch.hub.load(yolo_dir, 'custom', path = weights_path, source='local',force_reload=True)
    for buoy in buoys:
        #save name for output json
        save_json_file = 'inference_' + buoy + '.json'
        inf_save_path = os.path.join(video_dir, expr, buoy, model_name, save_json_file)
        
        # dictionary for results
        buoy_inf_output = {}
        
        for bitrate in bitrates:
            # assemble path to buoy, video name, and path to video
            buoy_path = os.path.join(video_dir, expr, buoy)
            video_name = buoy + '_' + str(bitrate) + '.mp4'
            video_path = os.path.join(buoy_path, video_name)
            frame_inf_output = {}
            img_save_dir = os.path.join(video_dir,expr,buoy,model_name,'inference_images',str(bitrate)+'/')
            # okay now get in to it !
            if run_inf == True:
                try:
                    vid = cv2.VideoCapture(video_path)
                    count = 1 # frist frame number 

                    while vid.isOpened():
                        ret,frame = vid.read() # read next frame
                        if frame is not None:
                         
                            # perform inference
                            results = model(frame, size = img_sz)
                            #results.save(labels=True, save_dir=img_save_dir) #exist_ok will rewrite existing folders instead of creating a new one so beware !
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
        # with open(inf_save_path,'w') as json_file:
        #     json.dump(buoy_inf_output,json_file)

### COMPARE TO GROUNDTRUTH ANNOTATIONS ###

# define vars
miniou = 0.05

if run_ver == True:
    for i in range(0,len(buoys)):
        buoy = buoys[i]

        gt_file = model_abrv + '_gt_' + buoy + '.json'
        g = os.path.join(video_dir, expr, buoy, lbls, gt_file)

        inf_json_file = 'inference_' + buoy + '.json'
        inf_path = os.path.join(video_dir, expr, buoy, model_name, inf_json_file)
        
        # load ground truth labels and inference labels
        with open(g,'r') as f:
            gt_lbls = json.load(f)
        with open(inf_path,'r') as f:
            buoy_inf = json.load(f)
        
        verified_buoy = {}

        # for each bitrate
        for bitrate in bitrates:
            ver = {}
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
                            ver[key] = inf[key]
            verified_buoy[str(bitrate) + ' kbps'] = ver

        # save results for each buoy
        save_ver_json = 'verified_iou' + str(iou_abrv) + '_' + buoy + '.json'
        save_ver_path = os.path.join(video_dir, expr, buoy, model_name, save_ver_json)
        with open(save_ver_path, 'w') as jfile:
            json.dump(verified_buoy,jfile)


### PLOT THE HOUSE DOWN BOOTS ###
if run_plot == True:
    csvname = 'frames_dist_corr.csv'
    for m in range(0,len(buoys)):
        buoy = buoys[m]
        figure, axis = plt.subplots(len(bitrates), 1, figsize=(12,10), sharex=True, sharey=True)

        # pull distance and frame info
        plot_save_path = os.path.join(video_dir, expr, buoy, model_name, date, 'bitrate_results_iouThresh' + str(iou_abrv)+ '.png')
        csv_path = os.path.join(video_dir,expr,buoy,csvname)
        frames, dists = cf.frames2distances(csv_path)
        
        # load verified detects
        ver_json = os.path.join(video_dir, expr, buoy, model_name, 'verified_iou' + str(iou_abrv) + '_' + buoy + '.json')
        with open(ver_json, 'r') as jsonfile:
            ver_data = json.load(jsonfile)
        
        for i in range(0,len(bitrates)):
            bitrate = bitrates[i]
            inf = ver_data[str(bitrate) + ' kbps']
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

            color_map = {0: 'darkblue', 1: 'limegreen', -1:'white'}
            colors = np.array([color_map[v] for v in clss])
            avg_t = np.nanmean(times)
            axis[i].scatter(dists,conf,s=2, c=colors)
            axis[i].grid(True, which='both', linestyle='--', color='gray', linewidth=0.5,alpha=0.5)
            #axis[i].invert_xaxis()
            axis[i].set_ylim(0,1.0)
            axis[i].set_yticks([0, 0.25, 0.5, 0.75, 1])
            axis[i].set_xlim(4,100)
            axis[i].set_xticks(np.linspace(5,100,20))
            if bitrate == 'native' and buoy == 'lobsterPot':
                axis[i].set_title(f"Bitrate = 7460 kbps (native)," + r'  $\bar{t} = $'+ f"{avg_t:.2f} ms, IoU > {miniou}", fontsize = 14)
            elif bitrate == 'native' and buoy == 'redBall':
                axis[i].set_title(f"Bitrate = 6991 kbps (native)," + r'  $\bar{t} = $'+ f"{avg_t:.2f} ms, IoU > {miniou}", fontsize = 14)
            elif bitrate == 'native' and buoy == 'mooringBall':
                axis[i].set_title(f"Bitrate = 8435 kbps (native)," + r'  $\bar{t} = $'+ f"{avg_t:.2f} ms, IoU > {miniou}", fontsize = 14)
            else:
                axis[i].set_title(f"Bitrate = {str(bitrates[i])} kbps," + r'  $\bar{t} = $'+ f"{avg_t:.2f} ms, IoU > {miniou}", fontsize = 14) 

        legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', markersize=12, label='0: navBuoy'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen', markersize=12, label='1: buoy')]

        # Add the legend to the plot
        title = plot_titles[m]
        figure.suptitle(f'Detection of {title} at Varying Bitrates', fontsize=22, y=0.96)
        figure.text(0.5, 0.03, 'Distance from Camera to Buoy', ha='center', va='center', fontsize=18)
        figure.text(0.03, 0.5, 'Confidence', ha='center', va='center', rotation='vertical', fontsize=18)
        figure.legend(handles=legend_elements, loc='center', ncol=3, bbox_to_anchor=(0.5, 0.91))
        plt.tight_layout(rect=[0.04, 0.04, 0.95, 0.95])  
        
        figure.savefig(plot_save_path, dpi=400)
        


        # code to graph all bitrates on one plot as opposed to subplots            
        # # define some plot values
        # plt.figure(figsize=(10,5))
        # plt.xlabel('Buoy Distance to Camera')
        # plt.ylabel('Confidence')

        # ax = plt.gca()
        
        # minor_ticks = np.arange(0,100,1)
        # major_ticks = np.linspace(5,100,20)
        # ax.set_xticks(minor_ticks, minor=True)
        # ax.grid(True, which='minor', linestyle='-.', linewidth=0.5, alpha = 0.6)
        # ax.set_xticks(major_ticks)
        # ax.grid(True, which = 'major', linestyle='-',linewidth='1.0',color='darkgray',alpha=0.6)
        # ax.set_ylim(0,1)
        # ax.set_xlim(0,100)
        # #ax.invert_xaxis()

        # color_map = sb.color_palette('tab10',len(bitrates))
        # markers = ['o', '^', '*', 's', '>']

                
        #     plt.plot(dists,conf, color= color_map[i],linewidth=0.5, alpha = 0.3)    
        #     plt.scatter(dists, conf, s= 10, color = color_map[i], marker = markers[i], label = bitrate)
        # #plt.grid(True, which = 'both', linestyle= '-', linewidth = '0.75')
        # plt.legend()
        # plt.savefig(plot_save_path)


            

                

            
            