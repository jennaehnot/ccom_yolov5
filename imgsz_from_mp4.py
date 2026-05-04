# Imports
import val
import ccom_utils.custom_funcs as cf
import ccom_utils.detects as dfuncs
import ccom_utils.json_parser as jpars
import matplotlib.pyplot as plt
import os, torch, cv2, shutil, json, csv
import numpy as np
import pandas as pd
from utils.metrics import bbox_iou
import utils.general as gen
from matplotlib.lines import Line2D

'''
in:
    gt_path: path to ground truth labels
    video_path: path to video
    weights_path: path to yolo weights
    csv_dist_path: path to csv file with frame-distance correlations
    json save_path: path to save json results
    plt_save_path: path to save plot results
    img_sz: list of image sizes to test

'''

## PATHS

# path to videos
video_dir = '/home/field/jennaDir/jennaehnot/eval_videos'

# path to yolo and weights 
yolo_dir ='/home/field/jennaDir/ccom_yolov5' 
weights_path= os.path.join(yolo_dir, 'model_testing/model4_3_best.pt')

save_path = os.path.join(video_dir, 'imgsz_results')

## FILE NAMES

# gt name
gt_file = 'gt_labels.json'
# csv name
csv_name = 'frames_dist_corr.csv'

# json save name
json_save_name = 'imgsz_results.json'

# plt save name
plot_save_name = 'iouThresh05_imgsz_results.png' # will be added on to buoy_vid name

## OTHER PARAMS

# buoy dir names
buoy_vids = ['mooringBall', 'lobsterPot', 'redFender']

# plot title in order of buoy_vids
plot_titles = ['White Mooring Ball', 'Lobster Pot', 'Red Fender Buoy'] # for plot titles while saving

# image sizes to test
img_dims = ['640x384', '960x544', '1280x736', '1600x928', '1920x1088 (native)']
img_sz = [640, 960, 1280, 1600, 1920]

# alright now load the model!
model = torch.hub.load(yolo_dir, 'custom', path=weights_path, source='local', force_reload=True)


def run_inference():
    for i, buoy in enumerate(buoy_vids):
        buoy_inf_output = {} # create output json for buoy

        # make gt labels path
        gt_path = os.path.join(video_dir, buoy , gt_file)
        # load gt labels
        with open(gt_path, 'r') as f:
            gt_lbls = json.load(f)

        # make csv path and load frame-distance correlations
        csv_path = os.path.join(video_dir, buoy, csv_name)
        frames, dists = cf.frames2distances(csv_path)

        for sz in img_sz:

            # create video path
            vid = os.path.join(video_dir, buoy, buoy + '.mp4')

            try:
                # open video
                v = cv2.VideoCapture(vid)
                count = 1
                imgsz_output = {} # store results for each frame at this image size

                while v.isOpened():
                    print(f'Processing video {buoy} at size {sz}')

                    ret, imframe = v.read()  # read next frame
                    if imframe is not None:
                        # convert to RGB
                        rgb_frame = imframe[:, :, ::-1]
                        results = model(rgb_frame, size=sz)  # inference
                        
                        # pull yolo stats
                        times = list(results.t)
                        times.append(sum(times))  # times = [prep_t, infer_t, nms_t, total]
                        detects = results.pandas().xywhn[0]
                        detects = detects.values.tolist()
                        _, _, img_h, img_w = results.s

                        # pull distance from csv if available
                        if len(np.where(frames == count)[0]) > 0:
                            d = dists[np.where(frames == count)[0]][0]
                        else:
                            d = None
                        
                        # pull gt labels from gt json
                        if str(count) in gt_lbls.keys():
                            gt = gt_lbls[str(count)]
                        else:
                            gt = []

                        if detects != []:
                                if gt == []:
                                    print(f'No ground truth for frame {count} of {buoy} at {sz} px')
                                else:
                                    iouv = torch.linspace(0.05, 0.95, 19, device=None) # range of iou values
                                    correct_bb, correct_class, iou = dfuncs.compare_dets_gts(detects, gt, img_w, img_h, iouv)
                                    euc, euc_norm = dfuncs.euclidean_distance(detects, gt, img_w, img_h) # euclidean distance between dets and gts in pixels
                                    
                                    if correct_bb[0][0] == False: # if bbox is false, meaning IoU <0.05
                                        match = False
                                    else:
                                        match = True
                                    
                        else: # if no detects in frame
                            match = None
                            correct_bb = correct_class = iou = None
                            euc = euc_norm = None

                        imgsz_output[count] = {
                            'Img Dimensions': [img_w, img_h],
                            'Time Stats': times,
                            'Distance': d,
                            'Detections': detects,
                            'Ground Truth': gt,
                            'Bbox Match': match,
                            'Class Match': correct_class,
                            "IoU 0.05:0.95": correct_bb,
                            'IoU': iou,
                            'Euclidean Distance': [euc, euc_norm]
                            }
                        count += 1
                    else:
                        v.release()
                        cv2.destroyAllWindows()
                buoy_inf_output[str(sz)] = imgsz_output

            except Exception as e:
                print(f"Error processing {buoy} at image size {sz}: {e}")

        # save results
        inf_save_path = os.path.join(save_path, f'{buoy}_{json_save_name}')
        with open(inf_save_path, 'w') as json_file:
            json.dump(buoy_inf_output, json_file)


def make_plots():
    for q, buoy in enumerate(buoy_vids):
        
        infile = os.path.join(save_path, f'{buoy}_{json_save_name}')
        with open(infile, 'r') as f:
            data = json.load(f)

        # create plot
        figure, axis = plt.subplots(len(img_sz), 1, figsize=(12, 10), sharex=True, sharey=True)
        plot_save_path = os.path.join(save_path,'plots', f'{buoy}_{plot_save_name}')
        
        for i, sz in enumerate(img_sz):
            d = data[str(sz)]
            d = pd.DataFrame.from_dict(d, orient='index')
            # pull time stats
            tot_t = d['Time Stats']
            tot_t = [t[3] for t in tot_t] # total time for each frame
            avg_t = np.mean(tot_t)
            # pull img size
            img_w, img_h = d['Img Dimensions'][0]

            # filter dataframes
            filt1 = d[d['Distance'].notna()] # filtered by frames where distance is known
            filt2 = filt1[filt1['Detections'].apply(len) > 0] # filtered out frames with no detections
            d_fc = filt2[(filt2['Class Match'] == False) & (filt2['Bbox Match'] == True)] # filt to false class matches with bbox match
            d_fp = filt2[filt2['Bbox Match'] == False] # false positives
            plt_fc = jpars.restruct4plt(d_fc)
            plt_fp = jpars.restruct4plt(d_fp)

            plt_data = jpars.restruct4plt(filt2) # reconfigure dataframe for easier plotting
            axis[i].scatter(plt_data['Distance'], plt_data['Confidence'].tolist(), s=2, color='limegreen', label='True Positive')
            axis[i].scatter(plt_fc['Distance'], plt_fc['Confidence'].tolist(), s=2, color='darkblue', label='False Class Match')
            axis[i].scatter(plt_fp['Distance'], plt_fp['Confidence'].tolist(), s=2, color='red', label='False Positive (IoU < 0.05)')
            axis[i].grid(True, which='both', linestyle='--', color='gray', linewidth=0.5, alpha=0.5)
            axis[i].set_ylim(0, 1.0)
            axis[i].set_yticks([0, 0.25, 0.5, 0.75, 1])
            axis[i].set_xlim(4, 100)
            axis[i].set_xticks(np.linspace(5, 100, 20))
            axis[i].set_title(f"Rescaled size = {img_w}x{img_h}," + r'  $\bar{t} = $' + f"{avg_t:.2f} ms", fontsize=14)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen', label='True Positive', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', label='False Class Match', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='False Positive (IoU < 0.05)', markersize=10)]
        
        title = plot_titles[q]
        figure.suptitle(f'Detection of {title} Rescaled to Varying Image Sizes', fontsize=22, y=0.96)
        figure.text(0.5, 0.03, 'Distance from Camera to Buoy', ha='center', va='center', fontsize=18)
        figure.text(0.03, 0.5, 'Confidence', ha='center', va='center', rotation='vertical', fontsize=18)
        figure.legend(handles=legend_elements, loc='center', ncol=3, bbox_to_anchor=(0.5, 0.91))
        plt.tight_layout(rect=[0.04, 0.04, 0.95, 0.95])

        figure.savefig(plot_save_path, dpi=400)


#run_inference()
make_plots()
# def run_inference():
#     model = torch.hub.load(yolo_dir, 'custom', path = weights_path, source='local',force_reload=True)
#     for buoy in buoy_vids:

#         # dict for save results
#         buoy_inf_output = {}
#         vid = os.path.join(annotated_dir, buoy, buoy + '.mp4')
        
#         # load gt labels
#         gt_path = os.path.join(annotated_dir, buoy , f'labels/m4_gt_{buoy}.json')
#         with open(gt_path,'r') as f:
#             gt_lbls = json.load(f)
        
#         # load frame-distance correlations
#         csv_path = os.path.join('/home/field/jennaDir/dive_tools/annotated_vids/bitrate_exp', buoy , 'frames_dist_corr.csv')
#         frames, dists = cf.frames2distances(csv_path)
        
#         for sz in img_sz:
#             try:
#                 v = cv2.VideoCapture(vid)
#                 count = 1
#                 imgsz_output = {}

#                 while v.isOpened():
#                     print(f'Processing video {buoy} at size {sz}')

#                     ret,imframe = v.read() # read next frame
#                     if imframe is not None: 
#                         # convert to RGB
#                         rgb_frame = imframe[:,:,::-1]
#                         results = model(rgb_frame, size=sz)  # inference
                        
#                         # pull yolo stats
#                         times = list(results.t)
#                         times.append(sum(times)) # times = [prep_t, infer_t, nms_t, total]
#                         detects = results.pandas().xywhn[0]
#                         detects = detects.values.tolist()
#                         _, _, img_h, img_w = results.s

#                         # pull distance if available
#                         if len(np.where(frames == count)[0]) > 0:
#                             d = dists[np.where(frames == count)[0]][0]
#                         else:
#                             d = None
                        
#                         # pull gt labels
#                         if str(count) in gt_lbls.keys():
#                             gt = gt_lbls[str(count)]
#                         else:
#                             gt = []                       
                        
#                         if detects != []:
#                             if gt == []:
#                                 print(f'No ground truth for frame {count} of {buoy} at {sz} px')
#                             else:
#                                 iouv = torch.linspace(0.05,0.95,19,device=None) # range of iou values
#                                 correct_bb, correct_class, iou = dfuncs.compare_dets_gts(detects, gt, img_w, img_h, iouv)
#                                 euc, euc_norm = dfuncs.euclidean_distance(detects, gt, img_w, img_h)
                                
#                                 if correct_bb[0][0] == False: # if bbox is false, meaning IoU <0.05
#                                     match = False
#                                 else:
#                                     match = True
                                
#                         else: # if no detects in frame
#                             match = None
#                             correct_bb = correct_class = iou = None
#                             euc = euc_norm = None


#                         imgsz_output[count] = {
#                             'Img Dimensions': [img_w, img_h],
#                             'Time Stats': times,
#                             'Distance': d,
#                             'Detections': detects,
#                             'Ground Truth': gt,
#                             'Bbox Match': match,
#                             'Class Match': correct_class,
#                             "IoU 0.05:0.95": correct_bb,
#                             'IoU': iou,
#                             'Euclidean Distance': [euc, euc_norm]

#                         }
#                         count += 1

#                     else:
#                         v.release()
#                         cv2.destroyAllWindows()
                
#                 buoy_inf_output[sz] = imgsz_output
                
#             except Exception as e:
#                 print(e)
#                 print('womp womp')
            
#         # save output
#         out_path = os.path.join(annotated_dir, buoy, 'RGBimgsz_results.json')
#         with open(out_path, 'w') as f:
#             json.dump(buoy_inf_output, f)

# def make_plots():
#     for q, m in enumerate(buoy_vids):
        
#         infile = os.path.join(annotated_dir, m, 'RGBimgsz_results.json')
#         with open(infile,'r') as f:
#             data = json.load(f)
#         # create plot
#         figure, axis = plt.subplots(len(img_sz), 1, figsize=(12,10), sharex=True, sharey=True)
#         plot_save_path = os.path.join(annotated_dir, m, m + plot_save_name)
        
#         for i, sz in enumerate(img_sz):
#             d = data[str(sz)]
#             d = pd.DataFrame.from_dict(d, orient='index')
#             # pull time stats
#             tot_t = d['Time Stats']
#             tot_t = [t[3] for t in tot_t] # total time for each frame
#             avg_t = np.mean(tot_t)
#             # pull img size
#             img_w, img_h = d['Img Dimensions'][0]

#             # filter dataframes
#             filt1 = d[d['Distance'].notna()] # filtered by frames where distance is not known
#             filt2 = filt1[filt1['Detections'].apply(len) > 0] # filtered out frames with no detections
#             d_fc = filt2[(filt2['Class Match'] == False) & (filt2['Bbox Match'] == True)] # filt to false class matches with bbox match
#             d_fp = filt2[filt2['Bbox Match'] == False] # 
#             plt_fc = jpars.restruct4plt(d_fc)
#             plt_fp = jpars.restruct4plt(d_fp)


#             plt_data = jpars.restruct4plt(filt2) # reconfigure dataframe for easier plotting
#             axis[i].scatter(plt_data['Distance'], plt_data['Confidence'].tolist(), s = 2, color='limegreen', label = 'True Positive')
#             axis[i].scatter(plt_fc['Distance'], plt_fc['Confidence'].tolist(), s=2, color='darkblue', label='False Class Match')
#             axis[i].scatter(plt_fp['Distance'], plt_fp['Confidence'].tolist(), s=2, color='red', label='False Positive (IoU < 0.05)')
#             axis[i].grid(True, which='both', linestyle='--', color='gray', linewidth=0.5,alpha=0.5)
#             axis[i].set_ylim(0,1.0)
#             axis[i].set_yticks([0, 0.25, 0.5, 0.75, 1])
#             axis[i].set_xlim(4,100)
#             axis[i].set_xticks(np.linspace(5,100,20))
#             axis[i].set_title(f"Rescaled size = {img_w}x{img_h}," + r'  $\bar{t} = $'+ f"{avg_t:.2f} ms", fontsize = 14) 

#         legend_elements = [
#             Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen', label = 'True Positive', markersize=10),
#             Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', label='False Class Match', markersize=10),
#             Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='False Positive (IoU < 0.05)', markersize=10)]
        
#         title = plot_titles[q]
#         figure.suptitle(f'Detection of {title} Rescaled to Varying Image Sizes', fontsize=22, y=0.96)
#         figure.text(0.5, 0.03, 'Distance from Camera to Buoy', ha='center', va='center', fontsize=18)
#         figure.text(0.03, 0.5, 'Confidence', ha='center', va='center', rotation='vertical', fontsize=18)
#         figure.legend(handles=legend_elements, loc='center', ncol=3, bbox_to_anchor=(0.5, 0.91))
#         plt.tight_layout(rect=[0.04, 0.04, 0.95, 0.95])

#         figure.savefig(plot_save_path, dpi=400)


# #load yolo model 
# dhufish_path ='/home/field/jennaDir/ccom_yolov5'

# yolo_dir = dhufish_path
# weights_path= os.path.join(yolo_dir, 'model_testing/model4_3_best.pt')
# model_name = 'model4_3'

# # path to videos
# annotated_dir = '/home/field/jennaDir/test_vids'
# buoy_vids = ['mooringBall', 'lobsterPot','redBall']
# plot_titles = ['White Mooring Ball', 'Lobster Pot', 'Red Mooring Ball'] # for plot titles while saving

# # remember to change iou !
# plot_save_name = '_iouThresh05_RGBimgsz_results.png' # will be added on to buoy_vid name
# img_dims= ['640x384', '960x544', '1280x736', '1600x928', '1920x1088 (native)']
# #img sizes to run inference at
# img_sz = [640, 960,1280, 1600, 1920]

# make_plots()