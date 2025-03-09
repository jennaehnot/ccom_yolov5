import val
import custom_funcs
import matplotlib.pyplot as plt
import os, torch, cv2, shutil, json
import numpy as np
import pandas as pd
from utils.metrics import bbox_iou
import utils.general as gen


''' STEP ONE: RUN INFERENCE ON VIDEO'''

# load yolo model
yolo_dir = '/home/jennaehnot/Desktop/ccom_yolov5'
weights_path= 'ccom_yolov5/model_testing/model3_4_best.pt'
# model = torch.hub.load('ccom_yolov5', 'custom', path = weights_path, source='local',force_reload=True)

# path to imgs, read filenames and sort them alphabetically
img_dir = '/home/jennaehnot/Desktop/ccom_yolov5/model_testing/redball_150cmH/images/val/'
img_filenames=os.listdir(img_dir)
img_filenames.sort()

# path to save inference dir
inf_dir = '/home/jennaehnot/Desktop/ccom_yolov5/model_testing/redball_150cmH/inference/'

# gt path
gt_lbl_dir = '/home/jennaehnot/Desktop/ccom_yolov5/model_testing/redball_150cmH/labels/val/'

# imgsz to run inference at
img_sz = [640, 960,1280, 1600, 1920]

# for sz in img_sz: #for every img size we want to run inference at
#     # make a new folder within the inference folder for each image size
#     imgsz_dir = 'imgsz' + str(sz)
#     img_save_dir = inf_dir + imgsz_dir
#     save_json_path = img_save_dir + '/inference_results.json'
    # output_stats = {}

    # for file in img_filenames:
    #     # run inference
    #     img_path = img_dir + file
    #     results = model(img_path, size= sz)
    #     results.save(labels=True, save_dir=img_save_dir,exist_ok =True) #exist_ok will rewrite existing folders instead of creating a new one so beware !
        
    #     times = list(results.t)
    #     times.append(sum(times)) # times = [prep_t, infer_t, nms_t, total]
    #     detects = results.pandas().xywhn[0]
    #     detects = detects.values.tolist()
    #     _, _, img_w, img_h = results.s

    #     output_stats[file] = {
    #         'Time Stats': times,
    #         'Detections xywhn:': detects,
    #         'Img Dimensions': [img_w, img_h]
    #     }
    
    # save stats 
    # with open(save_json_path,'w') as json_file:
    #     json.dump(output_stats,json_file)
   

# ### STEP TWO: COMPARE VALIDATION AND GROUND TRUTH LABELS
# read in json of inference. for each frame, compare inference results and ground
# truth labels and save correct inferencs to antoher json

img_sz = [640, 960,1280, 1600, 1920]

for sz in img_sz:

    pred_json = inf_dir +  'imgsz' + str(sz) + '/inference_results.json'  
    minconf = 0.6
    miniou = 0.2
    correct = [] 
    incorrect = []
    files = os.listdir(gt_lbl_dir)
    files.sort()

    with open(pred_json, 'r') as jsonfile:
        pred_data = json.load(jsonfile)
    validated={}
    #print(pred_data)
    for img_name in img_filenames: 
        frame_num = custom_funcs.imgname2frame(img_name)
        gt_lbl_path = gt_lbl_dir + img_name[:-4] + '.txt'
        gt_labels = custom_funcs.load_labels(gt_lbl_path)
        if gt_labels: #if theres an object in the img
            ## inf = custom_funcs.Inference(pred_data[img_name]) #inference data for that img
            inf = pred_data[img_name]
            if inf['Detections xywhn:']: # if the model detected something
                detects = inf['Detections xywhn:']
                # do iou for between all labels
                for i in range(0,len(gt_labels)):
                    gt_xywhn = gt_labels[i][1:]
                    for detect in detects:
                        d_xywhn = detect[0:4]

                        # do iou
                        if custom_funcs.compare_labels(gt_xywhn,d_xywhn, miniou): # compare labels returns false if boxes don't line up
                            validated[img_name] = inf
                            correct.append(img_name)
                        else:
                            incorrect.append(img_name)

    save_json_path = inf_dir +  'imgsz' + str(sz) + '/validated_results.json'
    with open(save_json_path,'w') as json_file:
         json.dump(validated,json_file)




