import cv2
import matplotlib.pyplot as plt
import os, torch, cv2, shutil, json, csv
import numpy as np
import pandas as pd
import prettytable
'''
Evaluating YOLOv5 model inference results
on frame from lobster pot video. In frame 1866,
buoy is at 20 m, in frame 1664 buoy is at 30 m.
In first section, squares from center of image are cropped
and inference is run with different imgsz arguments.
In second section, image is resized to different sizes
to different image resoltuions, and inference is run with 
different imgsz arguments. Results output to prettytables.
Future work: supress FutureWarnings from torch hub 
https://github.com/ultralytics/yolov5/pull/13244
'''

# paths n vars
vid_path = '/home/jennaehnot/Desktop/annotated_videos/lobsterPot_140cmH/lobsterPot_140cmH.mp4'
weights_path = '/home/jennaehnot/ccom_yolov5/model_testing/model4_3_best.pt'
yolo_path = '/home/jennaehnot/ccom_yolov5'
# load model
model = torch.hub.load(yolo_path, 'custom', path = weights_path, source='local',force_reload=True,verbose=False)

# open video
try:
    vidcap = cv2.VideoCapture(vid_path)
except Exception as e:
    print(f"Error opening video file: {e}")
    
else:
    # select frame from video
    frame_num = 1866 # 20 m 
    # frame_num = 1664 # 30 m

    # get frame  
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1) # cv2 is 0 indexed for frame nums
    success, image = vidcap.read()
    if not success:
        print(f"Failed to read frame {frame_num}")
    else:

        img_h, img_w, _ = image.shape
        print(f"Original image size: {img_w}x{img_h}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to rgb !!! cv2 is weird !!
        og_img = image.copy()

        ### Run inference on cropped squares ###
        
        # Define different imgsz arguments to test
        imgsz_args = [1920, 1080, 960, 800, 640, 480]        
        crop_sz = [0, 1080, 960, 800, 640, 480] # dimensions of central square crops, 0 = uncropped
        
        # init table 
        cropped_confTable = prettytable.PrettyTable(["imgsz arg \ Central Cropped Square Size"] + ['Uncropped', '1080', '960', '800', '640', '480'])
        cropped_timesTable = prettytable.PrettyTable(["imgsz arg \ Central Cropped Square Size"] + ['Uncropped', '1080', '960', '800', '640', '480'])
        
        for sz in imgsz_args:
            argsz = sz
            t_data = []
            t_data.append(argsz)
            c_data = []
            c_data.append(argsz)
            # evaluate cropped images with imgsz = arggsz
            for cropped in crop_sz:
                if cropped == 0:
                    cropped_img = og_img.copy() # without .copy() the image reverts to BGR and stacks all detcn results
                else:
                    # calc crop region
                    cx = img_w / 2
                    cy = img_h / 2
                    step = cropped / 2
                    x1 = int(cx - step)
                    y1 = int(cy - step)
                    x2 = int(cx + step)
                    y2 = int(cy + step)
                    if y1 < 0: # if dimension of square is larger than height of image
                        y1 = 0 # reference top of image
                    cropped_img = og_img[y1:y2, x1:x2].copy() # without .copy() the image reverts to BGR and stacks all detcn results
        
                # run inference
                results = model(cropped_img, size=argsz)
                # results.show()
                # parse results
                if results.pandas().xywhn[0].empty is True: # if no dets
                    c = [0]
                    t = sum(results.t)
                    t = [f"{t:.5f}"]
                    # print(f"Image Size Argument: {argsz}, Cropped Square: {cropped} --> No Detections")
                else:
                    c = list(results.pandas().xywhn[0]['confidence'])
                    c = [f"{n:.3f}" for n in c]
                    t = list(results.t)
                    t = sum(t)
                    t = [f"{t:.5f}"]
                    # print(f"Image Size Argument: {argsz}, Cropped Square: {cropped} --> Confidences and Times: {c}, {t}")

                c_data.append(c) # = c_data + c # c_data + c looks nicer but breaks table formatting if multiple detcs per img
                t_data.append(t) # = t_data + t # .append() allows nested lists for multiple detcs 

            # append results for this imgsz arg to tables
            cropped_confTable.add_row(c_data)
            cropped_timesTable.add_row(t_data)
        # display tables
        print(cropped_confTable)
        print(cropped_timesTable)

        ### Run inference for each combination of imgsz argument and image size ###

        imgsz_args = [1920, 1600, 1280, 960, 640]  # values for imgsz argument
        imgsz_h = [1088, 928, 736, 544, 384]  # corresponding heights for widths in imgsz_args, maintaining the aspect ratio of the original image (1920x1088)
        
        # init tables
        confTable = prettytable.PrettyTable(["imgsz arg \ True Image Dimensions"] + [f"({imgsz_args[i]}, {imgsz_h[i]})" for i in range(0,len(imgsz_args))])
        timesTable = prettytable.PrettyTable(["imgsz arg \ True Image Dimensions "] + [f"({imgsz_args[i]}, {imgsz_h[i]})" for i in range(0,len(imgsz_args))])

        for sz in imgsz_args:
            argsz = sz
            # init variables        
            t_data = []
            t_data.append(argsz)
            c_data = []
            c_data.append(argsz)

            for i in range(0,len(imgsz_h)):
                # Get new dimensions
                sz_h = imgsz_h[i]
                sz_w = imgsz_args[i]
                new_dims = (sz_w, sz_h)
                # Resize image
                new_img = cv2.resize(og_img, new_dims, interpolation=cv2.INTER_CUBIC)
                
                # run inference on resized image with imgsz argument = argsz
                results = model(new_img, size=argsz)
                # results.show()

                # parse results
                if results.pandas().xywhn[0].empty is True:
                    c = [0]
                    t = sum(results.t)
                    t = [f"{t:.5f}"]
                    # print(f"Image Size Argument: {argsz}, Image Dimensions: {new_dims} --> No Detections")
                else:
                    c = list(results.pandas().xywhn[0]['confidence'])
                    c = [f"{n:.3f}" for n in c]
                    t = list(results.t)
                    t = sum(t)
                    t = [f"{t:.5f}"]

                    # print(f"Image Size Argument: {argsz}, Image Dimensions: {new_dims} --> Confidences and Times: {r}")

                # append to results variables
                c_data = c_data + c
                t_data = t_data + t 
            # append results for this imgsz arg to tables
            confTable.add_row(c_data)
            timesTable.add_row(t_data)
        # display tables
        print(confTable)
        print(timesTable)

                

   