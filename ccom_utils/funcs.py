import cv2
import matplotlib.pyplot as plt
import os, torch, cv2, shutil, json, csv
import numpy as np
import pandas as pd
from ccom_utils.imgs import draw_boxes
from utils.general import xywhn2xyxy

'''
take video, results json, display false positives
'''

def display_falsepos(vid_path, jsonpath, savepath=None):
    try:
        with open(jsonpath, 'r') as f:
            jdata = json.load(f)

        vidcap = cv2.VideoCapture(vid_path)
    except Exception as e:
        print(f"Error opening video file: {e}")
        return
    
    else:
        
        for key in jdata.keys():
            param = key
            viddata = jdata[key]
            for frame in viddata.keys():
                frame_data = viddata[frame]
                if frame_data['Bbox Match'] == False:
                    frame_num = int(frame)
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                    success, image = vidcap.read()
                    if not success:
                        print(f"Failed to read frame {frame_num}")
                    
                    else:
                        # Convert BGR to RGB
                        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        img_h, img_w, _ = image.shape
                        
                        im = image.copy()
                        # Extract bbox coordinates
                        detect = frame_data['Detections']
                        d = frame_data['Distance']

                        if d is not None:
                            d = round(d,2)
                            cv2.putText(im, f"Frame: {frame_num} Dist: {d} m", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                            for det in detect:
                                xywh = det[:4]
                                xywh = [round(n, 4) for n in xywh]
                                c = det[4]
                                id = det[5]
                                # Convert to xyxy format
                                xyxy = xywhn2xyxy(torch.tensor([xywh]), img_w, img_h).numpy()[0]
                                # DRAW BBOX ON IMAGE
                                im = draw_boxes(im, xyxy, c, id, bool=False)
                            gts = frame_data['Ground Truth']
                            for g in gts:
                                xywh = g[1:5]
                                xywh = [round(n, 4) for n in xywh]
                                id = g[0]
                                c = 'GT'
                                # Convert to xyxy format
                                xyxy = xywhn2xyxy(torch.tensor([xywh]), img_w, img_h).numpy()[0]
                                # DRAW GT BBOX ON IMAGE IN GREEN
                                im = draw_boxes(im, xyxy,c, id, bool=True)
                            
                            if savepath:
                                savefile = os.path.join(savepath, param + f"_{d}m.png")
                                if not os.path.exists(savepath):
                                    os.makedirs(savepath)
                                cv2.imwrite(savefile, im)
                            else:
                                cv2.imshow(im)
                                print("Close the image window to continue...")

                        cv2.destroyAllWindows()

        vidcap.release()

def get_anchorboxes(model):
    try:
        model = model['model']
        if hasattr(model, 'model'):
            m = model.model
        else:
            m = model
        anchors = m[-1].anchors
        print("returning anchors in grid cell units")
        return(anchors)

    except Exception as e:
        print("Had an issue getting anchors, are you loading with with torch.load()?")
        print(e)
        return(None)

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
