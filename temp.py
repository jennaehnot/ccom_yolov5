import detect
import matplotlib.pyplot as plt
import os, torch, shutil, csv, json
import numpy as np
import pandas as pd
from utils.metrics import bbox_iou
import utils.general as gen
import custom_funcs as cf
from matplotlib.lines import Line2D
# our goal here is to pars validated_results.json into a beautiful graph
csvpath='/home/field/jennaDir/dive_tools/annotated_vids/lobsterPot_140cmH/frames_dist_coor.csv'
frames, dists = cf.frames2distances(csvpath) 

img_sz = [640, 960,1280, 1600, 1920]
figure, axis = plt.subplots(len(img_sz), 1, figsize=(12,10), sharex=True, sharey=True)
for i in range(0,len(img_sz)):

    validated_path = '/home/field/jennaDir/dive_tools/annotated_vids/lobsterPot_140cmH/inference/imgsz' + str(img_sz[i]) + '/validated_results.json'    
    # plot validated results for all the frames in the frames array

    with open(validated_path, 'r') as jsonfile:
            val_data = json.load(jsonfile) 
    conf = np.empty((0,len(frames)))
    times = np.empty((0,len(frames)))
    clss = np.empty((0,len(frames)))

    for j in frames:
        #make file name
        img_name = 'frame_' + str(f"{j:05d}") + '.png'
        try:
            det = val_data[img_name]['Detections']
            c = det[0][4]
            conf = np.append(conf, c)

            id = det[0][5]
            clss = np.append(clss, id)

            t = val_data[img_name]['Time Stats']
            total_t = t[3]
            times = np.append(times,total_t)

        except KeyError:
            #print(f"{img_name} did not have a detection in it")
            conf = np.append(conf, np.nan)
            times = np.append(times, np.nan)
            clss = np.append(clss, -1)

    color_map = {0: 'red', 1: 'blue', 2: 'green', -1:'white'}
    colors = np.array([color_map[val] for val in clss])

    avg_t = np.nanmean(times)
    axis[i].scatter(dists,conf,s=2, c=colors)
    axis[i].grid(True, which='both', linestyle='--', color='gray', linewidth=0.5,alpha=0.5)
    axis[i].invert_xaxis()
    axis[i].set_ylim(0,1.0)
    axis[i].set_xlim(100,4)
    axis[i].set_yticks([0, 0.25, 0.5, 0.75, 1])
    axis[i].set_xticks(np.linspace(100,5,20))
    axis[i].set_title(f"Input Size = {img_sz[i]}," + r'  $\bar{t} = $'+ f"{avg_t:.2f} ms", fontsize = 10) 
    #axis[i].text(0.01, 0.95, f'Avg. t = {avg_t:.2f} ms', transform= axis[i].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='0: navBuoy'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='1: mooringBall'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='2: fishingBuoy')
]

# Add the legend to the plot
#
figure.suptitle('Detection of Approaching Lobster Pot at Different Compression Sizes', fontsize=18, y=0.95)
figure.text(0.5, 0.03, 'Buoy Distance from Camera', ha='center', va='center', fontsize=18)
figure.text(0.03, 0.5, 'Confidence', ha='center', va='center', rotation='vertical', fontsize=18)
#figure.legend(handles=legend_elements, loc = 'upper center', bbox_to_anchor=(0.5, 0.92), fontsize=10)
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], pad = 0.2)
#plt.subplots_adjust(top= 0.8)
plt.show()
print('fart')