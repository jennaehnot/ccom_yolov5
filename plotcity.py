import os, numpy, seaborn, json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ccom_yolov5.ccom_utils.custom_funcs as cf
from matplotlib.lines import Line2D

video_dir = '/home/field/jennaDir/dive_tools/annotated_vids'
expr = 'bitrate_exp'
mdl = 'model4_3_best'
m_abrv = 'm4'
buoys = ['lobsterPot', 'redBall', 'mooringBall', ] #
bitrates = [250, 500, 1000, 2500, 5000, 'native']#] # in kbps
iouv = np.arange(0,1,0.15) # varied ious
date = '06242025'
#sizes = [20, 17, 14, 11, 8, 5, 2]
# shapes = ['o','x', 's', 'p', '^','v','*']
 
buoy_plot_name = ['Lobster Pot',  'Red Mooring Ball','White Mooring Ball'] #

for ii in range(0,len(buoys)):
    buoy = buoys[ii]

    figure, axis = plt.subplots(len(bitrates), 1, figsize=(12,10), sharex=True, sharey=True)
    iou_json = os.path.join(video_dir, expr, buoy, 'iouStats_' + buoy + ".json")
    
    with open(iou_json, 'r') as j:
        data = json.load(j)

    for q in range(0,len(bitrates)):
        bitrate = bitrates[q]
        b = data[str(bitrate) + ' kbps']
        #colors = plt.cm.dark2(np.linspace(0, 1, len(iouv)))
        colors = ['black', 'saddlebrown','darkviolet','mediumblue','limegreen', 'gold', 'red',]
        confs = []
        dists = []
        ious = []
        buul = []
        euc = []
        scatter_plots = []

        for key in b: # for each frame
            d = b[key] 
            detects = d['Detections']
            # handle possible case of multiple deetcts in one frame            
            for idx in range(0,len(detects)):
                if d['Distance'] != []: # last frames when buoy barely in frames, no dist data
                    detect = detects[idx]
                    confs.append(detect[4])
                    ious.append(d['IoU'][0][idx])
                    dists.append(d['Distance'][0])
                    buul.append(d['Class Match'][0][idx])
                    euc.append(d['Euclidean Distance'])

        df = pd.DataFrame({'distance': dists, 'c': confs, 'iou': ious, 'boolean': buul, 'euclidean': euc})
        df_true = df[df['boolean']== True]
        df_false = df[df['boolean'] == False]
        df_loweuc = df[df['euclidean'] <= 30]
        df_far = df[df['euclidean'] > 30]
    ### Historgram
        axis[q].set_title(f"Bitrate = {str(bitrate)} kbps")
        axis[q].hist(df_true['euclidean'])
    figure.suptitle(f'Eucilean Distance Hist of {buoy_plot_name[ii]}', fontsize=22, y=0.96)
    


    ### CONF vs Euc (only true classes)
    #     scat = axis[q].scatter(df_true['euclidean'], df_true['iou'], c=df_true['c'], cmap='jet',s=10)
    #     scatter_plots.append(scat)
    #     #axis[q].scatter(df_true['euclidean'],df_true['iou'],color='limegreen',marker='o',s=10)
    #     axis[q].scatter(df_false['euclidean'],df_false['iou'],color='black',marker='x',s=10)
    #     axis[q].set_title(f"Bitrate = {str(bitrate)} kbps")
    #     axis[q].grid(True, which='both', linestyle='--', color='gray', linewidth=0.5,alpha=0.5)
    #     axis[q].set_ylim(0,1.0)
    #     axis[q].set_yticks([0, 0.25, 0.5, 0.75, 1])
    #     #axis[q].set_xticks(np.arange(0,1,0.1))
    # plt.tight_layout(rect=[0.04, 0.04, 0.95, 0.95])
    # cbar = figure.colorbar(scatter_plots[0], ax=axis, orientation='vertical', pad=0.01  )
    # cbar.set_label('Confidence',rotation=270, labelpad=15, fontsize=12 )
    # figure.suptitle(f'Eucilean Distance vs IoU of {buoy_plot_name[ii]}', fontsize=22, y=0.96)
    # figure.text(0.5, 0.03, 'Euclidean Distance', ha='center', va='center', fontsize=18)
    # figure.text(0.03, 0.5, 'IoU', ha='center', va='center', rotation='vertical', fontsize=18)
    # figure.savefig(os.path.join(video_dir,expr,'plotcity', date,f"{buoy}_IOUxEuclideanxConf.png"))


    ### confidence by distance colored by euclidean distance

    #     scat = axis[q].scatter(df_true['distance'], df_true['c'], c=df_true['euclidean'], cmap='coolwarm',s=10,vmin=0, vmax=10)
    #     scatter_plots.append(scat)
    #     # title subplots
    #     if bitrate == 'native':
    #         if buoy == 'lobsterPot':
    #             axis[q].set_title(f"Bitrate = 7460 kbps (native)", fontsize = 14)
    #         elif buoy == 'mooringBall':
    #             axis[q].set_title(f"Bitrate = 8435 kbps (native)", fontsize = 14)
    #         elif buoy == 'redBall':
    #             axis[q].set_title(f"Bitrate = 6991 kbps (native)", fontsize = 14)
    #     else:
    #         axis[q].set_title(f"Bitrate = {str(bitrate)} kbps")

    #     axis[q].grid(True, which='both', linestyle='--', color='gray', linewidth=0.5,alpha=0.5)
    #     axis[q].set_ylim(0,1.0)
    #     axis[q].set_yticks([0, 0.25, 0.5, 0.75, 1])
    #     #axis[q].set_xlim(4,100)
    #     axis[q].set_xticks(np.linspace(5,100,20))
    # plt.tight_layout(rect=[0.04, 0.04, 0.95, 0.95])
    # cbar = figure.colorbar(scatter_plots[0], ax=axis, orientation='vertical', pad=0.01  )
    # cbar.set_label('Euclidean',rotation=270, labelpad=15, fontsize=12 )
    # # bitrate agnostic plot things
    # figure.suptitle(f'Detection of {buoy_plot_name[ii]} at Varying Bitrates', fontsize=22, y=0.96)
    # figure.text(0.5, 0.03, 'Distance from Camera to Buoy', ha='center', va='center', fontsize=18)
    # figure.text(0.03, 0.5, 'Confidence', ha='center', va='center', rotation='vertical', fontsize=18)
    #figure.savefig(os.path.join(video_dir,expr,'plotcity',date, f"{buoy}_confxDistxEuclidean_pixels.png"))

    ### CONF vs Euc (only true classes)
    #     axis[q].scatter(df_true['euclidean'],df_true['c'],color='limegreen',marker='o',s=10)
    #     axis[q].scatter(df_false['euclidean'],df_false['c'],color='r',marker='x',s=10)
    #     axis[q].set_title(f"Bitrate = {str(bitrate)} kbps")
    #     axis[q].grid(True, which='both', linestyle='--', color='gray', linewidth=0.5,alpha=0.5)
    #     axis[q].set_ylim(0,1.0)
    #     axis[q].set_yticks([0, 0.25, 0.5, 0.75, 1])
    #     #axis[q].set_xticks(np.arange(0,1,0.1))
    # figure.suptitle(f'Eucilean Distance vs Confidence for Detections of {buoy_plot_name[ii]}', fontsize=22, y=0.96)
    # figure.text(0.5, 0.03, 'Euclidean Distance', ha='center', va='center', fontsize=18)
    # figure.text(0.03, 0.5, 'Confidence', ha='center', va='center', rotation='vertical', fontsize=18)
    # plt.tight_layout(rect=[0.04, 0.04, 0.95, 0.95])
    # figure.savefig(os.path.join(video_dir,expr,'plotcity', date,f"{buoy}_confxEuclidean.png"))


    ### CONF vs IOU
    #     axis[q].scatter(df_true['iou'],df_true['c'],color='limegreen',marker='o',s=10)
    #     axis[q].scatter(df_false['iou'],df_false['c'],color='r',marker='x',s=10)
    #     axis[q].set_title(f"Bitrate = {str(bitrate)} kbps")
    #     axis[q].grid(True, which='both', linestyle='--', color='gray', linewidth=0.5,alpha=0.5)
    #     axis[q].set_ylim(0,1.0)
    #     axis[q].set_yticks([0, 0.25, 0.5, 0.75, 1])
    #     axis[q].set_xticks(np.arange(0,1,0.1))
    # figure.suptitle(f'IoU vs Confidence for Detections of {buoy_plot_name[ii]}', fontsize=22, y=0.96)
    # figure.text(0.5, 0.03, 'IoU', ha='center', va='center', fontsize=18)
    # figure.text(0.03, 0.5, 'Confidence', ha='center', va='center', rotation='vertical', fontsize=18)
    # plt.tight_layout(rect=[0.04, 0.04, 0.95, 0.95])
    # figure.savefig(os.path.join(video_dir,expr,'plotcity', date,f"{buoy}_confxIoU.png"))

    ### CONF vs IOU COLORED BY DISTANCE (correct classes only)
    #     scatter = axis[q].scatter(df_true['iou'],df_true['c'],c=df_true['distance'],cmap='coolwarm',s=10)
    #     scatter_plots.append(scatter)
    #     axis[q].set_title(f"Bitrate = {str(bitrate)} kbps")
    #     axis[q].grid(True, which='both', linestyle='--', color='gray', linewidth=0.5,alpha=0.5)
    #     axis[q].set_ylim(0,1.0)
    #     axis[q].set_yticks([0, 0.25, 0.5, 0.75, 1])
    #     axis[q].set_xticks(np.arange(0,1,0.1))
    # figure.suptitle(f'IoU vs Confidence for Detections of {buoy_plot_name[ii]}', fontsize=22, y=0.96)
    # figure.text(0.5, 0.03, 'IoU', ha='center', va='center', fontsize=18)
    # figure.text(0.03, 0.5, 'Confidence', ha='center', va='center', rotation='vertical', fontsize=18)
    # plt.tight_layout(rect=[0.04, 0.04, 0.98, 0.98])
    # cbar = figure.colorbar(scatter_plots[0], ax=axis, orientation='vertical', pad=0.01  )
    # cbar.set_label('Distance to Camera',rotation=270, labelpad=15, fontsize=12 )
    # figure.savefig(os.path.join(video_dir,expr,'plotcity', date,f"{buoy}_confxIoUxDist.png"))
 
    # # COlORED BY IOU 
    #     # now plot iou threshholds for each bitrate
    #     for i, threshold in enumerate(iouv):
    #         filtered_data = df[df['iou'] > threshold]
    #         axis[q].scatter(filtered_data['distance'], filtered_data['c'], s= 14, color=colors[i])

    #     # title subplots
    #     if bitrate == 'native':
    #         if buoy == 'lobsterPot':
    #             axis[q].set_title(f"Bitrate = 7460 kbps (native)", fontsize = 14)
    #         elif buoy == 'mooringBall':
    #             axis[q].set_title(f"Bitrate = 8435 kbps (native)", fontsize = 14)
    #         elif buoy == 'redBall':
    #             axis[q].set_title(f"Bitrate = 6991 kbps (native)", fontsize = 14)
    #     else:
    #         axis[q].set_title(f"Bitrate = {str(bitrate)} kbps")

    #     axis[q].grid(True, which='both', linestyle='--', color='gray', linewidth=0.5,alpha=0.5)
    #     axis[q].set_ylim(0,1.0)
    #     axis[q].set_yticks([0, 0.25, 0.5, 0.75, 1])
    #     #axis[q].set_xlim(4,100)
    #     axis[q].set_xticks(np.linspace(5,100,20))
    
    # # bitrate agnostic plot things
    # legend_elements = []
    # for i in range(0,len(colors)):
    #     legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor= colors[i], markersize=12, label=f"IoU Threshold > {iouv[i]:.02f}" ))
    # figure.legend(handles=legend_elements, loc='center', ncol=1, bbox_to_anchor=(0.8, 0.8))
    # figure.suptitle(f'Detection of {buoy_plot_name[ii]} at Varying Bitrates', fontsize=22, y=0.96)
    # figure.text(0.5, 0.03, 'Distance from Camera to Buoy', ha='center', va='center', fontsize=18)
    # figure.text(0.03, 0.5, 'Confidence', ha='center', va='center', rotation='vertical', fontsize=18)
    # figure.savefig(os.path.join(video_dir,expr,'plotcity',date, f"{buoy}_bitrate_coloredIoU.png"))

            