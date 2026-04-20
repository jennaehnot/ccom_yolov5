import cv2, json, torch, os
import ccom_utils.json_parser as jpars
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from skimage.measure import shannon_entropy as entropy
from utils.general import xywhn2xyxy
import numpy as np
import pandas as pd

# img = cv2.imread("testertest.jpg")
# im = img.tolist()


### bitrate results
buoys = ['mooringBall', 'lobsterPot','redBall']
def run_inference():
    for buoy in buoys:

        vid_path = f'/home/field/jennaDir/test_vids/{buoy}/{buoy}.mp4'
        json_path = f'/home/field/jennaDir/test_vids/{buoy}/RGBimgsz_results.json'

        try:
            with open(json_path,'r') as f:
                jdata = json.load(f)

        
            entpy_json = {}
            for key in jdata.keys():
                # open video 
                vidcap = cv2.VideoCapture(vid_path)
                print(f'Processing video {buoy} at {key} ')
                count = 1
                vid_data = jdata[key]
                new_vid_data = {}
                while vidcap.isOpened():

                    

                ## for frame in vid_data.keys(): # using frame to seek to frame in video, slow method
                    
                    success, image = vidcap.read()
                    if image is None:
                        print(f"Failed to read frame {count} or EOF")
                        vidcap.release()
                        cv2.destroyAllWindows()
                    else:

                        frame = str(count)
                        frame_num = count
                        frame_data = vid_data[frame]
                        
                        if (frame_data['Detections'] != []) & (frame_data['Distance'] is not None):

                            # convert to grayscale
                            im = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                            # pull other vars
                            img_h, img_w, _ = image.shape
                            detect = frame_data['Detections']
                            e = []
                            a = []
                            eofp = []
                            conf = []
                            #crop img
                            for det in detect:
                                xywh = det[:4]
                                xywh = [round(n, 4) for n in xywh]
                                c = det[4]
                                id = det[5]

                                # Convert to xyxy format
                                xyxy = xywhn2xyxy(torch.tensor([xywh]), img_w, img_h).numpy()[0]
                                x1, y1, x2, y2 = [round(a) for a in xyxy]
                        
                                # crop image
                                cropped = im[y1:y2, x1:x2]
                                if cropped.shape != (0,0): #verify crop worked
                                    # calc entropy
                                    e.append(entropy(cropped))
                                    a.append((x2 - x1)*(y2 - y1)) # entropy in bits
                                    eofp.append( entropy(cropped) / (x2 - x1)*(y2 - y1) ) 
                                    conf.append(c)

                                else:
                                    print("Image crop failed! Check dimensions of bbox")
                            
                            # save data
                            new_frame = frame_data
                            new_frame['Entropy'] = e
                            new_frame['Bbox Area'] = a
                            new_frame['Entropy per Pixel'] = eofp
                            new_frame['Confidence'] = conf
                            # pack dict for this video
                            new_vid_data[frame] = new_frame
                        count +=1

                entpy_json[key] = new_vid_data
                print(f"Completed {key} kbps")

                vidcap.release()
                cv2.destroyAllWindows()

            #save_path = f'/home/field/jennaDir/test_vids/{buoy}/bitrate_entropy_results.json'
            save_path = f'/home/field/jennaDir/test_vids/{buoy}/imgsz_entropy_results.json'
            with open(save_path, 'w') as jf:
                json.dump(entpy_json,jf)
                print(f"Sucessfully saved to {jf}")

        except Exception as e:
            print(f"There was an error: {e}")


def make_plots(expr):
    
    
    if expr == 'bitrate':
        jname = 'bitrate_entropy_results.json'
        parameter = [250, 500, 1000, 2500, 5000, 'native'] # in kbps
        plot_save_name = 'bitrate_ePerP_plt.png'
        
    elif expr == 'imgsz':
        jname = 'imgsz_entropy_results.json'
        plot_save_name = 'imgsz_ePerP_plt.png'
        parameter = [640, 960,1280, 1600, 1920]
        img_dims= ['640x384', '960x544', '1280x736', '1600x928', '1920x1088 (native)']
    else:
        print("Please set expr to 'bitrate' or 'imgsz'")
        return

    
    plot_titles = ['White Mooring Ball', 'Lobster Pot', 'Red Mooring Ball'] # for plot titles while saving
    buoys = ['mooringBall', 'lobsterPot','redBall']
    save_path = '/home/field/jennaDir/test_vids/'

    for q, buoy in enumerate(buoys):
       
        # load data file
        infile = os.path.join(save_path, buoy, jname)
        with open(infile,'r') as f:
            data = json.load(f)

        # create figure and save path
        figure, axis = plt.subplots(len(parameter), 1, figsize=(12,10), sharex=True, sharey=True)
        plot_save_path = os.path.join(save_path, buoy, 'ePerP_plots', buoy + '_' + plot_save_name)
        scatter_plots = []

        # first pull data from file and massage
        all_entropy_values = []

        for i, param in enumerate(parameter):

            d = data[str(param)]
            df = pd.DataFrame.from_dict(d, orient='index')
            
            df_e =df.explode('Entropy per Pixel')
            df_e =jpars.restruct4plt(df_e)
            all_entropy_values.extend(df_e['Entropy per Pixel'].tolist())
    
        if all_entropy_values:
            vmin, vmax = min(all_entropy_values), max(all_entropy_values)
            norm = Normalize(vmin=vmin, vmax=vmax)
            #avg_e = np.mean(all_entropy_values)

        # now lets make the plots
        for i, param in enumerate(parameter):
            ax = axis[i]
            d = data[str(param)]
            df = pd.DataFrame.from_dict(d, orient='index')
            df = df.explode(['Entropy per Pixel', 'Confidence', 'Detections'])
            d =jpars.restruct4plt(df)
            d_e =d['Entropy per Pixel'].tolist()
            avg_e = np.mean(d_e)

            
            # this is broken bad but eventually filter for TP, FP, and FC (false classification)
            # if not plt_fp.empty: # if there are false positives
            #     scat_fp = ax.scatter(plt_fp['Distance'], plt_fp['Confidence'], 
            #                         c = plt_fp['Entropy'], cmap='viridis', norm = norm, marker='x', s = 12)
            # if not plt_fc.empty: # if there are false positives
            #     scat_fc = ax.scatter(plt_fc['Distance'], plt_fc['Confidence'], 
            #                         c = plt_fc['Entropy'], cmap='viridis', norm = norm, marker='^', s = 12)
            # if not plt_tp.empty: # if there are false positives
            #     scat_tp = ax.scatter(plt_tp['Distance'], plt_tp['Confidence'], 
            #                         c = plt_tp['Entropy'], cmap='viridis', norm = norm, marker='o', s = 12)

            scat = ax.scatter(d['Distance'], d['Confidence'].tolist(), 
                              c=d['Entropy per Pixel'], cmap='viridis', norm=norm, marker='o', s=10)
            scatter_plots.append(scat)
            ax.grid(True, which='both', linestyle='--', color='gray', linewidth=0.5,alpha=0.5)
            ax.set_ylim(0,1.0)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xlim(4,100)
            ax.set_xticks(np.linspace(5,100,20))

            if expr == 'imgsz':
                ax.set_title(f"Image Size = {img_dims[i]}, Mean Entropy per Pixel = {avg_e:.2f}")
            elif expr == 'bitrate':
                if param == 'native' and buoy == 'lobsterPot':
                    ax.set_title(f"Bitrate = 7460 kbps (native), Mean Entropy per Pixel = {avg_e:.2f}")
                elif param == 'native' and buoy == 'mooringBall':
                    ax.set_title(f"Bitrate = 8435 kbps (native), Mean Entropy per Pixel = {avg_e:.2f}")
                elif param == 'native' and buoy == 'redBall':
                    ax.set_title(f"Bitrate = 6991 kbps (native), Mean Entropy per Pixel = {avg_e:.2f}")
                else:
                    ax.set_title(f"Bitrate = {param} kbps, Mean Entropy per Pixel = {avg_e:.2f}")
            

        plt.tight_layout(rect=[0.04, 0.04, 0.98, 0.95])
        cbar = figure.colorbar(scatter_plots[0], ax=axis, orientation='vertical', pad=0.02  )
        cbar.set_label('Entropy per Pixel',rotation=270, labelpad=15, fontsize=12 )
        figure.suptitle(f'Detection of {plot_titles[q]} at Varying Image Sizes', fontsize=22, y=0.96)
        figure.text(0.5, 0.03, 'Distance from Camera to Buoy', ha='center', va='center', fontsize=18)
        figure.text(0.03, 0.5, 'Confidence', ha='center', va='center', rotation='vertical', fontsize=18)
        figure.savefig(plot_save_path, dpi=400)
#run_inference()
make_plots('bitrate')