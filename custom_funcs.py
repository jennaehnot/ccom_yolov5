import os, cv2
import torch, csv, glob
import numpy as np
import json
import utils.general as gen
from utils.metrics import bbox_iou

def find_imgtype(imgpath):
    if os.path.exists(imgpath + '.jpg'):
        imgpath = imgpath + '.jpg'
    elif os.path.exists(imgpath + '.jpeg'):
        imgpath = imgpath + '.jpeg'
    elif os.path.exists(imgpath + '.png'):
        imgpath = imgpath + '.png'
    else:
        print('Img is not a jpg, jpeg, png')
        print(f'Path evaluated: {imgpath}')
        imgpath = ''
    return(imgpath)

def draw_boxes(filename, imgpath, correct):

    try:
        img = cv2.imread(imgpath)
        if img is None:
            raise FileNotFoundError(f" Image = None ! Make that make sense")
    except FileNotFoundError as e:
        print(f"{imgpath}")
        return None
    
    img_h, img_w, _ = img.shape 
    
    for i in range(0,len(correct)):
        detect = correct[i][:]
        class_id = detect[-1]
        detect = gen.xywhn2xyxy(np.array(detect[:-1]), img_w, img_h)
        rounded_detect = [round(coord)for coord in detect] #cv2 requires int for rectangle coords
        [x1, y1, x2, y2] = rounded_detect
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f'C: {class_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ( 0, 0, 255), 2)
        cv2.putText(img, f"{filename}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #cv2 is by default in bgr
    return(img)

def load_labels(filepath):
    if os.path.exists(filepath):
        labels = []
        with open(filepath, 'r') as file:
            for line in file:
                line = line.strip().split()
                line = list(map(float,line))
                line[0] = int(line[0])
                labels.append(list(line))
        return(labels)
    else:
        print(f"There was an issue opening {filepath}")
        return([])

def compare_labels(gt_box, det_box, min_iou):
    # box1, box2 are list, min_iou is float
    gt_box = torch.Tensor(gt_box)
    det_box = torch.Tensor(det_box)
    iou = bbox_iou(gt_box,det_box)
    if float(iou) >= min_iou:
        return(True)
    else:
        return(False)

def imgname2frame(img_name):
    # for file name convention: frame_%05d.png
    n = img_name[-9:-4] # access 5 ints before .png

    return(n)

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

def create_gt_json(gtlbl_dir, outf_name):
    try:
        fs = os.path.join(gtlbl_dir, 'frame*.txt')
        files = glob.glob(fs, recursive=False)
        files.sort()
        output = {}
        for f in files:
            frame_num = int(imgname2frame(f))
            lbl = load_labels(f)
            output[frame_num] = lbl
        
        with open(outf_name, 'w') as json_file:
            json.dump(output,json_file)
    except Exception as e:
        print(e)

    return


class Inference:
    def __init__(self, pred_data):
        self.times = pred_data['Time Stats']
        self.detects = [Detect(i) for i in pred_data['Detections']]
        self.imgsz = pred_data['Img Dimensions']
    
class Detect:
    def __init__(self,detects):
        if detects:
            self.xywhn = detects[0:4]
            self.conf = detects[4]
            self.class_id = detects[5]
            self.class_name = detects[6]
