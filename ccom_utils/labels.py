import os, torch, glob, json
from imgs import imgname2frame
from utils.metrics import bbox_iou

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
