import val
import matplotlib.pyplot as plt
import os, torch, cv2, shutil, json
import numpy as np
import pandas as pd
from utils.metrics import bbox_iou
import utils.general as gen

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

''' STEP ONE: RUN INFERENCE ON VIDEO'''

# load yolo model
yolo_dir = '/home/jennaehnot/Desktop/ccom_yolov5'
weights_path= 'ccom_yolov5/model_testing/model3_4_best.pt'
model = torch.hub.load('ccom_yolov5', 'custom', path = weights_path, source='local',force_reload=True)

# path to imgs, read filenames and sort them alphabetically
img_dir = '/home/jennaehnot/Desktop/ccom_yolov5/model_testing/redball_150cmH/images/val/'
img_filenames=os.listdir(img_dir)
img_filenames.sort()

# path to save inference dir
inf_dir = '/home/jennaehnot/Desktop/ccom_yolov5/model_testing/redball_150cmH/inference/'

# imgsz to run inference at
img_sz = [640, 960,1280, 1600, 1920]

for sz in img_sz: #for every img size we want to run inference at
    # make a new folder within the inference folder for each image size
    imgsz_dir = 'imgsz' + str(sz)
    img_save_dir = inf_dir + imgsz_dir
    output_stats = {}

    for file in img_filenames:
        # run inference
        img_path = img_dir + file
        results = model(img_path, size= sz)
        results.save(labels=True, save_dir=img_save_dir,exist_ok =True) #exist_ok will rewrite existing folders instead of creating a new one so beware !
        
        times = list(results.t)
        times.append(sum(times)) # times = [prep_t, infer_t, nms_t, total]
        detects = results.pandas().xywhn[0]
        detects = detects.values.tolist()
        _, _, img_w, img_h = results.s

        output_stats[file] = {
            'Time Stats': times,
            'Detections xywhn:': detects,
            'Img Dimensions': [img_w, img_h]
        }
    
    # save stats 
    save_file = img_save_dir + '/inference_results.json'
    with open(save_file,'w') as json_file:
        json.dump(output_stats,json_file)

        





    '''
# ### STEP TWO: COMPARE VALIDATION AND GROUND TRUTH LABELS
gt_lbl_dir = '/home/jennaehnot/Desktop/ccom_yolov5/model_testing/redball_150cmH/labels/val/'
pred_lbl_dir = lbl_save_dir +  'imgsz640/labels/' # make iterable later (+ name + '/labels') 
minconf = 0.6
miniou = 0.65
correct = [] 

files = os.listdir(gt_lbl_dir)
for filename in files:
    gtlabels= []
    predlabels = []
    if os.path.exists(pred_lbl_dir + filename): #if the model detected something in this frame
        
        with open(gt_lbl_dir + filename, 'r') as file: # read in gtlabels
            for line in file:
                 line = line.strip().split()
                 line = list(map(float,line))
                 line[0] = int(line[0])
                 gtlabels.append(list(line))

        with open(pred_lbl_dir + filename, 'r') as file: #read in vallabels
            for line in file:
                line = line.strip().split()
                line = list(map(float,line))
                if line[5] >= minconf: # parse low conf detects
                     line[0] = int(line[0])
                     predlabels.append(line)
        # do iou !
        for i in range(0,len(predlabels)):
            box1 = torch.Tensor(predlabels[i][1:5])
            for k in range(0,len(gtlabels)):
                box2 = torch.Tensor(gtlabels[k][1:])
                iou = bbox_iou(box1,box2)
                if float(iou)>= miniou:
                    c = box1.tolist()
                    c.append(gtlabels[k][0])
                    correct.append(c) # save val labels

        ### STEP THREE: SAVE THE RESULTS !
        # locate img file
        framenum = filename[:-4] # remove .txt
        imgpath = gt_lbl_dir.replace('labels','images') + framenum
        imgpath = find_imgtype(imgpath) # checks if img is jpg, jpeg, or png
        img_name = framenum + imgpath[-4:] # adds correct file type ending
        img_save_dir = lbl_save_dir.replace('labels','images') 

        if len(correct) >= 1: # if there's any correct inferences
            # draw boxes
            annted_img = draw_boxes(filename, imgpath, correct)

            #save img
            cv2.imwrite(img_save_dir + img_name, annted_img)#if dir does not exist, imwrite returns False    

            # save frame num, conf, class

        else: # there were inferences but they weren't correct
            # copy unannotated img to new dir 
            shutil.copyfil(imgpath,img_save_dir)

    else:
        pass

# yolopath = '/home/jennaehnot/Desktop/ccom_yolov5'
# modelpath = "/home/jennaehnot/Desktop/ccom_yolov5/model_testing/model3_4_best.pt"
# model = torch.hub.load(yolopath, "custom", path=modelpath, source="local") 

# imgs = ['ccom_yolov5/model_testing/redball_150cmH/images/val/rb150_1880.png','ccom_yolov5/model_testing/redball_150cmH/images/val/rb150_1881.png','ccom_yolov5/model_testing/redball_150cmH/images/val/rb150_1882.png']
# lbls = ['ccom_yolov5/model_testing/redball_150cmH/labels/val/rb150_1880.txt','ccom_yolov5/model_testing/redball_150cmH/labels/val/rb150_1881.txt','ccom_yolov5/model_testing/redball_150cmH/labels/val/rb150_1882.txt']
# detects = np.empty((0,6))
# labels = np.empty((0,5))
# imgsz = 1184
# for img in imgs:
#     results = model(img, size = imgsz)
#     results.save(labels=True, save_dir='results')
#     frame_name = results.files[0]
#     xywh = np.array(results.pandas().xywh[0])
#     [[xc, yc, w, h, conf, class_num, name]] = np.array(xywh)
#     detects = np.append(detects, [[ xc, yc, w, h, conf, class_num]],axis=0)

# for lbl in lbls:
#     with open(lbl,'r') as file:
#         for line in file:
#             line = line.strip()
#             line = line.split()
#             l = list(map(float,line))
#             l[0] = int(l[0])
#             labels = np.append(labels, [l], axis = 0)
# iouv = np.linspace(0.5, 0.95,10)

# # despite yolov5 documentation process_batch does Not want numpy arrays
# detects = torch.from_numpy(detects)
# labels = torch.from_numpy(labels)
# iouv = torch.from_numpy(iouv)

# correct =  val.process_batch(detects,labels,iouv)
# print('wait')
'''