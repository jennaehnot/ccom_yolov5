import val
import torch
import numpy as np

# val = model.val()
# path = '/home/jennaehnot/Desktop/yolov5/model_testing/redball_150cmH/labels'

# datapath = '/home/jennaehnot/Desktop/yolov5/model_testing/dataset.yaml'

# weightspath = '/home/jennaehnot/Desktop/yolov5/model_testing/model3_4_best.pt'
# results = val.run(datapath,weightspath, imgsz=1280)

yolopath = '/home/jennaehnot/Desktop/val_test/stock_yolov5/yolov5'
modelpath = "/home/jennaehnot/Desktop/val_test/stock_yolov5/yolov5/models/model3_4_best.pt"
model = torch.hub.load(yolopath, "custom", path=modelpath, source="local") 

imgs = ['yolov5/model_testing/redball_150cmH/images/val/rb150_1880.png','yolov5/model_testing/redball_150cmH/images/val/rb150_1881.png','yolov5/model_testing/redball_150cmH/images/val/rb150_1882.png']
lbls = ['yolov5/model_testing/redball_150cmH/labels/val/rb150_1880.txt','yolov5/model_testing/redball_150cmH/labels/val/rb150_1881.txt','yolov5/model_testing/redball_150cmH/labels/val/rb150_1882.txt']
detects = np.empty((0,6))
labels = np.empty((0,5))
for img in imgs:
    results = model(img, size = 1184)
    frame_name = results.files[0]
    xywh = np.array(results.pandas().xywh[0])
    [[xc, yc, w, h, conf, class_num, name]] = np.array(xywh)
    detects = np.append(detects, [[ xc, yc, w, h, conf, class_num]],axis=0)

for lbl in lbls:
    with open(lbl,'r') as file:
        for line in file:
            line = line.strip()
            line = line.split(' ')
            l = list(map(float,line))
            l[0] = int(l[0])
            labels = np.append(labels, [l], axis = 0)
iouv = np.linspace(0.5, 0.95,10)
correct =  val.process_batch(detects,labels,iouv)
print('wait')
