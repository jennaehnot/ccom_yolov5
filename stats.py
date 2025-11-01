import os, glob
import numpy as np
import pandas as pd
import collections
import cv2
import ccom_utils.imgs as imgs
import matplotlib.pyplot as plt
from ccom_utils.labels import load_labels
from PIL import Image

'''
- pull image size
    -plot height vs width with aspect ratio lines
    -make histogram of sizes
    -make histogram of aspect ratios

- pull bbox areas by class
    - make histogram of bbox areas by class
    - make histogram of bbox aspect ratios

'''

pth = '/home/jennaehnot/Desktop/dataset_v4/labels' 

results = {}
nc = 2
# label eval
for subset in ['train']:#, 'val']:
    sub_pth = os.path.join(pth, subset)
    img_sub_pth = sub_pth.replace('labels','images')

    # init variables
    class_instances = collections.defaultdict(int)
    class_areas = collections.defaultdict(list)
    class_areasnorm = collections.defaultdict(list)
    class_ratios = collections.defaultdict(list)
    class_images = collections.defaultdict(set)
    filenames = []
    img_ratios = []
    img_sizes = []
    total_f = 0


    for file in os.listdir(sub_pth):        
        # try to load label
        lbl = load_labels(os.path.join(sub_pth, file))
        if lbl is not []:
            total_f += 1
            # find image path
            img = file[:-4]
            filenames.append(img)
            imgpath = imgs.find_imgtype(os.path.join(img_sub_pth, img))
            if imgpath == '':
                print(f"Image not found for {file}")
                print("Is it something other than jpg, jpeg, png?")
                continue

            # load image to get size
            try:
                im = Image.open(imgpath)
                im.verify()  # Verify that it is, in fact an image
                img_w, img_h = im.size
                im.close()
                
            except Exception as e:
                print(f"pth: {imgpath}")
                print(e)

            else:
                img_sizes.append((img_w,img_h))
                img_ratios.append(img_w/img_h)
                #process labels
                for l in lbl:
                    # track instances per class
                    cls_id = l[0]
                    class_instances[cls_id] += 1
                    # track area
                    bbox_w = l[3]*img_w
                    bbox_h = l[4]*img_h
                    area = int(bbox_w) * int(bbox_h)
                    class_areas[cls_id].append(area)
                    class_areasnorm[cls_id].append(area / (img_w * img_h))
                    # track aspect ratio
                    ratio = bbox_w / bbox_h if bbox_h != 0 else 0
                    class_ratios[cls_id].append(ratio)
                    # track images per class 
                    class_images[cls_id].add(file)
        total_f += 1
                    
    #print(f"Subset: {subset}")

# plot of class 0 areas
fig, axs = plt.subplots(1,2)

axs[0].hist([class_areas[0], class_areas[1]], bins=[0,1024,9216,82944, max(max(a) for a in class_areas.values())], edgecolor='black')
axs[0].set_title('Class 0 BBox Areas')
# axs[0].set_xticks([0,1024,9216,82944, max(max(a) for a in class_areas.values())])
# axs[0].set_xticklabels(['0','32x32','96x96','288x288','max'])
axs[1].hist([class_areasnorm[0], class_areasnorm[1]], bins=[0, 0.003, 0.03,0.27,1], edgecolor='black')
# axs[1].set_xticks([0, 0.003, 0.03,0.27])
# axs[1].set_xticklabels(['< 0.3%', '0.3 - 3%','3-27%','27 - 100%'])
# axs[1].set_title('Class 0 Normalized BBox Areas')
# fig.legend(['Class 0', 'Class 1','Class 0', 'Class 1'], loc='upper right')
# bins = [0, 32*32, 96*96, max(class_areas[0])]
# plt.hist(class_areasnorm[0], bins=10, edgecolor='black')
# plt.title('Class 0 Bounding Box Areas')
# plt.xlabel('Area (pixels^2)')
# plt.ylabel('Frequency')
#plt.xticks(bins)

plt.show()

           
#img_name = file.replace('.txt','.jpg')
#img_path = os.path.join('/home/jennaehnot/Desktop/dataset_v4/images', subset, img_name)
# if os.path.exists(img_path):
#     img = cv2.imread(img_path)
#     h, w = img.shape[:2]
#     for box in lbl:
#         cls_id = box[0]
#         class_instances[cls_id] += 1
#         toal_inst += 1
#         # box is [class_id, x_center, y_center, w, h]
#         bw = box[3] * w
#         bh = box[4] * h
#         area = bw * bh
#         ratio = bw / bh if bh != 0 else 0
#         class_areas[cls_id].append(area)
#         class_ratios[cls_id].append(ratio)
# else:
#     print(f"Image not found: {img_path}")



