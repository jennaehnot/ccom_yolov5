import os, cv2
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

def draw_boxes_from_pth(filename, imgpath, bboxes):

    try:
        img = cv2.imread(imgpath)
        if img is None:
            raise FileNotFoundError(f" Image = None ! Make that make sense")
    except FileNotFoundError as e:
        print(f"{imgpath}")
        return None
    
    img_h, img_w, _ = img.shape 
    
    for i in range(0,len(bboxes)):
        detect = bboxes[i][:]
        class_id = detect[-1]
        detect = gen.xywhn2xyxy(np.array(detect[:-1]), img_w, img_h)
        rounded_detect = [round(coord)for coord in detect] #cv2 requires int for rectangle coords
        [x1, y1, x2, y2] = rounded_detect
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f'C: {class_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ( 0, 0, 255), 2)
        cv2.putText(img, f"{filename}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #cv2 is by default in bgr
    return(img)

def draw_boxes(img,box,conf, class_id, bool=False):
    if img is None:
        print(" Image = None ! Make that make sense")
        return None
    else:
        img_h, img_w, _ = img.shape
        x1, y1, x2, y2 = [round(n) for n in box]
        if bool == True: #in green if true positive
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"C: {conf} ID: {class_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else: # in red if false positive
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if x2 > img_w - 50: # to avoid putting text outside of image
                cv2.putText(img, f"C: {conf:.2f} ID: {class_id}", (x1 - 150, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif x1 < 50:
                cv2.putText(img, f"C: {conf:.2f} ID: {class_id}", (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif y1 < 50:
                cv2.putText(img, f"C: {conf:.2f} ID: {class_id}", (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif y2 > img_h - 50:
                cv2.putText(img, f"C: {conf:.2f} ID: {class_id}", (x1 + 10, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(img, f"C: {conf:.2f} ID: {class_id}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return(img)

def imgname2frame(img_name):
    # for file name convention: frame_%05d.png
    n = img_name[-9:-4] # access 5 ints before .png

    return(n)