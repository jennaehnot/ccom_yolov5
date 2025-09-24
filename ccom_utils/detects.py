import numpy as np
import torch
import utils.general as gen
from val import process_batch

def split_detect(det):
    # d_xwyhn: np.array (N, 4) [x_center, y_center, w, h, conf, class_id, class name]

    # return 
    det_xywhn = np.array(np.array(det)[:,:4],dtype=float)
    det_confid = np.array(det)[:,4:6] # np.arr(N X [ conf, class_id])
    return det_xywhn, det_confid

def split_gt(gt):
    # gt: np.array (M, 5) [class_id, x_center, y_center, w, h]
    gt_xywhn = np.array(gt, dtype=float)[:,1:] 
    gt_c = np.array(gt, dtype=float)[:,0:1].astype(float) # np.arr( M x [class])
    return gt_xywhn, gt_c

def euclidean_distance(det, gt, img_w, img_h):
    # both inputs are np.arrays
    # calc euclidean distance btwn gt center and detection center
    det_xywhn, _ = split_detect(det)
    gt_xywhn, _ = split_gt(gt)
    pix_d = det_xywhn[:,0:2] * np.array([img_w,img_h])
    pix_gt = gt_xywhn[0,0:2] * np.array([img_w,img_h])
    euc = np.linalg.norm(pix_d - pix_gt)
    euc_norm = np.linalg.norm(det_xywhn[:,0:2] - gt_xywhn[0,0:2])
    return euc, euc_norm

def compare_dets_gts(det, gt, img_w, img_h, iouv):

    # split inputs
    det_xywhn, det_confc = split_detect(det)
    gt_xywhn, gt_c = split_gt(gt)
    # convert to xyxy for iou calculation
    det_xyxy = gen.xywhn2xyxy(det_xywhn.astype(float), w=img_w, h=img_h) 
    gt_xyxy = gen.xywhn2xyxy(gt_xywhn.astype(float), w=img_w, h=img_h)

    # convert detects to np array (N,6) [x1, y1, x2, y2, conf, class]
    # convert gt to np array (m, 6) [class, x1, y1, x2, y2]
    d = np.hstack((det_xyxy, det_confc))
    g = np.hstack((gt_c, gt_xyxy))
    d = d.astype(float)
    g = g.astype(float) 
    correct_bb, correct_class, iou = process_batch(torch.Tensor(d), torch.Tensor(g), iouv)
    
    return correct_bb.tolist(), correct_class, iou.tolist()

  