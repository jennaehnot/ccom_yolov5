import json
import numpy as np
import pandas as pd


def unpack_inf_json(jsondata):
    data = jsondata.values
    frames = jsondata.index.to_numpy()
    times = data[:,1]
    dists = data[:,2]
    detects = data[:,3]
    confs = []
    for dets in detects:
        c = []
        for d in dets:
            cnf = d[4]
            confs.append(cnf) 
    bbox_bool = data[:,5]
    cls_bool = data[:,6]
    iou = data[:,8]
    eucs = data[:,9]

    df = pd.DataFrame({
        'Frame': frames,
        'Distance': dists,
        'IoU': iou,
        'Times': times,
        'Confidence': confs,
        'Bbox Match': bbox_bool,
        'Class Match': cls_bool,
        'Euc': eucs
    })
    return df



def restruct4plt(jdata):
    try: 
        # test if json file is from entropy calculations
        a = jdata['Bbox Area'].tolist()
    except KeyError:
        # if not 
        detects = jdata['Detections'].tolist()
        confs = []
        for dets in detects:
            c = []
            for d in dets:
                cnf = d[4]
                c.append(cnf)
            confs.append(c)

        df = pd.DataFrame({
            'Frame': jdata.index.to_list(),
            'Distance': jdata['Distance'].tolist(),
            'IoU': jdata['IoU'].tolist(),
            'Times': jdata['Time Stats'].tolist(),
            'Confidence': confs,
            'Bbox Match': jdata['Bbox Match'].tolist(),
            'Class Match': jdata['Class Match'].tolist(),
            'Euclidean Distance': jdata["Euclidean Distance"].tolist()
        })
        # handle multiple detections in each frame which generates multiple confs
        df = df.explode('Confidence')
        return df
    
    else:
        df = pd.DataFrame({
            'Frame': jdata.index.to_list(),
            'Distance': jdata['Distance'].tolist(),
            'IoU': jdata['IoU'].tolist(),
            'Times': jdata['Time Stats'].tolist(),
            'Confidence': jdata['Confidence'].tolist(),
            'Bbox Match': jdata['Bbox Match'].tolist(),
            'Class Match': jdata['Class Match'].tolist(),
            'Euclidean Distance': jdata["Euclidean Distance"].tolist(),
            'Entropy': jdata['Entropy'].tolist(),
            'Entropy per Pixel': jdata['Entropy per Pixel'].tolist(),
            'Bbox Area': jdata['Bbox Area'].tolist()
        })
        # handle multiple detections in each frame which generates multiple confs
        df = df.explode('Confidence')
        return df
