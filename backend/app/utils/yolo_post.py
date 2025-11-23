
import numpy as np
import cv2

def letterbox(im, new_size=640, color=(114,114,114)):
    h, w = im.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(h * scale), int(w * scale)
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_size - nh) // 2
    bottom = new_size - nh - top
    left = (new_size - nw) // 2
    right = new_size - nw - left
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=color)
    return im_padded, scale, left, top

def preprocess(img_bgr, img_size=640):
    img, scale, pad_x, pad_y = letterbox(img_bgr, img_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = img_rgb.astype(np.float32)/255.0
    x = np.transpose(x, (2,0,1))[None]
    meta = dict(scale=scale, pad_x=pad_x, pad_y=pad_y, orig_h=img_bgr.shape[0], orig_w=img_bgr.shape[1])
    return x, meta

def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:,0])
    y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2])
    y2 = np.minimum(box[3], boxes[:,3])
    inter = np.maximum(0, x2-x1)*np.maximum(0, y2-y1)
    area1 = (box[2]-box[0])*(box[3]-box[1])
    area2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    union = area1+area2-inter+1e-9
    return inter/union

def nms(boxes, scores, iou_thres=0.45):
    idxs = scores.argsort()[::-1]
    keep=[]
    while idxs.size>0:
        i=idxs[0]; keep.append(i)
        if idxs.size==1: break
        rest=idxs[1:]
        ious=compute_iou(boxes[i], boxes[rest])
        idxs=rest[ious<iou_thres]
    return keep

def postprocess(pred, meta, classes, conf_thres=0.25, iou_thres=0.45):
    pred = np.squeeze(pred, axis=0)
    boxes = pred[:, :4]
    obj = pred[:, 4:5]
    cls_scores = pred[:, 5:]
    cls_ids = np.argmax(cls_scores, axis=1)
    cls_conf = np.max(cls_scores, axis=1)
    scores = (obj.flatten()*cls_conf)

    mask=scores>=conf_thres
    boxes=boxes[mask]; scores=scores[mask]; cls_ids=cls_ids[mask]
    if len(boxes)==0: return []

    cx,cy,bw,bh=boxes.T
    x1=cx-bw/2; y1=cy-bh/2; x2=cx+bw/2; y2=cy+bh/2
    boxes_xyxy=np.stack([x1,y1,x2,y2], axis=1)

    final=[]
    for c in np.unique(cls_ids):
        idxs=np.where(cls_ids==c)[0]
        keep=nms(boxes_xyxy[idxs], scores[idxs], iou_thres)
        for k in keep:
            i=idxs[k]
            bx=boxes_xyxy[i].copy()
            bx[[0,2]]-=meta["pad_x"]; bx[[1,3]]-=meta["pad_y"]
            bx/=meta["scale"]
            bx[0]=max(0,min(meta["orig_w"],bx[0]))
            bx[2]=max(0,min(meta["orig_w"],bx[2]))
            bx[1]=max(0,min(meta["orig_h"],bx[1]))
            bx[3]=max(0,min(meta["orig_h"],bx[3]))
            cid=int(c)
            cname=classes[cid] if cid<len(classes) else str(cid)
            final.append({
                "class_id": cid,
                "class_name": cname,
                "confidence": float(scores[i]),
                "bbox_xyxy": [float(v) for v in bx]
            })
    return final
