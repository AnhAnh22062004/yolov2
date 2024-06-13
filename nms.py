import tqdm 
import torch
from detect import parse_args, ANCHOR_BOXS


def output_tensor_to_boxes(boxes_tensor, args):
    cell_w, cell_h = args.w / args.grid , args.h / args.grid 
    boxes = []
    probs = []
    
    for i in range(args.grid):
        for j in range(args.grid):
            for a in range(args.box):
                anchor_wh = torch.tensor(ANCHOR_BOXS[a])
                data = boxes_tensor[i, j, a]
                xy = torch.sigmoid(data[:2])
                wh = torch.exp(data[2:4]) * anchor_wh
                obj_prob = torch.sigmoid(data[4:5])
                cls_prob = torch.sigmoid(data[5:], dim = 1)
                combine_prob = obj_prob* max(cls_prob)
                
                
                if combine_prob > args.output_threah:
                    x_center , y_center , w , h = xy[0], xy[1], wh[0], wh[1]
                    x, y = x_center + j - w/2 , y_center + i - h/2
                    x, y , w, h = x*cell_w, y*cell_h, w* cell_w, h*cell_h
                    box = [x, y, w, h, combine_prob]
                    boxes.append(box)
    return boxes

def over_lap(interval_1, interval_2):
    x1, x2 = interval_1
    x3, x4 = interval_2
    if x3 < x1:
        if x4 < x1: 
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x1 < x3:
            return 0 
        else: 
            return min(x2, x4) - x3
        

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1[0], box1[1], box1[2], box1[3]
    x2, y2, w2, h2 = box2[0], box2[1], box2[2], box2[3]
    
    ## if box2 is inside box1 
    if x1 < x2  and y1 < y2  and w1 > w2  and h1 > h2:
        return 1 
    area1 , area2 = w1 * h1 , w2 * h2 
    intersect_w = over_lap((x1, x1 + w1), (x2, x2 + w2))
    intersect_h = over_lap((y1, y1 + h1), (y2, y2 + h2))
    intersect_area = intersect_w * intersect_h
    iou = intersect_area / (area1 + area2 - intersect_area)
    return iou 

def nonmax_suppresion(boxes, OUT_THRESH = 0.4):
    boxes = sorted(boxes, key = lambda x : x[4], reserver = True) #confident score
    for i, current_box in enumerate(boxes):
        if current_box[4] <= 0: 
            continue
        for j in range(i+1, len(boxes)):
            iou = compute_iou(current_box, boxes[j])
            if iou > OUT_THRESH:
                boxes[j][4] = 0
                
    boxes = [box for box in boxes if box[4] > 0]
    return boxes     