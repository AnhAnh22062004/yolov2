import tqdm 
import torch
from detect import parse_args, ANCHOR_BOXS

args = parse_args()

def output_tensor_to_boxes(boxes_tensor):
    cell_w, cell_h = args.w / args.gird , args.h / args.gird
    boxes = []
    probs = []
    
    for i in range(args.gird):
        for j in range(args.gird):
            for a in range(args.box):
                anchor_wh = torch.tensor(ANCHOR_BOXS[a])
                data = boxes_tensor[i, j, a]
                xy = torch.sigmoid(data[:2])
                