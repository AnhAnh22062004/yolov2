from matplotlib import pyplot as plt 
import torch
import cv2
from detect import parse_args

args = parse_args()

def plot_image(img, size=(7,7)):
    plt.figure(figsize=size)
    plt.imshow(img[:,:,::-1])
    plt.show()
    
# vẽ bounding box lên ảnh
def visualize_bbox(img, boxes, thickness=2, color= args.box_color, draw_center=True):
    img_copy = img.cpu().permute(1,2,0).numpy() if isinstance(img, torch.Tensor) else img.copy()
    for box in boxes:
        x,y,w,h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        img_copy = cv2.rectangle(
            img_copy,
            (x,y),(x+w, y+h),
            color, thickness)
        if draw_center:
            center = (x+w//2, y+h//2)
            img_copy = cv2.circle(img_copy, center=center, radius=3, color=(0,255,0), thickness=2)
    return img_copy

