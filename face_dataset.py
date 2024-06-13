from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch 
import cv2
import numpy as np 
from detect import parse_args
from random import randint
import albumentations as A
from albumentations.pytorch import ToTensorV2
import read_process_data_line

args = parse_args()

class Face_data(Dataset):
    def __init__(self, data, image_dir, transforms=None, is_train=True):
        self.data = data 
        self.image_dir = image_dir
        self.transforms = transforms
        self.is_train = is_train
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_data = self.data[index]
        image_filename = f"{self.image_dir}/{image_data['file_path']}"
        boxes = image_data['boxes']
        box_nb = image_data['box_nb']
        labels = torch.zeros((box_nb, 2), dtype=torch.int64)
        labels[:, 0] = 1 
        image = cv2.imread(image_filename).astype(np.float32) / 255.0
        
        
        try:
            if self.transforms:
                sample = self.transforms(**{
                    "image": image,
                    'bboxes': boxes,
                    'labels': labels
                })
                image = sample['image']
                boxes = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(0, 1)
        except Exception as e:
            print(f"Error in transforming: {e}")
            return self.__getitem__(randint(0, len(self.data) - 1))
        
        target_tensor = self.boxes_to_tensor(boxes.type(torch.float32), labels)
        return image, target_tensor

    
    def boxes_to_tensor(self, boxes, labels):
        boxes_tensor = torch.zeros((args.grid, args.grid, args.box, 5 + args.cls))
        cell_w, cell_h = args.w / args.grid, args.h / args.grid
        for i, box in enumerate(boxes):
            if len(box) != 4:
                continue
            x, y, w, h = box
            x, y, w, h = x / cell_w, y / cell_h, w / cell_w, h / cell_h
            center_x, center_y = x + w / 2, y + h / 2
            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))

            if grid_x < args.grid and grid_y < args.grid:
                boxes_tensor[grid_y, grid_x, :, 0:4] = torch.tensor(args.box * [[center_x - grid_x, center_y - grid_y, w, h]])
                boxes_tensor[grid_y, grid_x, :, 4] = torch.tensor(args.box * [1.])
                boxes_tensor[grid_y, grid_x, :, 5:] = torch.tensor(args.box * [labels[i].numpy()])
    
        return boxes_tensor

    def collate_fn(self, batch):
        images = []
        targets = []
        for image, target in batch:
            images.append(image)
            targets.append(target)
        return torch.stack(images), torch.stack(targets)

    
    def target_tensor_to_boxes(self, boxes_tensor):
        cell_w, cell_h = args.w / args.gird, args.h / args.gird
        boxes = []
        for i in range(args.gird):
            for j in range(args.gird):
                for a in range(args.box):
                    data = boxes_tensor[i, j, a]
                    x_center, y_center, w, h, obj_prob, cls_prob = data[0], data[1], data[2], data[3], data[4], data[5:]
                    prob = obj_prob * max(cls_prob)
                    if prob > args.output_thresh:
                        x, y = x_center + j - w / 2, y_center + i - h / 2    
                        x, y, w, h = x * cell_w, y * cell_h, w * cell_w, h * cell_h
                        box = [x, y, w, h]
                        boxes.append(box)
                        
        return boxes

if __name__ == '__main__': 
    train_transform = A.Compose([
        A.Resize(height=416, width=416),
        A.RandomSizedCrop(min_max_height=(350, 416), height=416, width=416, p=0.4),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(p=1.0)
    ], 
    bbox_params={
        "format": "coco",
        "label_fields": ['labels']
    })

    val_transform = A.Compose([
        A.Resize(height=416, width=416),
        ToTensorV2(p=1.0)
    ], 
    bbox_params={
        "format": "coco",
        "label_fields": ['labels']
    })
 
