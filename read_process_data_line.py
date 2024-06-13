from visualize import plot_image, visualize_bbox
from detect import parse_args
from random import choice
import cv2
import numpy as np
import os

def get_xywh_from_textline(text):
    coor = text.split(" ")
    xywh = [int(coor[i]) for i in range(4)] if len(coor) >= 4 else None
    return xywh

def read_data(file_path, face_nb_max=0):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            current_line = lines[i].strip()
            if '.jpg' in current_line:
                try:
                    box_nb = int(lines[i + 1].strip())
                except ValueError:
                    print(f"Skipping invalid box number at line {i + 1}: {lines[i + 1].strip()}")
                    i += 1
                    continue

                image_data = {
                    'file_path': current_line,
                    'box_nb': box_nb,
                    'boxes': [],
                }

                face_nb = image_data['box_nb']
                if face_nb <= face_nb_max or face_nb_max == 0:
                    for j in range(face_nb):
                        rect = get_xywh_from_textline(lines[i + 2 + j].strip())
                        if rect is not None:
                            image_data['boxes'].append(rect)
                    if len(image_data['boxes']) > 0:
                        data.append(image_data)
                i += 1 + face_nb + 1
            else:
                i += 1
    return data

if __name__ == '__main__':
    args = parse_args()
    train_data = read_data(args.train_annotations)
    val_data = read_data(args.val_annotations)
    image_data = choice(val_data)
    image_path = os.path.join(args.train_dir, image_data['file_path'])
    image = cv2.imread(image_path).astype(np.float32) / 255.0
    if image is None:
        print(f"Image not found at path: {image_path}")
    else:
        print(f"Image found at path: {image_path}")
        visualize_image = visualize_bbox(image, boxes=image_data['boxes'], thickness=3, color=args.box_color)
        plot_image(visualize_image)
