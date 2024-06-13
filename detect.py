import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOV2")
    parser.add_argument("--data_dir", type=str, default='../../../Dataset/WIDER', help='Please enter path')
    parser.add_argument("--train_dir", type=str, default='../../../Dataset/WIDER/WIDER_train', help='Please enter path')
    parser.add_argument("--val_dir", type=str, default='../../../Dataset/WIDER/WIDER_val', help='Please enter path')
    parser.add_argument("--test_dir", type=str, default='../../../Dataset/WIDER/WIDER_test', help='Please enter path')
    parser.add_argument("--train_annotations", default='../../../Dataset/WIDER/wider_face_train_bbx_gt.txt',help='Please enter path')
    parser.add_argument("--val_annotations", default ='../../../Dataset/WIDER/wider_face_val_bbx_gt.txt', help = 'Please eneter path')
    parser.add_argument("--test_annotations", default ='../../../Dataset/WIDER/wider_face_test_filelist.txt', help='Please enter path')
    parser.add_argument('--box_color', type=tuple, default=(0, 0, 255), help='Please enter color')
    parser.add_argument('--text_color', type=tuple, default=(255, 255, 255), help='Please enter color')
    parser.add_argument('--grid', type=int, default=13, help='Please enter number')
    parser.add_argument('--box', type=int, default=5, help='Number of boxes per cell') # numbers of bounding box predict 
    parser.add_argument('--cls', type=int, default=2, help='Number of classes')
    parser.add_argument('--h', type=int, default=416, help='Height of the input image')
    parser.add_argument('--w', type=int, default=416, help='Width of the input image')
    parser.add_argument('--output_thresh', type=float, default=0.7, help='Threshold for output')
    parser.add_argument("--batch_size", type= int , default= 8, help= "Please enter the number")

    args = parser.parse_args()
    return args

ANCHOR_BOXS = [
    [1.08, 1.19],
    [3.42, 4.41],
    [6.63, 11.38],
    [9.42, 5.11],
    [16.62, 10.52]
]
