import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tiny_yolo_model import TINYMODEL
from loss_function import custom_loss
from nms import nonmax_suppresion, output_tensor_to_boxes
from detect import parse_args
from visualize import plot_image, visualize_bbox 
import torch
from face_dataset import Face_data
from read_process_data_line import read_data
import albumentations as A
from albumentations.pytorch import ToTensorV2



def train_model(model, train_loader, val_loader, criterion, optimizer, device, args, num_epochs=1):
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Use tqdm for the training loop
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        
        for batch in train_loader_tqdm:
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Update tqdm description with the current loss
            train_loader_tqdm.set_postfix(loss=running_loss / (train_loader_tqdm.n + 1))

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
        
        # Validate the model
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_loader_tqdm = tqdm(val_loader, desc="Validation", leave=False)
            
            for images, targets in val_loader_tqdm:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                for i in range(images.size(0)):
                    boxes = output_tensor_to_boxes(outputs[i], args)
                    boxes = nonmax_suppresion(boxes)
                    img_with_boxes = visualize_bbox(images[i], boxes)
                    plot_image(img_with_boxes)
                    break  # Visualize only one image per epoch for simplicity

            print(f"Validation Loss: {val_loss / len(val_loader)}")

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


    args = parse_args()
    train_data = read_data(args.train_annotations)
    val_data = read_data(args.val_annotations)

    model = TINYMODEL(S=args.grid, BOX=args.box, CLS=args.cls)
    criterion = custom_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = Face_data(train_data, image_dir = args.train_dir, transforms=train_transform, is_train=True)
    val_dataset = Face_data(val_data, image_dir = args.val_dir, transforms=val_transform, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=val_dataset.collate_fn)


    train_model(model, train_loader, val_loader, criterion, optimizer, device, args, num_epochs=1)
