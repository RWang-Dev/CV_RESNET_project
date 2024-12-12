from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np
from PIL import Image
import torch, torchvision
import torchvision.transforms as T

class WasteObjectDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, class_names, removed, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_names = class_names
        self.transforms = transforms
        self.image_paths = glob.glob(os.path.join(images_dir, '*.*'))
        
        self.kept_classes = [cls for idx, cls in enumerate(class_names) if idx+1 not in removed]
        
        self.class_mapping = {}
        current_class_id = 1
        for original_id in range(1, len(class_names) + 1):
            if original_id not in removed:
                self.class_mapping[original_id] = current_class_id
                current_class_id += 1
        
        self.num_classes = len(self.kept_classes)
        self.removed = removed
        
        print("Kept Classes:", self.kept_classes)
        print("Class Mapping:", self.class_mapping)
        print("Number of Classes:", self.num_classes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        width, height = img.size

        filename = os.path.basename(img_path)
        label_path = os.path.join(self.labels_dir, os.path.splitext(filename)[0] + '.txt')

        boxes = []
        labels = []
        default_box_added = False

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    original_class_id = int(parts[0]) + 1
                
                    if original_class_id in self.removed:
                        continue
                    

                    if original_class_id in self.class_mapping:
                        class_id = self.class_mapping[original_class_id]
                    else:
                        continue
                    
                    # Convert YOLO format to Pascal VOC
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    bbox_width = float(parts[3])
                    bbox_height = float(parts[4])

                    x_min = (x_center - bbox_width / 2) * width
                    y_min = (y_center - bbox_height / 2) * height
                    x_max = (x_center + bbox_width / 2) * width
                    y_max = (y_center + bbox_height / 2) * height
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)
                    default_box_added = True


        if not default_box_added:

            default_class = min(self.class_mapping.values())
            x_min = width * 0.25
            y_min = height * 0.25
            x_max = width * 0.75
            y_max = height * 0.75
            
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(default_class)


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
        }


        if self.transforms:
            transformed = self.transforms(image=np.array(img), bboxes=boxes.tolist(), labels=labels.tolist())
            img = transformed['image']
            target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)
        else:
            img = T.ToTensor()(img)

        return img, target