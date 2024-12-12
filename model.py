import os
import glob
import numpy as np
from PIL import Image

import torch, torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
import torchvision.transforms as T
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import Subset
import random
from tqdm import tqdm
from torchvision.models.detection.rpn import AnchorGenerator
import random
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import CyclicLR, LambdaLR, SequentialLR

# custom modules
from sampler import SmallObjectSampler
from waste_dataset import WasteObjectDetectionDataset


from torchvision.ops import box_iou

def convert_bbox_from_yolo(boxes, width, height):
    return [
        [
            (x_center - bbox_width / 2) * width,  
            (y_center - bbox_height / 2) * height,  
            (x_center + bbox_width / 2) * width,   
            (y_center + bbox_height / 2) * height  
        ]
        for x_center, y_center, bbox_width, bbox_height in boxes
    ]
        
# def get_transform(train):
#     transforms = []
#     transforms.append(T.ToTensor())
#     # if train:
#     #     transforms.append(T.RandomHorizontalFlip(0.5))
#     return T.Compose(transforms)        # Test with no transform for now
#     return T.Compose(transforms)

def get_transform(train):
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                # A.Affine(
                #     scale=(0.8, 1.2),  
                #     rotate=(-30, 30), 
                #     shear=(-10, 10),  
                #     # translate_percent=(0.1, 0.1),  
                #     p=0.5  
                # ),
                # A.Resize(height=416, width=416),  
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
                ToTensorV2(),  
            ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
        )

# This function replaces the default box predictor of the faster rcnn which detects 91 classes down to the 18 classes we are detecting
# def get_object_detection_model(num_classes):
#     print("getting model with random weights")
#     # Load pre-trained model
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
#     # Get the number of input features for the classifier
#     in_features = model.roi_heads.box_predictor.cls_score.in_features

#     # anchor_generator = AnchorGenerator(
#     #     sizes=((16,), (32,), (64,), (128,), (256,)),  # One size for each feature map
#     #     aspect_ratios=((0.5, 1.0, 2.0),) * 5          # Same aspect ratios for all feature maps
#     # )
#     # model.rpn.anchor_generator = anchor_generator
    
#     # Replace the pre-trained head with a new one
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
#     return model

def get_object_detection_model(num_classes):
    print("Initializing Faster R-CNN model...")
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    print("Model initialized with ResNet-50 backbone.")
    return model


def visualize_predictions(model, test_dataset, device, class_names, num_samples=2):
    model.eval()
    
    indices = random.sample(range(len(test_dataset)), num_samples)
    # indices[0] = 65
    # indices = [3]
    
    denormalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])

    for idx in indices:

        img, target = test_dataset[idx]
        
        img_tensor = img.to(device).unsqueeze(0)

        img_denorm = denormalize(img.cpu()).permute(1, 2, 0).numpy()
        img_denorm = (img_denorm * 255).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_denorm)

        with torch.no_grad():
            predictions = model(img_tensor)

        predictions = predictions[0]
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()

        true_boxes = target['boxes'].cpu().numpy()
        true_labels = target['labels'].cpu().numpy()

        
        target_img = img_pil.copy()
        
        draw = ImageDraw.Draw(img_pil)
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5:
                x_min, y_min, x_max, y_max = box
                class_name = class_names[label-1]
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
                draw.text((x_min, y_min), f"{class_name} {score:.2f}", fill="red")

        target_draw = ImageDraw.Draw(target_img)
        for box, label in zip(true_boxes, true_labels):
            x_min, y_min, x_max, y_max = box
            class_name = class_names[label-1]
            target_draw.rectangle([x_min, y_min, x_max, y_max], outline="cyan", width=3)
            target_draw.text((x_min, y_min), f"{class_name}", fill="cyan")
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img_pil)
        plt.axis("off")
        plt.title(f"Predictions for Image Index {idx}")
        plt.show()

        plt.figure(figsize=(8, 8))
        plt.imshow(target_img)
        plt.axis("off")
        plt.title(f"Target for Image Index {idx}")
        plt.show()


def get_precision_at_confidence(predictions, targets, class_id, confidence_thresholds):
    precisions = []
    for conf_thresh in confidence_thresholds:
        tp = 0
        fp = 0
        fn = 0
        for pred, target in zip(predictions, targets):
            mask = (pred['scores'] >= conf_thresh) & (pred['labels'] == class_id)
            pred_boxes = pred['boxes'][mask]
            
            gt_mask = target['labels'] == class_id
            gt_boxes = target['boxes'][gt_mask]
            
            matched_gt = set()
            for pb in pred_boxes:
                if gt_boxes.numel() == 0:
                    fp += 1
                    continue
                ious = box_iou(pb.unsqueeze(0), gt_boxes)
                iou_max, idx = ious.max(1)
                if iou_max >= 0.5 and idx.item() not in matched_gt:
                    tp += 1
                    matched_gt.add(idx.item())
                else:
                    fp += 1
            fn += len(gt_boxes) - len(matched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precisions.append(precision)
    return precisions

def get_total_precision_at_confidence(predictions, targets, confidence_thresholds):
    precisions = []
    for conf_thresh in confidence_thresholds:
        tp = 0
        fp = 0
        for pred, target in zip(predictions, targets):
            mask = pred['scores'] >= conf_thresh
            pred_boxes = pred['boxes'][mask]
            pred_labels = pred['labels'][mask]
            
            gt_boxes = target['boxes']
            gt_labels = target['labels']
            
            matched_gt = set()
            for pb, pl in zip(pred_boxes, pred_labels):
                gt_mask = (gt_labels == pl)
                relevant_gt_boxes = gt_boxes[gt_mask]
                if relevant_gt_boxes.numel() == 0:
                    fp += 1
                    continue
                ious = box_iou(pb.unsqueeze(0), relevant_gt_boxes)
                iou_max, idx = ious.max(1)
                if iou_max >= 0.5 and idx.item() not in matched_gt:
                    tp += 1
                    matched_gt.add(idx.item())
                else:
                    fp += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precisions.append(precision)
    return precisions

def custom_small_object_loss(model_output, targets, small_threshold=500, small_weight=2.0):
    loss_dict = model_output
    classification_loss = loss_dict['loss_classifier']
    localization_loss = loss_dict['loss_box_reg']
    objectness_loss = loss_dict.get('loss_objectness', torch.tensor(0.0))
    rpn_loss = loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0))
    
    small_object_mask = []
    for t in targets:
        areas = (t['boxes'][:, 2] - t['boxes'][:, 0]) * (t['boxes'][:, 3] - t['boxes'][:, 1])
        mask = (areas < small_threshold)
        small_object_mask.append(mask)
    
    any_small = any(m.any() for m in small_object_mask)
    if any_small:
        classification_loss = classification_loss * small_weight
        localization_loss = localization_loss * small_weight

    total_loss = classification_loss + localization_loss + objectness_loss + rpn_loss
    return total_loss

def warmup_scheduler(epoch):
    if epoch < 2:
        return epoch / 2
    return 1


def main():
    print("main function test 1")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # small_items = ['Cigarettes', 'Bottle cap', 'Broken glass', 'Pop tab']
    small_items = [3,4,7,15]
    # small_items = [1,3,6, 14]
    
    class_names = ['Aluminium foil', 'Bottle', 'Bottle cap', 'Broken glass', 'Can', 'Carton', 'Cigarette',
                'Cup', 'Lid', 'Other litter', 'Other plastic', 'Paper', 'Plastic bag - wrapper',
                'Plastic container', 'Pop tab', 'Straw', 'Styrofoam piece', 'Unlabeled litter']
    # class_names = ['Aluminium foil', 'Bottle', 'Can', 'Carton',
    #             'Cup', 'Lid', 'Other litter', 'Other plastic', 'Paper', 'Plastic bag - wrapper',
    #             'Plastic container',  'Straw', 'Styrofoam piece', 'Unlabeled litter']
    
    # remapping = {
    #     class_id: new_id
    #     for new_id, class_id in enumerate(sorted(set(range(len(class_names))) - set(small_items)))
    # }
    num_classes = len(class_names) + 1  # Include background class

    train_dataset_full = WasteObjectDetectionDataset(
        images_dir='data/train/images',
        labels_dir='data/train/labels',
        class_names=class_names,
        removed=small_items,
        transforms=get_transform(train=True)
        
    )


    subset_fraction = 1

    num_samples = int(len(train_dataset_full) * subset_fraction)

    indices = list(range(len(train_dataset_full)))
    random.shuffle(indices)

    subset_indices = indices[:num_samples]

    train_dataset = Subset(train_dataset_full, subset_indices)
    

    valid_dataset = WasteObjectDetectionDataset(
        images_dir='data/valid/images',
        labels_dir='data/valid/labels',
        class_names=class_names,
        removed=small_items,
        transforms=get_transform(train=False)
        
    )

    test_dataset = WasteObjectDetectionDataset(
        images_dir='data/test/images',
        labels_dir='data/test/labels',
        class_names=class_names,
        removed=small_items,
        transforms=get_transform(train=False)
        
    )

    batch_size = 4
    # train_sampler = SmallObjectSampler(train_dataset, batch_size=batch_size, small_fraction=0.5)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        # sampler=train_sampler,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # model_path = "waste_detection_100_24epoch_4batch.pth"
    model_path = "waste_detection_100_24epoch_4batch.pth"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = torch.load(model_path)
        model.to(device)
        print("Model loaded successfully.")
    else:
        print("num classes: ", num_classes)
        model = get_object_detection_model(num_classes)
        model.to(device)
        
        params = [p for p in model.parameters() if p.requires_grad]
        

        # optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0002)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, 
        #     T_0=10,  
        #     T_mult=2,
        #     eta_min=1e-5
        # )
        optimizer = torch.optim.SGD(
            params, 
            lr=0.005,      
            momentum=0.9, 
            weight_decay=0.0005
        )


        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones = [14, 18], gamma=0.1
        # )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=6, 
            T_mult=2,
            eta_min=1e-4  
        )

        metric = MeanAveragePrecision(class_metrics=True)
        best_map = 0.0
        num_epochs = 24

        print("Starting training...")
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            total_classification_loss = 0
            total_localization_loss = 0
            num_batches = len(train_loader)
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for images, targets in pbar:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                classification_loss = loss_dict['loss_classifier'].item()
                localization_loss = loss_dict['loss_box_reg'].item()
                # losses = custom_small_object_loss(loss_dict, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                total_loss += losses.item()
                total_classification_loss += classification_loss
                total_localization_loss += localization_loss
                
                pbar.set_postfix({
                    'train_loss': losses.item(),
                    'cls_loss': classification_loss,
                    'loc_loss': localization_loss
                })
            
            avg_train_loss = total_loss / num_batches
            avg_cls_loss = total_classification_loss / num_batches
            avg_loc_loss = total_localization_loss / num_batches
            

            model.eval()
            metric.reset()

            with torch.no_grad():
                for images, targets in valid_loader:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    predictions = model(images)
                    
                    metric.update(predictions, targets)
            
            metrics = metric.compute()
            map_score = metrics['map'].item()
            precision = metrics['map_50'].item()
            mar_10 = metrics['mar_10'].item() 

            lr_scheduler.step()
            
            if map_score > best_map:
                best_map = map_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'map_score': map_score,
                }, 'best_model_checkpoint.pth')
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Classification Loss: {avg_cls_loss:.4f}")
            print(f"Localization Loss: {avg_loc_loss:.4f}")
            print(f"Validation mAP: {map_score:.4f}")
            print(f"Validation Precision (IoU 0.5): {precision:.4f}")
            print(f"Recall @10: {mar_10:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)

        print(f"\nTraining completed! Best mAP: {best_map:.4f}")
        

        torch.save(model, "waste_detection_model.pth")

    print("Computing precision-confidence curves...")

    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for images, targets in tqdm(valid_loader, desc='Collecting predictions'):
            images = list(image.to(device) for image in images)
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
            
            
            predictions = model(images)
            predictions = [{k: v.cpu() for k, v in p.items()} for p in predictions]
                   
            all_predictions.extend(predictions)
            all_targets.extend(targets)

    
    confidence_thresholds = np.linspace(0.0, 1.0, 50)

    
    plt.figure(figsize=(12, 8))
    for class_id in range(1, num_classes):
        class_name = class_names[class_id - 1]
        precisions = get_precision_at_confidence(all_predictions, all_targets, class_id, confidence_thresholds)
        
        plt.plot(confidence_thresholds, precisions, label=class_name)
    
    
    total_precisions = get_total_precision_at_confidence(all_predictions, all_targets, confidence_thresholds)
    plt.plot(confidence_thresholds, total_precisions, label='Total', linewidth=2, color='black')
    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Precision')
    plt.title('Precision-Confidence Curves for All Classes')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    plt.grid(True)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig('precision_confidence_curves_all_classes.png')
    plt.close()
    print("Saved combined precision-confidence curves for all classes as 'precision_confidence_curves_all_classes.png'")

    
    print("Computing confusion matrix...")

    
    all_true_labels = []
    all_pred_labels = []

    
    iou_threshold = 0.5


    for pred, target in zip(all_predictions, all_targets):
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']

        
        conf_threshold = 0.5
        high_conf_indices = pred_scores >= conf_threshold
        pred_boxes = pred_boxes[high_conf_indices]
        pred_labels = pred_labels[high_conf_indices]
        pred_scores = pred_scores[high_conf_indices]

        # If there are no predictions or ground truths, continue
        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            continue

        matched_gt_indices = set()

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes)

            for i in range(len(pred_boxes)):
                iou_max, gt_idx = ious[i].max(0)
                if (iou_max >= iou_threshold and gt_idx.item() not in matched_gt_indices):

                    pred_label = pred_labels[i].item()
                    matched_gt_indices.add(gt_idx.item())
                    true_label = gt_labels[gt_idx].item()
                    all_true_labels.append(true_label)
                    all_pred_labels.append(pred_label)

                else:
                    all_true_labels.append(0)  # Background
                    all_pred_labels.append(pred_labels[i].item())

           
            for gt_idx in range(len(gt_boxes)):

                if gt_idx not in matched_gt_indices:
                    all_true_labels.append(gt_labels[gt_idx].item())
                    all_pred_labels.append(0)

        else:
           
            for i in range(len(pred_boxes)):
                all_true_labels.append(0) 
                all_pred_labels.append(pred_labels[i].item())
            for gt_idx in range(len(gt_boxes)):
                all_true_labels.append(gt_labels[gt_idx].item())
                all_pred_labels.append(0)

    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=range(num_classes))

    mask = (cm <= 1)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm.T, mask=mask.T, annot=True, fmt='d', cmap='Blues',
                xticklabels=['background'] + class_names,
                yticklabels=['background'] + class_names)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()
    print("Confusion matrix has been saved as 'confusion_matrix.png'.")

    # Visualize some predictions
    print("Visualizing results")
    visualize_predictions(
        model=model, 
        test_dataset=test_dataset, 
        device=device, 
        class_names=class_names,
        num_samples=5
    )

if __name__ == '__main__':
    main()