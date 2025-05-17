import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


class NighttimeDataset(Dataset):
    """
        Custom dataset for loading and augmenting nighttime images with optional Copy-Paste augmentation.

        Args:
            img_dir (str): Path to the directory containing images and their YOLO-format label files.
            object_lib (str, optional): Path to object library for Copy-Paste augmentation. If None, no Copy-Paste applied.
            img_size (int): Image resize target (width and height).
            transform (albumentations.Compose, optional): Albumentations transform pipeline applied to images and labels.

        Attributes:
            img_files (list): List of image filenames.
            object_lib (dict): Dictionary mapping class_id to list of object crops.
    """

    def __init__(
                self,
                img_dir: str,
                object_lib: str = None,
                img_size: int = 640,
                transform: A.Compose = None
            ) -> None:
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.img_size = img_size
        self.object_lib = self._load_object_lib(object_lib) if object_lib else {}

        self.transform = transform

        self.to_tensor = A.Compose([ToTensorV2()])

    def _load_object_lib(self, obj_dir):
        """
            Load the object library from directory, structured as class-wise subfolders of cropped images.

            Args:
                obj_dir (str): Path to the object library directory.

            Returns:
                dict: A dictionary with class_id (int) as keys and list of object images (np.ndarray) as values.
        """
        obj_dict = {}
        for class_folder in os.listdir(obj_dir):
            cls_id = int(class_folder.replace('class_', ''))
            path = os.path.join(obj_dir, class_folder)
            obj_dict[cls_id] = [cv2.imread(os.path.join(path, f)) for f in os.listdir(path) if f.endswith('.jpg')]
        return obj_dict

    def _load_yolo_labels(self, txt_path: str) -> tuple[list[list[float]], list[int]]:
        """
            Load bounding boxes and class labels from a YOLO-format text file.

            Args:
                txt_path (str): Path to the label file.

            Returns:
                tuple:
                    - list of bounding boxes in YOLO format [x_center, y_center, width, height].
                    - list of class IDs (int).
        """
        boxes, labels = [], []
        if not os.path.exists(txt_path):
            return boxes, labels
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5: continue
                cls, x, y, w, h = map(float, parts)
                boxes.append([x, y, w, h])
                labels.append(int(cls))
        return boxes, labels

    def _copy_paste(
        self,
        img: np.ndarray,
        bboxes: list[list[float]],
        labels: list[int]
        ) -> tuple[np.ndarray, list[list[float]], list[int]]:
        """
        Apply Copy-Paste augmentation using pre-cropped objects from object_lib.

        Args:
            img (np.ndarray): Input image.
            bboxes (list): Existing bounding boxes in YOLO format.
            labels (list): Corresponding class labels.

        Returns:
            tuple:
                - Augmented image (np.ndarray).
                - Combined bounding boxes after augmentation.
                - Combined class labels after augmentation.
        """
        h, w, _ = img.shape
        new_boxes, new_labels = [], []
        for cls_id, obj_list in self.object_lib.items():
            if np.random.rand() < 0.3:
                obj = random.choice(obj_list)
                oh, ow = obj.shape[:2]
                scale = np.random.uniform(0.3, 0.7)
                new_w, new_h = int(ow * scale), int(oh * scale)
                obj_resized = cv2.resize(obj, (new_w, new_h))

                px = np.random.randint(0, w - new_w)
                py = np.random.randint(0, h - new_h)

                img[py:py + new_h, px:px + new_w] = obj_resized

                cx = (px + new_w / 2) / w
                cy = (py + new_h / 2) / h
                bw = new_w / w
                bh = new_h / h

                new_boxes.append([cx, cy, bw, bh])
                new_labels.append(cls_id)
        return img, bboxes + new_boxes, labels + new_labels

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[np.ndarray, list[list[float]], list[int]]:
        """
            Get an augmented image and corresponding bounding boxes and labels.

            Args:
                idx (int): Index of the sample.

            Returns:
                tuple:
                    If transform is applied:
                        - torch.Tensor: Transformed image.
                        - torch.Tensor: Bounding boxes.
                        - torch.Tensor: Class labels.
                    Otherwise:
                        - np.ndarray: Raw image.
                        - list: YOLO bounding boxes.
                        - list: Class labels.
        """

        img_name = self.img_files[idx]
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.img_dir, img_name)
        txt_path = os.path.join(self.img_dir, base + '.txt')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, labels = self._load_yolo_labels(txt_path)

        if self.object_lib:
            image, bboxes, labels = self._copy_paste(image, bboxes, labels)

        if len(bboxes) == 0:
            bboxes = [[0.5, 0.5, 0.1, 0.1]]
            labels = [0]
        if self.transform:
          transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)
          tensor_img = self.to_tensor(image=transformed['image'])['image']
          target_boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
          target_labels = torch.tensor(transformed['class_labels'], dtype=torch.long)
          return tensor_img, target_boxes, target_labels

        else:
          return image, bboxes, labels


def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable number of bounding boxes per image.

    Args:
        batch (list): List of tuples in the form (image_tensor, bbox_tensor, label_tensor).

    Returns:
        Tuple:
            - torch.Tensor: Stacked image tensors of shape (B, C, H, W).
            - list[torch.Tensor]: List of bounding box tensors per image.
            - list[torch.Tensor]: List of label tensors per image.
    """

    images = []
    boxes = []
    labels = []

    for img, box, label in batch:
        images.append(img)
        boxes.append(box)
        labels.append(label)

    images = torch.stack(images, dim=0)
    return images, boxes, labels


def get_dataloaders_kfold(
    base_dir: str = "/content/nighttime_split",
    num_folds: int = 5,
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: int = 640,
    object_library: str = None
) -> list:
    """
        Returns a list of (train_loader, val_loader) pairs for each fold in k-fold cross-validation.

        Args:
            base_dir (str): Root directory containing fold subdirectories named "fold_0", "fold_1", etc.
            num_folds (int): Number of cross-validation folds.
            batch_size (int): Number of samples per batch to load.
            num_workers (int): Number of subprocesses for data loading.
            img_size (int): Image size to resize input images to.
            object_library (str): Path to object library directory for Copy-Paste augmentation in training.

        Returns:
            list: A list of tuples, each containing (train_loader, val_loader) for one fold.
    """

    fold_loaders = []

    for fold in range(num_folds):
        fold_path = f"{base_dir}/fold_{fold}"

        # Augmentations for training
        train_transform = A.Compose([
            A.CLAHE(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(p=0.3),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.4),
            A.RandomResizedCrop(size=(img_size,img_size), scale=(0.95, 1.0), p=0.5),
            A.Resize(img_size, img_size),
            ],
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']),
            additional_targets={'image_cp': 'image'})

        val_transform = transforms.Compose([
            transforms.CenterCrop(576),
        ])

        train_dataset = NighttimeDataset(
            img_dir=f"{fold_path}/train",
            transform=train_transform,
            img_size=img_size,
            object_lib=object_library,
        )

        val_dataset = NighttimeDataset(
            img_dir=f"{fold_path}/val",
            transform=val_transform,
            img_size=img_size,
            object_lib=None,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn)

        fold_loaders.append((train_loader, val_loader))

    return fold_loaders
