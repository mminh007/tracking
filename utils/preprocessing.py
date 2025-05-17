import os
import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from collections import defaultdict
from utils.display import visualize_bboxes_distribution
from shutil import copyfile
from utils.labels import load_yolo_wh

MAPPING_CLASSES = {
    "motorcycle": 0,
    "car": 1,
    "bus": 2,
    "truck": 3,
}

def kmeans_anchors(boxes, k=9):
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    kmeans.fit(boxes)
    anchors = kmeans.cluster_centers_
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]  # sort by area
    return anchors

def scale_anchors(anchors, img_size):
    return (anchors * img_size).round().astype(int)

def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    union = box_area + cluster_area - intersection
    return intersection / union

def avg_iou(boxes, clusters):
    return np.mean([np.max(iou(box, clusters)) for box in boxes])

def compute_anchors(directory_path, k=3, img_size=640, plot_result=None):
    """
    Computes anchor boxes using k-means clustering on YOLO label data,
    scales them to the input image size, and reports the average IoU.

    Parameters:
        directory_path (str): Path to the directory containing YOLO label (.txt) files.
        k (int): Number of anchor boxes to generate (default is 3).
        img_size (int): Target image size used to scale anchor dimensions.
        plot (bool): If True, calls the visualization function to display boxes and anchors.

    Returns:
        None. Prints the scaled anchor boxes and the average IoU score.
    """

    boxes = load_yolo_wh(directory_path)
    anchors = kmeans_anchors(boxes, k=k)  
    scaled_anchors = scale_anchors(anchors, img_size)
    mean_iou = avg_iou(boxes, anchors)

    print(f"Anchors (k={k}, width x height):")
    for w, h in scaled_anchors:
        print(f"[{w}, {h}]")
    print(f"Average IoU: {mean_iou:.4f}")

    # plot
    if plot_result:
        visualize_bboxes_distribution(boxes=boxes, anchors=anchors, show_anchor=True)

    return scaled_anchors, mean_iou


def update_labels_in_folder(label_dir, mapping=None, dry_run=False):
    """
    Update class labels in YOLO-format annotation files within a given folder.

    This function scans all `.txt` files in the specified directory, 
    replaces the class IDs according to a provided mapping, and writes 
    the updated labels back to the files. Only lines with exactly five 
    elements (class_id, x_center, y_center, width, height) are considered valid.

    Args:
        label_dir (str): Path to the directory containing YOLO `.txt` label files.
        mapping (dict): Dictionary mapping old class IDs (int) to new class IDs (int).
        dry_run (bool, optional): If True, changes are not written to disk. 
                                  Useful for previewing changes. Defaults to False.

    Returns:
        None. Prints a summary of how many labels were updated across how many files.
    
    Example:
        mapping = {0: 1,
                   2: 3 }
        update_labels_in_folder("labels/", mapping, dry_run=True)
    """
    updated_count = 0
    total_files = 0
    
    if mapping is None:
        mapping = MAPPING_CLASSES

    for filename in os.listdir(label_dir):
        if not filename.endswith(".txt"):
            continue

        txt_path = os.path.join(label_dir, filename)
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            if cls_id in mapping:
                new_cls = mapping[cls_id]
                new_line = f"{new_cls} {' '.join(parts[1:])}\n"
                new_lines.append(new_line)

        if not dry_run:
            with open(txt_path, 'w') as f:
                f.writelines(new_lines)

        updated_count += len(new_lines)
        total_files += 1

    print(f"✅ Updated {updated_count} labels in {total_files} files.")

def create_kflods(
    directory_path: str,
    k_folds: int = 5,
    save_dir: str = "kfolds"
) -> None:
    """
    Create K-folds for YOLO dataset.

    Args:
        directory_path (str): Path to the directory containing YOLO label files.
        k_folds (int): Number of folds to create.
        drop_boxes (int): Number of boxes to drop from each fold.
        save_dir (str): Directory where K-folds will be saved.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    samples = sorted([f[:-4] for f in os.listdir(directory_path) if f.endswith('.jpg')])
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(samples)):
        fold_dir = os.path.join(save_dir, f'fold_{fold}')
        train_dir = os.path.join(fold_dir, 'train')
        val_dir = os.path.join(fold_dir, 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        for idxs, outdir in [(train_idx, train_dir), (val_idx, val_dir)]:
            for idx in idxs:
                base = samples[idx]
                for ext in ['.jpg', '.txt']:
                    src = os.path.join(directory_path, base + ext)
                    dst = os.path.join(outdir, base + ext)
                    if os.path.exists(src):
                        copyfile(src, dst)

    print("✅ Done splitting into 5 folds.")


def build_object_library(
    images_dir: str,
    labels_dir: str,
    class_map: dict,
    selected_classes: list,
    max_per_class: int = 50,
    save_dir: str = "object_lib",
    min_area: int = 1000
) -> dict:
    """
    Create an object library by cropping instances from images using YOLO annotations.

    Args:
        images_dir (str): Directory containing input images.
        labels_dir (str): Directory containing YOLO-format `.txt` annotation files.
        class_map (dict): Mapping from class names to class IDs (e.g., {'car': 0, 'truck': 1}).
        selected_classes (list): List of class names to extract objects from (e.g., ['car', 'truck']).
        max_per_class (int): Maximum number of cropped objects to save per class.
        save_dir (str): Directory where cropped objects are saved, organized by class.
        min_area (int): Minimum area (in pixels) for a bounding box to be accepted and cropped.

    Returns:
        dict: `object_library` dictionary with class IDs as keys and lists of cropped object images (numpy arrays) as values.
    """
    os.makedirs(save_dir, exist_ok=True)
    #object_library = defaultdict(list)
    counter = defaultdict(int)

    selected_ids = {class_map[name] for name in selected_classes}

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, x_c, y_c, w, h = map(float, parts)
                cls_id = int(cls_id)

                if cls_id not in selected_ids or counter[cls_id] >= max_per_class:
                    continue

                x1 = int((x_c - w / 2) * w_img)
                y1 = int((y_c - h / 2) * h_img)
                x2 = int((x_c + w / 2) * w_img)
                y2 = int((y_c + h / 2) * h_img)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)

                if (x2 - x1) * (y2 - y1) < min_area:
                    continue

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                cls_dir = os.path.join(save_dir, f'class_{cls_id}')
                os.makedirs(cls_dir, exist_ok=True)

                counter[cls_id] += 1
                filename = f"obj_{counter[cls_id]:04d}.jpg"
                save_path = os.path.join(cls_dir, filename)
                cv2.imwrite(save_path, crop)

    #             object_library[cls_id].append(crop)

    # return object_library

import os
import yaml

def create_data_yaml_for_kfolds(kfold_root="kfolds", class_names=None):
    """
        Automatically generates `data.yaml` files for each fold in a k-fold cross-validation setup
        for training YOLOv11 models.

        Args:
            kfold_root (str): Path to the root directory containing k-fold subfolders (e.g., kfold_0, kfold_1, ...).
                            Each subfolder should contain `train/` and `val/` directories.
            class_names (list of str, optional): List of class names. If not provided, defaults to
                                                ["truck", "car", "bus", "bike"].

        Functionality:
            - Iterates through folds named `kfold_0` to `kfold_4` inside the `kfold_root` directory.
            - For each fold, constructs absolute paths to the `train/` and `val/` directories.
            - Generates a corresponding `data.yaml` file for each fold in the format required by YOLOv11.
            - Each `data.yaml` file includes:
                - train: absolute path to training images/labels
                - val: absolute path to validation images/labels
                - nc: number of classes
                - names: list of class names

        Example usage:
            create_data_yaml_for_kfolds(kfold_root="kfolds", class_names=["truck", "car", "bus", "bike"])

        Output:
            Creates and saves `data.yaml` inside each `kfold_*` subdirectory.
    """
    if class_names is None:
        # Thay đổi nếu bạn có danh sách class khác
        class_names = ["truck", "car", "bus", "bike"]

    num_classes = len(class_names)

    for fold in range(5):
        fold_path = os.path.join(kfold_root, f"fold_{fold}")
        train_path = os.path.abspath(os.path.join(fold_path, "train"))
        val_path = os.path.abspath(os.path.join(fold_path, "val"))
        
        yaml_dict = {
            "train": train_path,
            "val": val_path,
            "nc": num_classes,
            "names": class_names
        }

        yaml_path = os.path.join(fold_path, "data.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_dict, f)

        print(f"✅ Created {yaml_path}")
