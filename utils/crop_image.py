import os
import cv2
from tqdm import tqdm
from labels import convert_yolo_to_abs, convert_abs_to_yolo


def bbox_iou_area(original, cropped):
    """
    Calculate the intersection-over-area (IoA) ratio between two bounding boxes.

    Args:
        original (list or tuple): The original bounding box in [x1, y1, x2, y2] format.
        cropped (list or tuple): The cropped/intersecting bounding box in [x1, y1, x2, y2] format.

    Returns:
        float: The ratio of the intersection area over the area of the original bounding box (range: 0.0 to 1.0).
    """

    inter_x1 = max(original[0], cropped[0])
    inter_y1 = max(original[1], cropped[1])
    inter_x2 = min(original[2], cropped[2])
    inter_y2 = min(original[3], cropped[3])

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    orig_area = (original[2] - original[0]) * (original[3] - original[1])
    return inter_area / orig_area

def crop_and_save(img, boxes, patch_x, patch_y, patch_idx, img_name, out_dir, class_counts_patch, img_size, PATCH):
    """
    Crop a patch from the original image and extract bounding boxes that significantly overlap with the patch.
    Saves the cropped image and the corresponding YOLO-format label file.

    Args:
        img (np.ndarray): The original image.
        boxes (list): List of tuples in the form (class_id, [x_center, y_center, width, height]) in YOLO format.
        patch_x (int): Top-left x-coordinate of the patch.
        patch_y (int): Top-left y-coordinate of the patch.
        patch_idx (int): Index of the current patch (used for naming).
        img_name (str): Filename of the original image.
        out_dir (str): Output directory to save patched images and label files.
        class_counts_patch (dict): Dictionary to count the number of bboxes per class after patching.
        img_size (tuple): Size of the original image (width, height).
        PATCH (tuple): Size of each patch (width, height).

    Returns:
        None. Saves the patch image and label file to the specified output directory.
    """

    IMG_WIDTH, IMG_HEIGHT = img_size
    PATCH_W, PATCH_H = PATCH
    patch_boxes = []

    for cls, box in boxes:
        x1, y1, x2, y2 = convert_yolo_to_abs(box, IMG_WIDTH, IMG_HEIGHT)

        # Calculate the intersection between the bbox and the patch
        inter_x1 = max(x1, patch_x)
        inter_y1 = max(y1, patch_y)
        inter_x2 = min(x2, patch_x + PATCH_W)
        inter_y2 = min(y2, patch_y + PATCH_H)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            continue  # No intersection

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        bbox_area = (x2 - x1) * (y2 - y1)
        inter_ratio = inter_area / bbox_area

        if inter_ratio >= 0.7:
            # Crop the remaining bbox within the patch
            new_x1 = inter_x1 - patch_x
            new_y1 = inter_y1 - patch_y
            new_x2 = inter_x2 - patch_x
            new_y2 = inter_y2 - patch_y
            new_box = convert_abs_to_yolo([new_x1, new_y1, new_x2, new_y2], PATCH_W, PATCH_H)
            patch_boxes.append((cls, new_box))
            class_counts_patch[cls] += 1

    patch_img = img[patch_y:patch_y+PATCH_H, patch_x:patch_x+PATCH_W]

    base_name, ext = os.path.splitext(img_name)
    patch_img_name = f"{base_name}_patch{patch_idx}{ext}"
    patch_label_name = f"{base_name}_patch{patch_idx}.txt"

    cv2.imwrite(os.path.join(out_dir, "images", patch_img_name), patch_img)

    with open(os.path.join(out_dir, "labels", patch_label_name), "w") as f:
        for cls, box in patch_boxes:
            f.write(f"{cls} {' '.join([str(round(b, 6)) for b in box])}\n")


def split_dataset(path_dir, out_dir, img_size=(1280,720), patch_size=(2,2)):

    """
    Split all images and their YOLO-format annotations in a directory into smaller patches.
    Only bounding boxes with >= 70% overlap with a patch are retained and converted accordingly.

    Args:
        path_dir (str): Directory containing original images and YOLO `.txt` label files.
        out_dir (str): Output directory to save patched images and labels.
        img_size (tuple, optional): Size of the original images (width, height). Defaults to (1280, 720).
        patch_size (tuple, optional): Number of horizontal and vertical splits (cols, rows). Defaults to (2, 2).

    Returns:
        None. Outputs patched images and updated labels, and prints summary statistics.
    """

    PATCH_W, PATCH_H = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
    PATCHES = [(0, 0), (PATCH_W, 0), (0, PATCH_H), (PATCH_W, PATCH_H)]

    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)

    image_files = [f for f in os.listdir(path_dir) if f.endswith(".jpg") or f.endswith(".png")]
    total_orig_bboxes = 0
    total_new_bboxes = 0
    class_counts_patch = {}

    for img_file in tqdm(image_files):
        img_path = os.path.join(path_dir, img_file)
        label_path = os.path.join(path_dir, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))
        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        with open(label_path) as f:
            lines = f.readlines()

        boxes = []
        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])
            if cls not in class_counts_patch:
                class_counts_patch[cls] = 0
            box = list(map(float, parts[1:]))
            boxes.append((cls, box))
        total_orig_bboxes += len(boxes)

        for idx, (patch_x, patch_y) in enumerate(PATCHES):
            crop_and_save(img, boxes, patch_x, patch_y, idx, img_file, out_dir, class_counts_patch, img_size, (PATCH_W, PATCH_H))

    total_new_bboxes = sum(class_counts_patch.values())

    print("\n📊 Summary::")
    print(f"🔸 Original total bboxes: {total_orig_bboxes}")
    print(f"🔸 Total bbboxes after patching: {total_new_bboxes}")
    print(f"📉 Lost: {total_orig_bboxes - total_new_bboxes} bbox ({round((1 - total_new_bboxes / total_orig_bboxes) * 100, 2)}%)")
    print("🔹 Per-class count after patching:")
    for k, v in class_counts_patch.items():
        print(f"  - Class {k}: {v} bbox")



