import os
import json
import cv2
import numpy as np
import datetime


def convert_yolo_to_xy(yolo_line, img_width, img_height):
    """
    Converts a YOLO format bounding box to xmin, ymin, xmax, ymax.

    Args:
        yolo_line: A string representing a YOLO format bounding box.
        img_width: The width of the image.
        img_height: The height of the image.

    Returns:
        A tuple containing (xmin, ymin, xmax, ymax).
    """

    class_id, x_center, y_center, width, height = map(float, yolo_line.split())
    xmin = int((x_center - width / 2) * img_width)
    ymin = int((y_center - height / 2) * img_height)
    xmax = int((x_center + width / 2) * img_width)
    ymax = int((y_center + height / 2) * img_height)

    return xmin, ymin, xmax, ymax, class_id

def convert_yolo_to_coco(yolo_line, img_width, img_height):
    class_id, x_center, y_center, width, height = map(float, yolo_line.split())
    x_min = int((x_center - width / 2) * width)
    y_min = int((y_center - height / 2) * height)

    box_width = int(width * img_width) 
    box_height = int(height * img_height)

    return x_min, y_min, box_width, box_height, class_id

def load_yolo_wh(label_folder):
    boxes = []
    for file_name in os.listdir(label_folder):
        if not file_name.endswith(".txt"):
            continue
        with open(os.path.join(label_folder, file_name), "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, _, _, w, h = map(float, parts)
                boxes.append([w, h])  # YOLO format: normalized width & height
    return np.array(boxes)

def yolo_to_coco(args):
    """
    Convert YOLO annotations to COCO format.

    Parameters:
    - directory_path: Directory containing YOLO annotation files and images.
    - output_json: Path to save the output COCO JSON file.
    - categories: List of category names in the format [{'id': int, 'name': str}].

    Returns:
    - None, saves JSON to `output_json`.
    """

    # COCO JSON structure
    coco_data = {
        "info": {"description": "Converted YOLO to COCO format",
                 "date_created": datetime.date.today().isoformat()},
                 "total samples": None,
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "motorbike"},
                       {"id": 1, "name": "automobile"},
                       {"id": 2, "name": "tourist car"},
                       {"id": 3, "name": "truck"}]
    }

    annotation_id = 0

    # Iterate over YOLO annotation files
    for filename in os.listdir(args.src_dir):
        if filename.endswith(".txt"):
            txt_path = os.path.join(args.src_dir, filename)
            image_path = os.path.join(args.src_dir, filename.replace(".txt", ".jpg"))
            img_id = image_path.split("/")[-1].split(".jpg")[0]

        # Image information
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                height, width = img.shape[:2]
    
            # Read YOLO annotation file
            with open(txt_path, 'r') as file:
                for line in file:
                    # Parse YOLO line
                    class_id, x_center, y_center, w, h = map(float, line.strip().split())
                    
                    # Convert to COCO format
                    x_min = int((x_center - w / 2) * width)
                    y_min = int((y_center - h / 2) * height)
    
                    box_width = int(w * width)
                    box_height = int(h * height)
    
                    if args.drop_boxes:
                      if box_width > args.drop_boxes or box_height > args.drop_boxes:
                        continue
    
                    if int(class_id) == 4:
                      class_id = 0
                    elif int(class_id) == 5:
                      class_id = 1
                    elif int(class_id) == 6:
                      class_id = 2
                    elif int(class_id) == 7:
                      class_id = 3
    
                    # Add annotation to COCO structure
                    coco_data["annotations"].append({
                      "id": annotation_id,
                      "image_id": img_id,
                      "category_id": int(class_id),
                      "bbox": [x_min, y_min, box_width, box_height],
                      "area": box_width * box_height,
                      "iscrowd": 0
                    })
    
                    annotation_id += 1
            
            # if if images dont have bbox
            if coco_data["annotations"][-1]["image_id"] == coco_data["images"][-1]["id"]:
               continue
    
            # Add image to COCO structure
            coco_data["images"].append({
                "id": img_id,
                "file_name": f"{img_id}.jpg",
                "width": width,
                "height": height
            })
            
        
    # Save to JSON
    coco_data["info"]["total samples"] = len(coco_data["images"])
    
    save_path = os.path.join(args.dataset_dir, "coco")
    os.makedirs(save_path, exist_ok=True)

    output_json = os.path.join(save_path, f"{args.directory_path}_coco_annotations.json")
    with open(output_json, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)

    print(f"Conversion complete. COCO annotations saved to {output_json}.")

def convert_yolo_to_abs(box, img_w, img_h):
    # Format: x_center, y_center, width, height (normalized)
    x, y, w, h = map(float, box)
    abs_x = x * img_w
    abs_y = y * img_h
    abs_w = w * img_w
    abs_h = h * img_h
    x1 = abs_x - abs_w / 2
    y1 = abs_y - abs_h / 2
    x2 = abs_x + abs_w / 2
    y2 = abs_y + abs_h / 2
    return [x1, y1, x2, y2]

def convert_abs_to_yolo(box, img_w, img_h):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    x_center = x1 + w / 2
    y_center = y1 + h / 2
    return [x_center / img_w, y_center / img_h, w / img_w, h / img_h]


