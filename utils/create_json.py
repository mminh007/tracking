import datetime
import os
import json
import cv2


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

