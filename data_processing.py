import os
import pandas as pd
import cv2
import yaml
import shutil
from sklearn.model_selection import KFold
from utils.data_config import setup_parse, update_config
from utils.crop_image import convert_yolo_to_coco, convert_yolo_to_xy
from utils.preprocessing import yolo_to_coco
import datetime
import json


def process_directory(args):
    """
    Processes a directory containing image and txt files to create a dataframe.
    Args:
        directory_path: The path to the directory.
    Returns:
        A pandas DataFrame with columns 'xmax', 'ymax', 'xmin', 'ymin'.
    """
    data = []
    for filename in os.listdir(args.img_dir):
        # if filename.endswith(".jpg"):
        image_path = os.path.join(args.img_dir, filename)
        txt_path = os.path.join(args.txt_dir, filename.replace(".jpg", ".txt"))

        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            img_height, img_width = img.shape[:2]

        with open(txt_path, 'r') as f:
            for line in f:
                x_min, y_min, box_width, box_height, class_id = convert_yolo_to_coco(line.strip(), img_width, img_height)

                # if int(class_id) == 4:
                #     class_id = 0
                # elif int(class_id) == 5:
                #     class_id = 1
                # elif int(class_id) == 6:
                #     class_id = 2
                # elif int(class_id) == 7:
                #     class_id = 3

                # if args.drop_boxes is not None: # drop all box has width or height > drop_boxes size
                #     if box_width > args.drop_boxes or box_height > args.drop_boxes:
                #         continue
                    
                data.append([filename, x_min, y_min, box_width, box_height, class_id])

    df = pd.DataFrame(data, columns=['filename', 'xmin', 'ymin', 'box_w', 'box_h', "class"])
    return df

# split imgs and boxes to 2 folders
def split(args):
    
    for filename in os.listdir(args.src_dir):
        if filename.endswith(".jpg"):
            continue

        if filename.endswith(".txt"):
            save_path = args.txt_dir + "/" + filename
            with open(os.path.join(args.src_dir, filename), "r") as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.strip().split())
                    box_width = int(w * 1280)
                    box_height = int(h * 720)

                    if args.drop_boxes is not None:
                        if box_width > args.drop_boxes or box_height > args.drop_boxes:
                            continue

                    if cls == 4:
                        cls = 0
                    elif cls == 5:
                        cls = 1
                    elif cls == 6:
                        cls = 2
                    elif cls == 7:
                        cls = 3

                    line = f"{cls} {x} {y} {w} {h}\n"
                    
                    with open(save_path, "a", encoding='utf-8') as f:
                        f.write(line)
             # check if anno.txt dont have, dont save image           
            if not os.path.exists(save_path):
                continue
                
            shutil.copy(os.path.join(args.src_dir, filename.replace(".txt", ".jpg")), args.img_dir)


# datasets/kfolds/datetime/direc_path/fold-1
def create_kfolds(args):
    df = process_directory(args)
    results = df.groupby('filename')["class"].value_counts().unstack(fill_value=0)

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    kfolds = list(kf.split(results))

    save_path = os.path.join(args.kfolds_dir, f"{datetime.date.today().isoformat()}")
    os.makedirs(save_path, exist_ok=True)
    
    for i, (train_id, val_id) in enumerate(kfolds):
        train_paths = [os.listdir(args.img_dir)[j] for j in train_id]
        val_paths = [os.listdir(args.img_dir)[j] for j in val_id]

        split_dir = os.path.join(save_path, f"{args.directory_path}")  # datasets/kfolds/datetime/direct_path
        os.makedirs(split_dir, exist_ok=True)
    
        if args.mode == "yolo":
            train_imgs_dir = os.path.join(split_dir, f"fold_{i}/train/images") #./fold_i/train/images
            train_annos_dir = os.path.join(split_dir, f"fold_{i}/train/labels") # ./fold_i/train/labels
            os.makedirs(train_imgs_dir, exist_ok=True)
            os.makedirs(train_annos_dir, exist_ok=True)
                
            val_imgs_dir = os.path.join(split_dir, f"fold_{i}/val/images")
            val_annos_dir = os.path.join(split_dir, f"fold_{i}/val/labels")
            os.makedirs(val_imgs_dir, exist_ok=True)
            os.makedirs(val_annos_dir, exist_ok=True)
    
            
            for filename in train_paths:
                shutil.copy(os.path.join(args.img_dir, filename), train_imgs_dir)
                shutil.copy(os.path.join(args.txt_dir, filename.replace(".jpg", ".txt")), train_annos_dir)
                
            for filename in val_paths:
                shutil.copy(os.path.join(args.img_dir, filename), val_imgs_dir)
                shutil.copy(os.path.join(args.txt_dir, filename.replace(".jpg", ".txt")), val_annos_dir)
    
    
            #creat yaml file
            yaml_path = os.path.join(split_dir, f"fold_{i}/data_{i}.yaml")
            info = {
                "train": "train/images",
                "val": "val/images",
                "nc": 4,
                "names": ["motorbike", "automobile", "tourist car", "truck"]
            }
            with open(yaml_path, "w") as f:
                f.write(yaml.dump(info)) 
    
            #create coco format
        else:
            anno_dir = os.path.join(split_dir, f"fold_{i}") # ./fold_i/annotations
            os.makedirs(anno_dir, exist_ok=True)
            
            with open(f".datasets/coco/{args.directory_path}_coco_annotations.json", "r") as file:
                data = json.load(file)
    
            train_file = (os.path.join(anno_dir, "train_annotations.json"))
            train_data = {
                "images": [img for img in data["images"] if img["file_name"] in train_paths],
                "annotations": [item for item in data["annotations"] if item["image_id"]+".jpg" in train_paths],
                "categories": data["categories"]
            }
    
            val_file = (os.path.join(anno_dir, "val_annotations.json"))
            
            val_data = {
                "images": [img for img in data["images"] if img["file_name"] in val_paths],
                "annotations": [item for item in data["annotations"] if item["image_id"]+".jpg" in val_paths],
                "categories": data["categories"]
            }
    
            with open(train_file, 'w') as f:
                json.dump(train_data, f)
    
            with open(val_file, 'w') as f:
                json.dump(val_data, f)
       

if __name__ == "__main__":
    parser = setup_parse()

    args = parser.parse_args()
    args = update_config(args)

    os.makedirs(args.img_dir) if not os.path.exists(args.img_dir) else None
    os.makedirs(args.txt_dir) if not os.path.exists(args.txt_dir) else None 
    os.makedirs(args.kfolds_dir, exist_ok=True)
    #os.makedirs(os.path.join(args.kfolds_dir, args.directory_path), exist_ok=True)

    if args.mode == "coco":
        if not os.listdir("./datasets/coco"):
            yolo_to_coco()

    split(args)
    create_kfolds(args)
   






