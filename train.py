import os
from ultralytics import YOLO
from pathlib import Path
from utils.data_config import update_config, setup_parse
import datetime
from utils.preprocessing import create_kflods, update_labels_in_folder, build_object_library, create_data_yaml_for_kfolds
from utils.data import get_dataloaders_kfold
# from torchmetrics.detection.mean_ap import MeanAveragePrecision
# import torch


# @torch.no_grad()
# def compute_map(model, dataloader, device="cuda"):
#     model.eval()
#     metric = MeanAveragePrecision()  # COCO-style AP

#     for images, targets_boxes, targets_labels in dataloader:
#         images = images.to(device)

#         # Forward
#         outputs = model(images)

#         # Ensure predictions are in expected format
#         # Format: List[Dict[boxes: Tensor[N, 4], scores: Tensor[N], labels: Tensor[N]]]
#         preds = []
#         for output in outputs:
#             pred_boxes = output["boxes"].detach().cpu()
#             pred_scores = output["scores"].detach().cpu()
#             pred_labels = output["labels"].detach().cpu()
#             preds.append({
#                 "boxes": pred_boxes,
#                 "scores": pred_scores,
#                 "labels": pred_labels
#             })

#         # Format targets
#         targets = []
#         for boxes, labels in zip(targets_boxes, targets_labels):
#             targets.append({
#                 "boxes": boxes.cpu(),
#                 "labels": labels.cpu()
#             })

#         # Update mAP metric
#         metric.update(preds, targets)

#     result = metric.compute()
#     return result



def main(args):


    update_labels_in_folder(args.directory_path)

    # create objects_Library
    class_map = {
    "motorcycle": 0,
    "car": 1,
    "bus": 2,
    "truck": 3,
    }

    build_object_library(images_dir=args.directory_path,
                                          labels_dir=args.directory_path,
                                          class_map=class_map,
                                          selected_classes=args.selected_classes,
                                          max_per_class=args.max_per_class,
                                          save_dir=args.objects_lib_dir,)

    if args.use_kfolds:
        # create kfolds
        try:
            create_kflods(args.directory_path, args.n_folds, args.kfolds_dir)

        except Exception as e:
            print(f"[ERROR] Failed to create dataloaders: {e}")    
        
        # get dataloaders
        try:
            get_dataloaders_kfold( base_dir=args.kfolds_dir,
                               num_folds=args.n_folds,
                               batch_size=args.batch,
                               num_workers=args.num_workers,
                               img_size=args.imgsz,
                               object_library=args.objects_lib_dir)
        except Exception as e:
            print(f"[ERROR] Failed to create dataloaders: {e}")
        
        try:
            create_data_yaml_for_kfolds(kfold_root=args.kflods_dir, class_names=args.class_name)
        except Exception as e:
            print(f"[ERROR] Failed to create data yaml: {e}")

    # check model 
    model = YOLO(args.weight)
       
    output_dir = Path(args.output_dir) / f"{datetime.date.today().isoformat()}/checkpoint_{args.model}_{args.directory_path}_s{args.imgsz}_b{args.batch}_e{args.epochs}"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(args.n_folds):
        # data_yaml = os.path.join("/content/datasets/fkolds", f"{datetime.date.today().isoformat()}/{args.directory_path}/fold_{i}/data_{i}.yaml")
        # data_yaml = "./kfolds" + f"/{datetime.date.today().isoformat()}/{args.directory_path}/fold_{i}/data_{i}.yaml"
        data_yaml = "./kfolds" + f"/fold_{i}/data.yaml"
        model.train(data=data_yaml,
                        epochs = args.epochs,
                        batch = args.batch,
                        imgsz = args.imgsz,
                        project = output_dir)
    
    

if __name__ == "__main__":
    parser = setup_parse()

    args = parser.parse_args()
    args = update_config(args)

    main(args)