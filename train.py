
import os
from ultralytics import YOLO
from pathlib import Path
from utils.data_config import update_config, setup_parse
import datetime



def main(args):
    
    # check model 
    model = YOLO(args.weight)
       
    output_dir = Path(args.output_dir) / f"{datetime.date.today().isoformat()}/checkpoint_{args.model}_{args.directory_path}_s{args.imgsz}_b{args.batch}_e{args.epochs}"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(args.k_folds):
        # data_yaml = os.path.join("/content/datasets/fkolds", f"{datetime.date.today().isoformat()}/{args.directory_path}/fold_{i}/data_{i}.yaml")
        data_yaml = "/datasets/kfolds" + f"/{datetime.date.today().isoformat()}/{args.directory_path}/fold_{i}/data_{i}.yaml"
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