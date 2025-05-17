import yaml
import os
import argparse

def setup_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--weight", type=str)
    parser.add_argument("--cfg-file", type=str)
    
    parser.add_argument("--directory-path", type=str)
    parser.add_argument("--max-per-class", type=int)

    parser.add_argument("--n-folds", type=int)

    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--epochs",type=int)

    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--class-name", nargs='+', type=str, default=['motorcycle', 'car', 'bus', 'truck'])
    parser.add_argument("--selected_classes", nargs='+', type=str, default=["bus", "truck"])
    parser.add_argument("--objects-lib-dir", type=str, default="objects_lib")
    parser.add_argument("--use-kfolds", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--kfolds-dir", type=str, default="kfolds")
    parser.add_argument("--debug", action="store_false")

    
    return parser

def update_config(args: argparse.Namespace):
    if not args.cfg_file:
        return args
    
    cfg_path = args.cfg_file + ".yaml" if not args.cfg_file.endswith(".yaml") else args.cfg_file

    with open(cfg_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    
    for key, value in data.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    # config_args = argparse.Namespace(**data)
    # args = parser.parse_args(namespace=config_args)
    return args