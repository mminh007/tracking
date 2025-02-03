import yaml
import os
import argparse

def setup_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--weight", type=str)
    parser.add_argument("--cfg-file", type=str)
    
    parser.add_argument("--dataset-dir", type=str)
    parser.add_argument("--src-dir", type=str)
    parser.add_argument("--img-dir", type=str)
    parser.add_argument("--txt-dir", type=str)

    parser.add_argument("--kfolds-dir", type=str)
    parser.add_argument("--directory-path", type=str)
    parser.add_argument("--k-folds", type=int)
    parser.add_argument("--drop-boxes", type=int)

    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--epochs",type=int)

    parser.add_argument("--output-dir", type=str)
    
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