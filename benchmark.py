import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import trange, tqdm
import numpy as np
from utils.onnx import *
from utils.data import create_data_loader
from models import *
from utils.argparser import create_benchmark_argparse

strategy_class={
    'pooled':'Classifier_Pooled',
    'concat':'Classifier_Concat', 
    'avg':'Classifier_AVG'
}

args = create_benchmark_argparse()

BASE_TRANSFORMER=args.model_name
TUNING_STRATEGY=strategy_class[args.tuning_strategy]
STATE_DICT_PATH=args.model_state_dict
OUTPUT_PATH=args.output_path
BATCH_SIZES=[1, 4, 8]

# Runtime device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data from parent directory
data = pd.read_csv("data/desc_val_split.csv", header=0, names=['target','title','desc'], nrows=100)

models_class_name=globals()[TUNING_STRATEGY]
model=models_class_name(BASE_TRANSFORMER, 4)

state = torch.load(STATE_DICT_PATH, map_location=device)
model.load_state_dict(state)

if device.type == 'cuda':
    benchmark_model_gpu(model, data, BATCH_SIZES, BASE_TRANSFORMER, OUTPUT_PATH, device, remove_dir=False, opset_version=11)    
else:
    benchmark_model_cpu(model, data, BATCH_SIZES, BASE_TRANSFORMER, OUTPUT_PATH, device, remove_dir=False, opset_version=11)

