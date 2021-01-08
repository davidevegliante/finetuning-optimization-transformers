import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
from utils.argparser import create_train_argparse
from models import *
from utils.data import NewsDataset, create_data_loader
from utils.train import compute_report, train_model
import os

args = create_train_argparse()

strategy_class={
    'pooled':'Classifier_Pooled',
    'concat':'Classifier_Concat', 
    'avg':'Classifier_AVG'
}

# PARAMS
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
SAVED_MODEL_DIR = args.output_path
LR = args.lr
RANDOM_SEED = args.fixed_seed
MAX_SEQ_LEN=args.max_seq_len
BASE_TRANSFORMER = args.model_name
TUNING_STRATEGY= strategy_class[args.tuning_strategy] 
EVALUATE_TEST=args.evaluate_test_set

# set seed and deterministic operation
if RANDOM_SEED is not None: 
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.set_deterministic(True)


# Runtime device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("="*10,"Train info", "="*10)
print("Base Transformer Model: {}\nTuning Strategy: {}".format(BASE_TRANSFORMER, TUNING_STRATEGY))
print("Batch Size: {}\nEpochs: {}\nLR: {}".format(BATCH_SIZE, EPOCHS, LR))
print("Random Seed: {}".format(RANDOM_SEED))
print("device: {}\n".format(device))

# Load Data
train_data = pd.read_csv("data/desc_train_split.csv", header=0,names=['target','title','desc'])
val_data = pd.read_csv("data/desc_val_split.csv", header=0, names=['target','title','desc'])
test_data = pd.read_csv("data/desc_test_split.csv", header=0, names=['target','title','desc'])

print("="*10,"Data info", "="*10,)
print("Train example: {}\nVal Example {}\nTest Example {}".format(train_data.shape[0], val_data.shape[0], test_data.shape[0]))
print("Seq Len: {}\n".format(MAX_SEQ_LEN))

# create data loaders
print("Create DataLoaders...")
train_data_loader = create_data_loader(train_data, BASE_TRANSFORMER, MAX_SEQ_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(val_data, BASE_TRANSFORMER, MAX_SEQ_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_data, BASE_TRANSFORMER, MAX_SEQ_LEN, BATCH_SIZE)

# init model
print("\nInit Model...")
models_class_name=globals()[TUNING_STRATEGY]
model=models_class_name(BASE_TRANSFORMER, 4).to(device)

# Run training
out_dir_name=BASE_TRANSFORMER+TUNING_STRATEGY+str(EPOCHS)+'e'
best_model_path=os.path.join(SAVED_MODEL_DIR, out_dir_name, out_dir_name+'-best.bin')

print("\nStart training...")
train_history = train_model(model, 
                            train_data_loader,
                            val_data_loader,
                            EPOCHS,
                            device,
                            model_name=out_dir_name,
                            output_dir=SAVED_MODEL_DIR)

if EVALUATE_TEST:
    # load best model
    print("\nLoad best model...")
    best_state = torch.load(best_model_path)
    model.load_state_dict(best_state)
    print("Evaluating on test set...")
    compute_report(model, test_data_loader, device)
