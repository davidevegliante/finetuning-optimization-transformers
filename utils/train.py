from tqdm import tqdm
import torch
from torch import nn
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import os
import pickle

def train_model(model,
                train_data,
                val_data,
                epochs,
                device,
                save_best_model=True,
                save_checkpoint=True,
                scheduler = True,
                lr = 2e-5,
                model_name = 'best_model',
                output_dir = '',
                save_dict=True,
                train_state_path = None):
  """Run the training phase

  Args:
      model (torch.nn.Module): The model to train
      train_data (torch.utils.data.DataLoader): Train dataloader
      val_data (torch.utils.data.DataLoader): Validation dataloader
      epochs (Number): Number of epochs
      device (torch.device): Training platform
      save_best_model (bool, optional): Save the best model (based on validation accuracy). Defaults to True.
      save_checkpoint (bool, optional): Save every model epochs. Defaults to True.
      scheduler (bool, optional): Use LR linear scheduler. Defaults to True.
      lr ([type], optional): Learning Rate. Defaults to 2e-5.
      model_name (str, optional): Output name of the model. Defaults to 'best_model'.
      output_dir (str, optional): Outpur directory. Defaults to ''.
      save_dict (bool, optional): Save dict history of training phase. Defaults to True.
      train_state_path (str, optional): State_dict path to restart training. Defaults to None.

  Returns:
      dict: Traning dict history
  """

  # init optimizer and lr scheduler
  total_steps = len(train_data) * epochs
  optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
  scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=0,
                                              num_training_steps=total_steps)

  # init history dict
  history = {
      'train_loss': [],
      'val_loss': [],
      'train_acc': [],
      'val_acc': []
  }
  
  best_accuracy = 0

  if train_state_path is not None:
    model, optimizer, scheduler, history, best_accuracy = load_checkpoint(model, optimizer, scheduler, train_state_path)

  # init loss function
  loss_fn = nn.CrossEntropyLoss().to(device)

  # config output directory and checkpoint directory
  if output_dir != '' and not os.path.exists(os.path.join(output_dir, model_name)):
    os.makedirs(os.path.join(output_dir, model_name))
    os.makedirs(os.path.join(output_dir, model_name, 'checkpoint'))
  
  # start training loop
  for epoch in range(epochs):

    print(f'Epoch {epoch + 1}/{epochs}')
    print('-' * 10)

    # training epoch
    train_acc, train_loss = run_train_epoch(model,
                                            train_data,    
                                            loss_fn, 
                                            optimizer, 
                                            device, 
                                            scheduler)

    print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)

    # evaluation
    if val_data is not None:
      val_acc, val_loss = run_eval(model,
                                  val_data,
                                  loss_fn, 
                                  device)

      print(f'Val loss {val_loss:.4f} accuracy {val_acc:.4f}')
      history['val_acc'].append(val_acc)
      history['val_loss'].append(val_loss)

    # save state dict of model with improved acc on val set
    if val_acc > best_accuracy:
      print('Val_acc improvement, saving model...')
      torch.save(model.state_dict(), os.path.join(output_dir, model_name, model_name+'-best.bin'))
      best_accuracy = val_acc

    if save_checkpoint:
      print('Saving model, optimizer and scheduler...')
      torch.save({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'scheduler': scheduler.state_dict(),
          'history': history,
          'best_accuracy': best_accuracy},
          os.path.join(output_dir, model_name, 'checkpoint', model_name+'.bin'))

    print()

  if save_dict:
    save_dict_as_pickle(history, os.path.join(output_dir, model_name, model_name+'_history.pkl'))

  return history

def run_train_epoch(model, 
                    train_data, 
                    loss_fn, 
                    optimizer, 
                    device, 
                    scheduler=None):
  """Training procedure of a single epoch

  Args:
      model (torch.nn.Module): The model to train
      train_data (torch.utils.data.DataLoader): Train dataloader
      loss_fn: Loss function to use 
      optimizer: Optimizer to use
      device (torch.device): Training platform
      scheduler (optional): Scheduler to use. Defaults to None.

  Returns:
      Number: accuracy score,
      Number: loss
  """

  # train mode
  model = model.train()
  model.to(device)
  
  losses = []
  y_target_list = []
  y_pred_list = []
  
  # iterate over batches 
  t = tqdm(iter(train_data), leave=False, total=len(train_data))
  for batch in t:

    # send actual batch to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    targets = batch["targets"].to(device)

    # forward pass
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    # compute loss and predictions
    loss = loss_fn(outputs, targets)
    _, preds = torch.max(outputs, dim=1)

    # append targets, predictions and loss
    y_target_list += targets.tolist()
    y_pred_list += preds.tolist()
    losses.append(loss.item())

    # backward
    loss.backward()

    # gradient clipping
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    if scheduler is not None:
      scheduler.step()

    optimizer.zero_grad()

  return accuracy_score(y_target_list, y_pred_list), np.mean(losses)

def run_eval(model, val_data, loss_fn, device):
  """ Do evalutation on data

  Args:
      model (torch.nn.Module): The model to train
      val_data (torch.utils.data.DataLoader): Validation dataloader
      loss_fn: Loss function to use 
      device (torch.device): Training platform

  Returns:
      Number: accuracy score,
      Number: loss
  """

  # eval mode
  model = model.eval()

  losses = []
  y_target_list = []
  y_pred_list = []


  with torch.no_grad():

    # iterate over batches 
    t = tqdm(iter(val_data), leave=False, total=len(val_data))
    for batch in t:
      
      # send actual batch to device
      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      targets = batch["targets"].to(device)

      # forward pass
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )

      # predictions and loss
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      y_target_list += targets.tolist()
      y_pred_list += preds.tolist()
      losses.append(loss.item())

  return accuracy_score(y_target_list, y_pred_list), np.mean(losses)

def get_predictions(model, test_data, device):
  """

  Args:
      model (torch.nn.Module): The model to train
      test_data (torch.utils.data.DataLoader): Test dataloader
      device (torch.device): Training platform

  Returns:
      (list): List of predictions,
      (list): List of prediction probs,
      (list): List of real y
  """

  # eval mode
  model = model.eval()
  
  # init return list
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():

    # iterate over batches
    t = tqdm(iter(test_data), leave=False, total=len(test_data))
    for batch in t:
      
      # send actual batch to device
      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      targets = batch["targets"].to(device)

      # forward pass
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      
      # prediction
      _, preds = torch.max(outputs, dim=1)

      # append predictions, probabilities and targets
      predictions.extend(preds)
      prediction_probs.extend(outputs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()  
  return predictions, prediction_probs, real_values

def save_dict_as_pickle(dict_to_save, pkl_path):
  """Serialize dict as pickle file

  Args:
      dict_to_save (dict): The target obj to serialize
      pkl_path (str): Output path
  """

  with open(pkl_path, 'wb') as f:
      pickle.dump(dict_to_save, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(pkl_path):
  """Deserialize as pickle file

  Args:
      pkl_path (str): Pickle file

  Returns:
      (dict): Deserialized obj
  """

  with open(pkl_path, 'rb') as f:
    return pickle.load(f)

def load_checkpoint(model, optimizer, scheduler, filename='checkpoint.bin'):
  """Load checkpoint to restart training
      
      Note: Input model & optimizer should be pre-defined.  This routine only updates their states.

  Args:
      model (torch.nn.Module): The model to update
      optimizer ([type]): The optimizer to update
      scheduler ([type]): The scheduler to update
      filename (str, optional): Name to use. Defaults to 'checkpoint.bin'.

  Returns:
      (torch.nn.Module): Updated model,
      Updated scheduler,
      (dict): Training history,
      (Number): Best accuracy in training history
  """
  start_epoch = 0

  if os.path.isfile(filename):
      print("Loading checkpoint '{}'".format(filename))
      checkpoint = torch.load(filename)
      start_epoch = checkpoint['epoch']
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      scheduler.load_state_dict(checkpoint['scheduler'])
      print("Done! Restart from Epoch {}. Best Val Accuracy: {}".format(checkpoint['epoch'], checkpoint['best_accuracy']))
      
      return model, optimizer, scheduler, checkpoint['history'], checkpoint['best_accuracy']

  else:
      print("No checkpoint found at '{}'".format(filename))
  
  return model, optimizer, scheduler, None, None

def compute_report(model, data, device):
  """Compute and print classification report

  Args:
      model (torch.nn.Module): The model to use
      data (torch.utils.data.DataLoader): Dataloader
      device (torch.device): Training platform
  """
  y_pred, y_pred_probs, y = get_predictions(model, data, device)
  print(classification_report(y, y_pred, target_names=['World', 'Sports', 'Business', 'Sci/Tech'], labels=np.unique(y)))
