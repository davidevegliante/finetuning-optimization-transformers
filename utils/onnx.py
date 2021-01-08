import onnx
import onnxruntime as ort    
from onnxruntime.quantization import QuantizationMode, quantize
import torch
import os
import onnx
from pathlib import Path
import pandas as pd
from tqdm import trange, tqdm
import time
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os
from utils.data import create_data_loader

def prepare_onnx_input(batch_data):
    """Prepare input to ONNX format

    Args:
        batch_data: Batch to process. It must contain input_ids and attention_mask

    Returns:
        dict: input_ids, attention_mask numpy matrices
    """
    return { 
        'input_ids': batch_data['input_ids'].numpy(),
        'attention_mask': batch_data['attention_mask'].numpy()
        }

def create_inference_session(onnx_model, options=None, verbose=False):
	"""Return an ONNX inference session

	Args:
			onnx_model (str): The onnx str path of the model
			options: Option to use for Inference Session. Defaults to None.
			verbose (bool, optional): Defaults to False.

	Returns:
			Inference Session
  """

	ort_session = ort.InferenceSession(onnx_model, sess_options=options)

	if verbose:
		print("ONNX runtime: ", ort.get_device())

	return ort_session
        
def export_ONNX_model(model, 
                      output_dir, 
                      X_batch,
                      input_names,
                      output_names,
                      device,
                      opset_version=11,
                      dynamic_axes={'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      verbose=False):
	"""Export PyTorch model to ONNX model

	Args:
			model: Model to export
			output_dir (str): Outputh path
			X_batch: Batch data to feed
			input_names: Names of the inputs
			output_names ([type]): Names of the ouput
			device: The platform to use
			opset_version (int, optional): Defaults to 11.
			dynamic_axes (dict, optional): Defaults to {'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'}, 'output': {0: 'batch_size'}}.
			verbose (bool, optional): Defaults to False.

	Returns:
			str: output path of the model
	"""

	# set the model to inference mode
	model.eval().to(device)

	os.makedirs(os.path.join(output_dir, 'onnx'), exist_ok=True)
	path = os.path.join(output_dir, 'onnx')

	# unpack data
	input_ids = X_batch["input_ids"].to(device)
	attention_mask = X_batch["attention_mask"].to(device)
	output = X_batch["targets"].to(device)

	if verbose:
		print("Input shape")
		print("-"*11)
		print(input_ids.shape, type(input_ids))
		print(attention_mask.shape, type(input_ids))
		print(output.shape, type(input_ids))
		
	# Export the base model
	torch.onnx.export(
			model,
			(input_ids, attention_mask),  # model input 
			os.path.join(path, 'model.onnx'),   # output path
			export_params=True,        # store the trained parameter
			opset_version=opset_version,  # ONNX version 
			do_constant_folding=True,  # whether to execute constant folding for optimization
			input_names = input_names,   # the model's input names
			output_names = output_names, # the model's output names
			dynamic_axes=dynamic_axes,
			verbose=verbose,

	)

	return os.path.join(path, 'model.onnx')

def compute_report(y_pred, y_pred_probs, y):
  print(classification_report(y, y_pred, target_names=['World', 'Sports', 'Business', 'Sci/Tech'], labels=np.unique(y)))

def get_predictions(model, test_data, batch_size, device):
  
  # eval mode
  model = model.eval().to(device)
  
  # init return list
  predictions = []
  prediction_probs = []
  real_values = []

  times = []
  with torch.no_grad():

    # iterate over batches
    t = tqdm(iter(test_data), leave=False, total=len(test_data))
    for batch in t:
      
      # send actual batch to device
      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      targets = batch["targets"].to(device)
      
      if len(input_ids) != batch_size:
        continue
      
      start_time = time.time()
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
      )
      elapsed_time = (time.time() - start_time) * 1000
      times.append(elapsed_time)

      # softmax and prediction
      _, preds = torch.max(outputs, dim=1)

      # append predictions, probabilities and targets
      predictions.extend(preds)
      prediction_probs.extend(outputs)
      real_values.extend(targets)

  #prediction_probs = torch.stack(prediction_probs).cpu()

  y_pred = torch.stack(predictions).cpu()
  y = torch.stack(real_values).cpu()  
  print(f'Batch mean execution time: {np.mean(times):.2f} ms')
  print(f"Accuracy: {accuracy_score(y, y_pred)}")

def get_onnx_predictions(ort_session, test_data, batch_size, device):
    
  # init return list
  predictions = []
  prediction_probs = []
  real_values = []
  times = []

  # iterate over batches
  t = tqdm(iter(test_data), leave=False, total=len(test_data))
  for batch in t:
    
    # send actual batch to device
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    targets = batch["targets"]
    
    if len(input_ids) != batch_size:
      continue
    
    io_binding = ort_session.io_binding()
    onnx_input=prepare_onnx_input(batch)

    io_binding.bind_cpu_input('input_ids', onnx_input['input_ids'])
    io_binding.bind_cpu_input('attention_mask', onnx_input['attention_mask'])
    io_binding.bind_output('output')

    start_time = time.time()
    ort_session.run_with_iobinding(io_binding)
    elapsed_time = (time.time() - start_time) * 1000
    times.append(elapsed_time)

    outputs_onnx = io_binding.copy_outputs_to_cpu()[0]
    # softmax and prediction
    _, preds = torch.max(torch.Tensor(outputs_onnx), dim=1)

    # append predictions, probabilities and targets
    predictions.extend(preds)
    prediction_probs.extend(torch.Tensor(outputs_onnx))
    real_values.extend(targets)
  
  y_pred = torch.stack(predictions).cpu()
  y = torch.stack(real_values).cpu()  
  print(f'Batch mean execution time: {np.mean(times):.2f} ms')
  print(f"Accuracy: {accuracy_score(y, y_pred)}")

def benchmark_model_gpu(model, data, batch_sizes, model_type, output_dir, device, remove_dir=True, opset_version=11, runtime_opts=None):
  
  print("BENCHMARK MODEL")
  print("-"*15)
  print("")

  import warnings
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # init dataloader
    data_loader=create_data_loader(data, model_type, 125, 1)
    batch=next(iter(data_loader))

    if runtime_opts:
      print("Runtime option inizializzati")

    print(f"Device: {device}\n")
    print("Esporto i modelli sul disco\n")

    os.makedirs(output_dir, exist_ok=True)

    # save pytorch model
    torch.save({'model': model},
               os.path.join(output_dir, 'pytorch.pt'))

    # export ONNX models (base, opt, quantized)
    onnx_base_model=export_ONNX_model(model,
                                  output_dir,
                                  batch,
                                  ['input_ids', 'attention_mask'],
                                  ['output'],
                                  device,
                                  opset_version=opset_version, 
                                  verbose=False)

    print(f"TORCH BASE: {get_file_size_in_bytes(os.path.join(output_dir, 'pytorch.pt')):.2f}MB")
    print(f"ONNX BASE: {get_file_size_in_bytes(onnx_base_model):.2f}MB")
    
    print("\nCreo le inference session")
    ort_session_base = create_inference_session(onnx_model=onnx_base_model, options=runtime_opts)

    for i in batch_sizes:
      print("")
      print(f"Batch size: {i}")
      print("-"*15)
      print("")

      # init dataloader
      data_loader=create_data_loader(data, model_type, 200, i)
      batch=next(iter(data_loader))

      # pure pytorch model
      print("Pure pytorch model")
      get_predictions(model, data_loader, i, device)
      print("")

      # ONNX model 
      print("ONNX model")
      get_onnx_predictions(ort_session_base, data_loader, i, device)
      print("")
      
      if remove_dir:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)

def benchmark_model_cpu(model, data, batch_sizes, model_type, output_dir, device, remove_dir=True, opset_version=11, runtime_opts=None):
  
  print("BENCHMARK MODEL")
  print("-"*15)
  print("")

  import warnings
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # init dataloader
    data_loader=create_data_loader(data, model_type, 125, 1)
    batch=next(iter(data_loader))

    if runtime_opts:
      print("Runtime option inizializzati")

    print(f"Device: {device}\n")
    print("Esporto i modelli sul disco\n")

    os.makedirs(output_dir, exist_ok=True)

    # save pytorch model
    torch.save({'model': model},
               os.path.join(output_dir, 'pytorch.pt'))

    # export ONNX models (base, opt, quantized)
    onnx_base_model=export_ONNX_model(model,
                                  output_dir,
                                  batch,
                                  ['input_ids', 'attention_mask'],
                                  ['output'],
                                  device,
                                  opset_version=opset_version, 
                                  verbose=False)

    onnx_quant_dep=quantize_model_dep(onnx_base_model)

    print(f"TORCH BASE: {get_file_size_in_bytes(os.path.join(output_dir, 'pytorch.pt')):.2f}MB")
    print(f"ONNX BASE: {get_file_size_in_bytes(onnx_base_model):.2f}MB")
    print(f"ONNX QUANTIZED: {get_file_size_in_bytes(onnx_quant_dep):.2f}MB")
    
    print("\nCreo le inference session")
    ort_session_base = create_inference_session(onnx_model=onnx_base_model, options=runtime_opts)
    ort_session_quant = create_inference_session(onnx_model=onnx_quant_dep, options=runtime_opts)

    for i in batch_sizes:
      print("")
      print(f"Batch size: {i}")
      print("-"*15)
      print("")

      # init dataloader
      data_loader=create_data_loader(data, model_type, 200, i)
      batch=next(iter(data_loader))

      # pure pytorch model
      print("Pure pytorch model")
      get_predictions(model, data_loader, i, device)
      print("")

      # ONNX model 
      print("ONNX model")
      get_onnx_predictions(ort_session_base, data_loader, i, device)
      print("")

      # ONNX QUANTIZE DEPRECATED 
      print("ONNX quantized (deprecated) model")
      get_onnx_predictions(ort_session_quant, data_loader, i, device)
      print("")
      
      if remove_dir:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)

def get_file_size_in_bytes(file_path):
	"""Get size of file at given path in bytes

	Args:
			file_path (str): File path

	Returns:
			Number: size in bytes
	"""
	size = os.path.getsize(file_path)
	return size / (1024**2)	

def quantize_model_dep(onnx_model):
  model = onnx.load(onnx_model)

  quantized_model = quantize(
      model=model,
      quantization_mode=QuantizationMode.IntegerOps,
      force_fusions=True,
      symmetric_weight=True,
  )

  path = Path(onnx_model)
  parent_path = path.parent
  output_path = os.path.join(parent_path, 'model-quant-dep.onnx')

  onnx.save_model(quantized_model, output_path)
  return output_path  

def quantize_model(onnx_model):
  import onnx
  from onnxruntime.quantization import quantize_dynamic, QuantType

  path = Path(onnx_model)
  parent_path = path.parent
  output_path = os.path.join(parent_path, 'model-dyn-quant.onnx')

  quantized_model = quantize_dynamic(
      onnx_model,
      output_path,
      weight_type=QuantType.QUInt8)

  return output_path