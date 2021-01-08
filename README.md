# Sequence Classification Fine-Tuning and 8bit quantization
Based on 🤗 [HuggingFace Transformers](https://huggingface.co/) and [ONNX](https://onnx.ai/).

Sequence Classification task.\
Dataset [AG News Classification](https://www.kaggle.com/amananandrai/ag-news-classification-dataset)

The script was used to test and compare the performance of BERT<sub>BASE</sub> and RoBerta<sub>BASE</sub> on a classification task. Feel free to try it on others HuggingFace model using `model_name` param. It will automatically download and instantiate the pre-trained model and its tokenizer.

The script `benchmark.py` can be used to export the model to ONNX file format. It will automatically export the original model and a 8bit quantized version of it. After that it will benchmark them with the original PyTorch.

## Fine-tuning Strategies
3 fine-tuning strategies

* Pooler output;
* AVG of [CLS] token's last 4 layers
* Concat of [CLS] token's last 4 layers

## Quantization
In a very simplified manner, the quantization allows to reduce the size of the model and speedup the inference by transforming the model weights from `float32` to `int8`. More information [here](https://medium.com/microsoftazure/faster-and-smaller-quantized-nlp-with-hugging-face-and-onnx-runtime-ec5525473bb7) and here [https://pytorch.org/docs/stable/quantization.html].

In this repo Dynamic Quantization is applied. 

# Getting Started
The files `train.py` and `benchmark.py` are responsible to do the job. 

## Dependencies
* PyTorch >= 1.6.0
* Transformers >= 3.3.1
* onnx, onnxruntime==1.4.0, onnxruntime-tools
* beautifulsoup4, lxml

## Pre-process data
The data is already in /data.\
Before doing the training step do the pre-processing step using the command down below.

`python3 data/pre_process.py`

Otherwise open the file and apply your own pre-processing step.

## Training

`usage: train.py [-h] --model_name MODEL_NAME --tuning_strategy {pooled,concat,avg} [--output_path OUTPUT_PATH]
                [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--lr LR] [--fixed_seed FIXED_SEED]
                [--max_seq_len MAX_SEQ_LEN] [--evaluate_test_set EVALUATE_TEST_SET]`

Example of Fine-Tuning BERT base with avg strategy.

`python3 train.py --model_name bert-base-uncased --tuning_strategy avg
`

## Quantization and Benchmark

`usage: benchmark.py [-h] --model_name MODEL_NAME --tuning_strategy {pooled,concat,avg} --model_state_dict
                    MODEL_STATE_DICT --output_path OUTPUT_PATH`

Example of ONNX conversion, quantization and benchmark

`python3 benchmark.py  --model_name bert-base-uncased --tuning_strategy avg --model_state_dict models/bert-base-uncasedClassifier_AVG4e/bert-base-uncasedClassifier_AVG4e-best.bin --output_path opt`

