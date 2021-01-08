import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

class Classifier_Pooled(nn.Module):

  def __init__(self, pre_trained_model_name, n_classes, dropout_p=0.3):
    super(Classifier_Pooled, self).__init__()

    print("Init",pre_trained_model_name, "model with Pooled strategy")
    self.transformer = AutoModel.from_pretrained(pre_trained_model_name)
    self.drop = nn.Dropout(p=dropout_p)
    self.out = nn.Linear(self.transformer.config.hidden_size, n_classes)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input_ids, attention_mask):
    
    _, pooled_output = self.transformer(
      input_ids=input_ids,
      attention_mask=attention_mask,
      return_dict=False
    )
    
    output = self.drop(pooled_output)
    output = self.out(output)

    return self.softmax(output)
    
class Classifier_Concat(nn.Module):
 
  def __init__(self, pre_trained_model_name, n_classes, dropout_p=0.3, n_layers=4):
    super(Classifier_Concat, self).__init__()

    self.last_layers = n_layers
    print("Init", pre_trained_model_name, "model with Concat strategy")
    self.transformer = AutoModel.from_pretrained(pre_trained_model_name)
    self.drop = nn.Dropout(p=dropout_p)
    self.out = nn.Linear(self.transformer.config.hidden_size * n_layers, n_classes)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input_ids, attention_mask):
   
    _, pooled_output, hidden_states = self.transformer(
      input_ids=input_ids,
      attention_mask=attention_mask,
      output_hidden_states=True,
      return_dict=False

    )
    
    # hidden_state = tuple of torch.FloatTensor (output embeddings, output of each layer)
    # get only output of layers (batch_size, sequence_length, hidden_size)
    output = []
    for i in reversed(range(len(hidden_states)-self.last_layers, len(hidden_states), 1)):
      
      # select CLS token hidden layer
      output.append(hidden_states[i][:, 0, :])

    output = torch.cat(output, 1)

    output = self.drop(output)
    output = self.out(output)

    return self.softmax(output)   

class Classifier_AVG(nn.Module):

  def __init__(self, pre_trained_model_name, n_classes, dropout_p=0.3, n_layers=4):
    super(Classifier_AVG, self).__init__()

    print("Init", pre_trained_model_name, "model with AVG strategy")
    self.last_layers = n_layers
    self.transformer = AutoModel.from_pretrained(pre_trained_model_name)
    self.drop = nn.Dropout(p=dropout_p)
    self.out = nn.Linear(self.transformer.config.hidden_size, n_classes)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input_ids, attention_mask):

    _, pooled_output, hidden_states = self.transformer(
      input_ids=input_ids,
      attention_mask=attention_mask,
      output_hidden_states=True,
      return_dict=False

    )
    
    # hidden_state = tuple of torch.FloatTensor (output embeddings, output of each layer)
    # get only output of layers (batch_size, sequence_length, hidden_size)
    output = []
    for i in reversed(range(len(hidden_states)-self.last_layers, len(hidden_states), 1)):
      
      # select CLS token hidden layer
      output.append(hidden_states[i][:, 0, :])

    output = torch.mean(torch.stack(output), 0)
    
    output = self.drop(pooled_output)
    output = self.out(output)

    return self.softmax(output)
    
