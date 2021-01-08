from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch


class NewsDataset(Dataset):

  def __init__(self, X, targets, tokenizer, max_len):
    """ NewsDataset.
    Extends torch.utils.data.Dataset

    Args:
        X (pandas.DataFrame): Data features
        targets (pandas.DataFrame): Data targets 
        tokenizer (String): Transformer model name. It helps to download the model's tokenizer
        max_len (Number): Pad and truncate news to this length
    """

    self.X = X
    self.targets = targets

    if isinstance(tokenizer, str):
      self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    else:
      self.tokenizer = tokenizer

    self.max_len = max_len

  def __len__(self):
    return len(self.X)

  def __getitem__(self, item):
    
    news = str(self.X[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      news,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding='max_length',
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
      
    )

    return {
      'news': news,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

def create_data_loader(data, tokenizer, max_len, batch_size):
  """ Create and return Datadet and DataLoader

  Args:
      data (pandas.DataFrame): Data to use
      tokenizer (String): Transformer model name. It helps to download the model's tokenizer
      max_len (Number): Pad and truncate news to this length
      batch_size (Number): Size of every batch

  Returns:
      torch.utils.data.DataLoader: DataLoader
  """
  dataset = NewsDataset(
    X=data.desc.to_numpy(),
    targets=data.target.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    dataset,
    shuffle=False,
    batch_size=batch_size
  )