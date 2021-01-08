
import pandas as pd
from bs4 import BeautifulSoup
from html import unescape
import re
from sklearn.model_selection import train_test_split

RANDOM_SEED=42

def count_item_with_no_description(dataframe):
  no_title = (dataframe['title'].values == '').sum()
  no_desc = (dataframe['desc'].values == '').sum()

  return (no_title, no_desc)

def remove_html_tags(text):
  soup = BeautifulSoup(unescape(text), 'lxml')
  return soup.text.replace('quot;', ' ')

def remove_special_characters(text):
  return re.sub('[^a-zA-Z.\d\s\']', ' ', text)
  
def remove_extra_whitespace_tabs(text):
    pattern = r'^\s*|\s\s*'
    return re.sub(pattern, ' ', text).strip()

def remove_URL(text):
    return re.sub(r"http\S+", "", text)

def normalize_text(text):
  text = remove_html_tags(text)
  text = remove_special_characters(text)
  text = remove_extra_whitespace_tabs(text)
  text = remove_URL(text)
  
  return text

# read data
train_data = pd.read_csv("train.csv", header=0,names=['target','title','desc'])
test_data = pd.read_csv("test.csv", header=0, names=['target','title','desc'])

print("Train shape: {}; Test shape: {}".format(train_data.shape, test_data.shape) )

# count empty item
print("Empty title and description in train set", count_item_with_no_description(train_data))
print("Empty title and description in test set", count_item_with_no_description(test_data))
train_data.title = train_data.title.apply(normalize_text)

print("\nApply normalization...")
train_data.desc = train_data.desc.apply(normalize_text)
test_data.title = test_data.title.apply(normalize_text)
test_data.desc = test_data.desc.apply(normalize_text)

# shift target from 1 to 4 to 0 to 3
print("Shift target value...")
train_data.target = train_data.target.apply(lambda x: x - 1)
test_data.target = test_data.target.apply(lambda x: x - 1)

train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=RANDOM_SEED, stratify=train_data.target)

print("Done!\n")
print('Train set shape: ', train_data.shape)
print('Validation set shape: ', val_data.shape)
print('Test set shape: ', test_data.shape)

# Save split on disk
train_data.to_csv('desc_train_split.csv')
val_data.to_csv('desc_val_split.csv') 
test_data.to_csv('desc_test_split.csv')    
