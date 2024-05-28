# Load model  directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
import torch
import numpy as np
import transformers
import torch.nn.functional as F



class CustomModel:
  def __init__(self, model_path_binary, model_path_multi):
    self.model_path_binary = model_path_binary
    self.tokenizer_binary = AutoTokenizer.from_pretrained(model_path_binary)
    self.model_binary = AutoModelForSequenceClassification.from_pretrained(model_path_binary)

    self.model_path_multi = model_path_multi
    self.tokenizer_multi = AutoTokenizer.from_pretrained(model_path_multi)
    self.model_multi = AutoModelForSequenceClassification.from_pretrained(model_path_multi)

  def inference(self, text):
    encodings_binary = self.tokenizer_binary.encode_plus(
        text,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False,
        max_length=128,
        return_tensors='pt'
    )

    # text -> tokenizer -> 12343

    encodings_multi = self.tokenizer_multi.encode_plus(
        text,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False,
        max_length=128,
        return_tensors='pt'
    )
    
    

    with torch.no_grad():
        outputs_binary = self.model_binary(encodings_binary['input_ids'], encodings_binary['attention_mask'])
        outputs_multi = self.model_multi(encodings_multi['input_ids'], encodings_multi['attention_mask'])
    
    # 3 4 2 2 -> 0.23 0.54
     
    # Extract logits and concatenate them into a single list of six values
    logits_binary = outputs_binary.logits.squeeze().tolist()  # Assuming the output shape is [1, 2]
    logits_multi = outputs_multi.logits.squeeze().tolist()  # Assuming the output shape is [1, 4]
    
    return logits_binary + logits_multi # a list of six numbers (probs)







