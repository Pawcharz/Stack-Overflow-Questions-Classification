import torch
import numpy as np
from torch import nn
from transformers import AutoModelForSequenceClassification

class AutoCompositeModel(nn.Module):
  def __init__(self, device):
    super(AutoCompositeModel, self).__init__()
    
    self.device = device
    # self.model_content = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=64)
    # self.model_title = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=32)
    
    # self.lstm = nn.LSTM(128, 128, 3)
    
    self.transformerEncoder = nn.TransformerEncoderLayer(d_model=128, nhead=8, norm_first=True)
    self.test_model = nn.Sequential(
      nn.LayerNorm(128),
      nn.Linear(128, 256),
      nn.ReLU(),
      nn.Linear(256, 64),
      nn.ReLU(),
      nn.Linear(64, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 5)
    )
  
    # self.lstm = nn.LSTM(128, 256, 3)
    # self.test_model = nn.Sequential(
    #   nn.LayerNorm(256),
    #   nn.Linear(256, 512),
    #   nn.ReLU(),
    #   nn.Linear(512, 256),
    #   nn.ReLU(),
    #   nn.Linear(256, 64),
    #   nn.ReLU(),
    #   nn.Linear(64, 256),
    #   nn.ReLU(),
    #   nn.Linear(256, 128),
    #   nn.ReLU(),
    #   nn.Linear(128, 5)
    # )
    # self.my_new_layers = nn.Sequential(
    #   nn.Linear(128, 5),
    # )
  def forward(self, inputs):
    # x = self.lstm(x)[0]
    
    text_content = torch.tensor(inputs['content_input_ids']).to(self.device).float()
    x = self.transformerEncoder(text_content).float()    
    x = self.test_model(x)
    
    return x